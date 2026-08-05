// Reading Q/K/V and hidden states back as words.
//
// Q, K and V live in their own spaces — a Query vector cannot be compared
// against an embedding directly. What *is* valid is to push the whole
// vocabulary through the same projection and compare inside that space, which
// is what projectVocabulary does.

import { gelu, layerNorm, multiplyVectorByFlatMatrix } from "./linalg.js";

// Printable ASCII plus accented Latin letters (café, naïve), optionally as a
// "##" piece. Rare Unicode punctuation such as "›" is excluded along with
// non-Latin scripts — those embeddings are near-untrained and cluster together.
const LATIN_TOKEN = /^(##)?[\x21-\x7EÀ-ɏ]+$/;

/**
 * Two groups of vocabulary entries are excluded from every top-k list:
 *
 *  - special tokens and BERT's ~1000 untrained [unusedN] slots;
 *  - non-Latin-script tokens (CJK, Sinhala, …). Their embeddings barely moved
 *    during training on English text, which leaves them clustered near each
 *    other and floating to the top of cosine rankings as pure noise.
 *
 * "##" continuation pieces are kept — a head that looks for the suffix "-ing"
 * is a genuinely interesting result.
 */
export const MINIMUM_WORD_LENGTH = 4;

function buildCandidateMask(vocabList, minimumLength) {
  const mask = new Uint8Array(vocabList.length);

  for (let i = 0; i < vocabList.length; i += 1) {
    const token = vocabList[i];
    const isSpecial = token.startsWith("[") && token.endsWith("]");
    const bare = token.replace(/^##/, "");

    mask[i] =
      !isSpecial && LATIN_TOKEN.test(token) && bare.length >= minimumLength
        ? 1
        : 0;
  }

  return mask;
}

// Two masks: with and without the short-fragment cut, built once each.
const maskCache = new Map();

function candidateMask(vocabList, minimumLength) {
  const key = `${vocabList.length}:${minimumLength}`;

  if (!maskCache.has(key)) {
    maskCache.set(key, buildCandidateMask(vocabList, minimumLength));
  }

  return maskCache.get(key);
}

/**
 * Runs every vocabulary entry through the embedding layer *at one position*
 * and then through one head's Q, K or V projection.
 *
 * This answers "what if the word at this slot were X instead?", so it is only
 * meaningful for layer 0 — at deeper layers a token's input depends on the
 * whole sentence, and a single-word substitution would change everything.
 */
export function projectVocabulary(model, { kind, headIndex, position }) {
  const { manifest, tensor, wordEmbeddingTable, vocabList } = model;
  const { hiddenSize, headDim, layerNormEps, vocabSize } = manifest;

  const weight = tensor(`layer0.${kind}.weight`);
  const bias = tensor(`layer0.${kind}.bias`);

  const lnWeight = tensor("embeddings_ln.weight");
  const lnBias = tensor("embeddings_ln.bias");
  const positionEmbeddings = tensor("position_embeddings");
  const typeEmbeddings = tensor("token_type_embeddings");

  const positionVector = positionEmbeddings.subarray(
    position * hiddenSize,
    (position + 1) * hiddenSize
  );

  const columnStart = headIndex * headDim;
  const projected = new Float32Array(vocabSize * headDim);
  const summed = new Float32Array(hiddenSize);

  for (let token = 0; token < vocabSize; token += 1) {
    const embeddingOffset = token * hiddenSize;

    for (let d = 0; d < hiddenSize; d += 1) {
      summed[d] =
        wordEmbeddingTable[embeddingOffset + d] +
        positionVector[d] +
        typeEmbeddings[d];
    }

    const normalized = layerNorm(summed, lnWeight, lnBias, layerNormEps);
    const outputOffset = token * headDim;

    for (let d = 0; d < hiddenSize; d += 1) {
      const value = normalized[d];
      if (value === 0) continue;

      const rowOffset = d * hiddenSize + columnStart;
      for (let c = 0; c < headDim; c += 1) {
        projected[outputOffset + c] += value * weight[rowOffset + c];
      }
    }

    for (let c = 0; c < headDim; c += 1) {
      projected[outputOffset + c] += bias[columnStart + c];
    }
  }

  return projected;
}

function topK(scores, vocabList, { limit, excludeIds = [], minimumLength = 1 }) {
  const mask = candidateMask(vocabList, minimumLength);
  const excluded = new Set(excludeIds);

  // A bounded insertion list beats sorting 30k entries.
  const best = [];

  for (let token = 0; token < scores.length; token += 1) {
    if (!mask[token] || excluded.has(token)) continue;

    const score = scores[token];
    if (best.length === limit && score <= best[best.length - 1].score) continue;

    const entry = { token: vocabList[token], id: token, score };
    let index = best.length;
    while (index > 0 && best[index - 1].score < score) index -= 1;

    best.splice(index, 0, entry);
    if (best.length > limit) best.pop();
  }

  return best;
}

/** Nearest vocabulary entries to `target` inside the same projected space. */
export function nearestInProjection(
  model,
  projected,
  target,
  { limit = 8, excludeIds = [], minimumLength = 1 } = {}
) {
  const { headDim, vocabSize } = model.manifest;

  let targetNorm = 0;
  for (let d = 0; d < headDim; d += 1) targetNorm += target[d] * target[d];
  targetNorm = Math.sqrt(targetNorm);

  const scores = new Float32Array(vocabSize);

  for (let token = 0; token < vocabSize; token += 1) {
    const offset = token * headDim;

    let dot = 0;
    let norm = 0;
    for (let d = 0; d < headDim; d += 1) {
      const value = projected[offset + d];
      dot += target[d] * value;
      norm += value * value;
    }

    const denominator = targetNorm * Math.sqrt(norm);
    scores[token] = denominator === 0 ? 0 : dot / denominator;
  }

  return topK(scores, model.vocabList, { limit, excludeIds, minimumLength });
}

/**
 * "If any word in the vocabulary sat in this sentence, which one would this
 * Query attend to most?" — the attention score itself, extended past the
 * sentence to the whole vocabulary.
 */
export function searchVocabularyByQuery(
  model,
  query,
  projectedKeys,
  { limit = 8, excludeIds = [], minimumLength = 1 } = {}
) {
  const { headDim, vocabSize } = model.manifest;
  const scale = Math.sqrt(headDim);

  const scores = new Float32Array(vocabSize);

  for (let token = 0; token < vocabSize; token += 1) {
    const offset = token * headDim;

    let score = 0;
    for (let d = 0; d < headDim; d += 1) {
      score += query[d] * projectedKeys[offset + d];
    }

    scores[token] = score / scale;
  }

  return topK(scores, model.vocabList, { limit, excludeIds, minimumLength });
}

/**
 * The mirror direction: "which word's Query would pick this Key up?" — i.e.
 * who is looking for this token. Uses the same W_Q W_K^T product, so it is
 * just as well-defined as searchVocabularyByQuery.
 */
export function searchVocabularyByKey(
  model,
  key,
  projectedQueries,
  { limit = 8, excludeIds = [], minimumLength = 1 } = {}
) {
  const { headDim, vocabSize } = model.manifest;
  const scale = Math.sqrt(headDim);

  const scores = new Float32Array(vocabSize);

  for (let token = 0; token < vocabSize; token += 1) {
    const offset = token * headDim;

    let score = 0;
    for (let d = 0; d < headDim; d += 1) {
      score += projectedQueries[offset + d] * key[d];
    }

    scores[token] = score / scale;
  }

  return topK(scores, model.vocabList, { limit, excludeIds, minimumLength });
}

/**
 * What a token would write into the residual stream if a head attended to it
 * with full weight: v·W_O, decoded back into words.
 *
 * W_V and W_O only ever act as the product W_V W_O (the "OV circuit"), so this
 * reading — unlike a nearest-neighbour list in V space — is well-defined.
 */
export function readValueAsWords(
  model,
  value,
  { layerIndex, headIndex, limit = 8 } = {}
) {
  const { hiddenSize, headDim } = model.manifest;

  // Rows [head*headDim, +headDim) of W_O are this head's slice, and contiguous.
  const outputWeight = model
    .tensor(`layer${layerIndex}.attn_out.weight`)
    .subarray(headIndex * headDim * hiddenSize);

  const delivered = multiplyVectorByFlatMatrix(
    Float32Array.from(value),
    outputWeight,
    headDim,
    hiddenSize
  );

  return logitLens(model, delivered, { limit });
}

/**
 * Logit lens: decode a hidden state into vocabulary probabilities through the
 * masked-language-model head. Applied to the final layer this is exactly BERT's
 * [MASK] prediction; applied earlier it shows what the model "has in mind" so
 * far. (nostalgebraist, 2020)
 */
export function logitLens(model, hiddenVector, { limit = 8 } = {}) {
  const { manifest, tensor, wordEmbeddingTable, vocabList } = model;
  const { hiddenSize, layerNormEps, vocabSize } = manifest;

  const denseWeight = tensor("mlm.dense.weight");
  const denseBias = tensor("mlm.dense.bias");

  const transformed = new Float32Array(hiddenSize);
  for (let out = 0; out < hiddenSize; out += 1) {
    let sum = denseBias[out];
    for (let i = 0; i < hiddenSize; i += 1) {
      sum += hiddenVector[i] * denseWeight[i * hiddenSize + out];
    }
    transformed[out] = gelu(sum);
  }

  const normalized = layerNorm(
    transformed,
    tensor("mlm.ln.weight"),
    tensor("mlm.ln.bias"),
    layerNormEps
  );

  // The decoder is weight-tied to the input embedding table.
  const decoderBias = tensor("mlm.bias");
  const logits = new Float32Array(vocabSize);

  for (let token = 0; token < vocabSize; token += 1) {
    const offset = token * hiddenSize;

    let logit = decoderBias[token];
    for (let d = 0; d < hiddenSize; d += 1) {
      logit += normalized[d] * wordEmbeddingTable[offset + d];
    }

    logits[token] = logit;
  }

  // Probabilities are over the whole vocabulary, not just the top-k — softmaxing
  // the shortlist alone would inflate every number.
  let maximum = -Infinity;
  for (let token = 0; token < vocabSize; token += 1) {
    if (logits[token] > maximum) maximum = logits[token];
  }

  let partition = 0;
  for (let token = 0; token < vocabSize; token += 1) {
    partition += Math.exp(logits[token] - maximum);
  }

  return topK(logits, vocabList, { limit }).map((entry) => ({
    ...entry,
    probability: Math.exp(entry.score - maximum) / partition,
  }));
}
