// Runs a real BERT encoder forward pass and captures the intermediate values
// for one (layer, head) so the UI can render them.
//
// Layers before the selected one are executed in full — attention *and* the
// feed-forward block — because the hidden state entering layer L is only
// correct if every earlier layer ran completely.

import {
  addInPlace,
  dotProduct,
  gelu,
  layerNorm,
  multiplyVectorByFlatMatrix,
  softmax,
} from "./linalg.js";

export function runBertAttention(
  model,
  text,
  { layerIndex = 0, headIndex = 0 } = {}
) {
  const { manifest, tokenizer, tensor } = model;
  const { hiddenSize, headDim, numLayers, layerNormEps } = manifest;

  const encoded = tokenizer.encode(text);
  if (encoded.length === 0) return null;

  const positionEmbeddings = tensor("position_embeddings");
  const typeEmbeddings = tensor("token_type_embeddings");

  // 1. token + position + segment, then LayerNorm.
  const tokenEmbeddings = [];
  const positionVectors = [];

  const hidden = encoded.map((entry, position) => {
    const wordVector = model.wordEmbedding(entry.id);
    const positionVector = positionEmbeddings.subarray(
      position * hiddenSize,
      (position + 1) * hiddenSize
    );

    tokenEmbeddings.push(Float32Array.from(wordVector));
    positionVectors.push(Float32Array.from(positionVector));

    const summed = Float32Array.from(wordVector);
    addInPlace(summed, positionVector);
    addInPlace(summed, typeEmbeddings.subarray(0, hiddenSize)); // segment A

    return layerNorm(
      summed,
      tensor("embeddings_ln.weight"),
      tensor("embeddings_ln.bias"),
      layerNormEps
    );
  });

  const targetLayer = Math.min(Math.max(layerIndex, 0), numLayers - 1);
  const intermediateSize = model.shapeOf(`layer0.ffn_in.weight`)[1];

  let capture = null;
  let states = hidden;

  // Every layer runs, not just up to the selected one: the capture comes from
  // `targetLayer`, but `finalHidden` has to be the true output of the model for
  // the logit lens to mean what it claims.
  for (let layer = 0; layer < numLayers; layer += 1) {
    const result = runLayer({
      tensor,
      states,
      layer,
      hiddenSize,
      headDim,
      intermediateSize,
      layerNormEps,
      captureHead: layer === targetLayer ? headIndex : null,
    });

    if (result.capture) capture = result.capture;
    states = result.states;
  }

  return {
    tokens: encoded.map((entry, index) => ({
      ...entry,
      tokenEmbedding: Array.from(tokenEmbeddings[index]),
      positionEmbedding: Array.from(positionVectors[index]),
      embedding: Array.from(capture.layerInput[index]),
      query: Array.from(capture.query[index]),
      key: Array.from(capture.key[index]),
      value: Array.from(capture.value[index]),
      output: Array.from(capture.context[index]),
      projectedOutput: Array.from(capture.projection[index]),
      finalHidden: Array.from(states[index]),
    })),
    rawScores: capture.rawScores,
    scaledScores: capture.scaledScores,
    attentionMatrix: capture.attention,
    layerIndex: targetLayer,
    headIndex,
  };
}

function runLayer({
  tensor,
  states,
  layer,
  hiddenSize,
  headDim,
  intermediateSize,
  layerNormEps,
  captureHead,
}) {
  const prefix = `layer${layer}`;
  const numHeads = hiddenSize / headDim;

  const queryWeight = tensor(`${prefix}.query.weight`);
  const keyWeight = tensor(`${prefix}.key.weight`);
  const valueWeight = tensor(`${prefix}.value.weight`);
  const outputWeight = tensor(`${prefix}.attn_out.weight`);

  // Fused projections: every head at once, sliced per head below.
  const queries = states.map((state) =>
    multiplyVectorByFlatMatrix(state, queryWeight, hiddenSize, hiddenSize, {
      bias: tensor(`${prefix}.query.bias`),
    })
  );
  const keys = states.map((state) =>
    multiplyVectorByFlatMatrix(state, keyWeight, hiddenSize, hiddenSize, {
      bias: tensor(`${prefix}.key.bias`),
    })
  );
  const values = states.map((state) =>
    multiplyVectorByFlatMatrix(state, valueWeight, hiddenSize, hiddenSize, {
      bias: tensor(`${prefix}.value.bias`),
    })
  );

  const tokenCount = states.length;
  const scale = Math.sqrt(headDim);

  const contexts = states.map(() => new Float32Array(hiddenSize));
  let capture = null;

  for (let head = 0; head < numHeads; head += 1) {
    const start = head * headDim;

    const headQueries = queries.map((q) => q.subarray(start, start + headDim));
    const headKeys = keys.map((k) => k.subarray(start, start + headDim));
    const headValues = values.map((v) => v.subarray(start, start + headDim));

    const rawScores = headQueries.map((q) =>
      headKeys.map((k) => dotProduct(q, k))
    );
    const scaledScores = rawScores.map((row) => row.map((s) => s / scale));
    const attention = scaledScores.map((row) => softmax(row));

    for (let i = 0; i < tokenCount; i += 1) {
      for (let j = 0; j < tokenCount; j += 1) {
        const weight = attention[i][j];
        for (let d = 0; d < headDim; d += 1) {
          contexts[i][start + d] += weight * headValues[j][d];
        }
      }
    }

    if (head === captureHead) {
      const headContexts = contexts.map((context) =>
        Float32Array.from(context.subarray(start, start + headDim))
      );

      // Rows [head*headDim, +headDim) of W_O are exactly this head's slice of
      // the output projection, and they are contiguous in row-major order.
      const headOutputWeight = outputWeight.subarray(start * hiddenSize);

      capture = {
        layerInput: states,
        query: headQueries,
        key: headKeys,
        value: headValues,
        rawScores,
        scaledScores,
        attention,
        context: headContexts,
        projection: headContexts.map((context) =>
          multiplyVectorByFlatMatrix(
            context,
            headOutputWeight,
            headDim,
            hiddenSize
          )
        ),
      };
    }
  }

  // attention output projection + residual + LayerNorm
  const afterAttention = contexts.map((context, index) => {
    const projected = multiplyVectorByFlatMatrix(
      context,
      outputWeight,
      hiddenSize,
      hiddenSize,
      { bias: tensor(`${prefix}.attn_out.bias`) }
    );

    addInPlace(projected, states[index]);

    return layerNorm(
      projected,
      tensor(`${prefix}.attn_ln.weight`),
      tensor(`${prefix}.attn_ln.bias`),
      layerNormEps
    );
  });

  // feed-forward + residual + LayerNorm
  const outputStates = afterAttention.map((state) => {
    const intermediate = multiplyVectorByFlatMatrix(
      state,
      tensor(`${prefix}.ffn_in.weight`),
      hiddenSize,
      intermediateSize,
      { bias: tensor(`${prefix}.ffn_in.bias`) }
    );

    for (let i = 0; i < intermediate.length; i += 1) {
      intermediate[i] = gelu(intermediate[i]);
    }

    const projected = multiplyVectorByFlatMatrix(
      intermediate,
      tensor(`${prefix}.ffn_out.weight`),
      intermediateSize,
      hiddenSize,
      { bias: tensor(`${prefix}.ffn_out.bias`) }
    );

    addInPlace(projected, state);

    return layerNorm(
      projected,
      tensor(`${prefix}.ffn_ln.weight`),
      tensor(`${prefix}.ffn_ln.bias`),
      layerNormEps
    );
  });

  return { states: outputStates, capture };
}
