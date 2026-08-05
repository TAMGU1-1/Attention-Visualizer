// Loads the assets produced by scripts/build-model.mjs.
//
// Weights ship as raw binary rather than JSON: the embedding table alone is
// ~3.9M floats, which would be ~30MB as text.

import { createTokenizer } from "./wordpiece.js";

export const DEFAULT_MODEL_PATH = "/models/bert-tiny";

function decodeFloat16Buffer(uint16) {
  // Float16Array is only in very recent browsers, so decode by hand.
  const out = new Float32Array(uint16.length);

  for (let i = 0; i < uint16.length; i += 1) {
    const bits = uint16[i];
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >> 10) & 0x1f;
    const fraction = bits & 0x03ff;

    if (exponent === 0) {
      out[i] = sign * fraction * 2 ** -24;
    } else if (exponent === 0x1f) {
      out[i] = fraction ? NaN : sign * Infinity;
    } else {
      out[i] = sign * (fraction + 1024) * 2 ** (exponent - 25);
    }
  }

  return out;
}

export async function loadBertModel(
  basePath = DEFAULT_MODEL_PATH,
  onProgress = () => {}
) {
  // Progress is reported *after* each await so that callers can call setState
  // from the callback without it running synchronously inside an effect.
  const manifest = await (await fetch(`${basePath}/manifest.json`)).json();

  onProgress("vocabulary");
  const vocabText = await (await fetch(`${basePath}/${manifest.vocabFile}`)).text();

  onProgress("weights");
  const coreBuffer = await (
    await fetch(`${basePath}/${manifest.core.file}`)
  ).arrayBuffer();

  onProgress("embedding table");
  const embeddingBuffer = await (
    await fetch(`${basePath}/${manifest.wordEmbeddings.file}`)
  ).arrayBuffer();

  return createBertModel({
    manifest,
    vocabText,
    coreBuffer,
    embeddingBuffer,
  });
}

/** Assembles a model from already-fetched bytes (also used by the Node tests). */
export function createBertModel({
  manifest,
  vocabText,
  coreBuffer,
  embeddingBuffer,
}) {
  const vocabList = vocabText.split("\n");
  if (vocabList.at(-1) === "") vocabList.pop();

  const core = new Float32Array(coreBuffer);
  const wordEmbeddings = decodeFloat16Buffer(new Uint16Array(embeddingBuffer));

  function tensor(name) {
    const entry = manifest.core.tensors[name];
    if (!entry) throw new Error(`unknown tensor: ${name}`);

    const length = entry.shape.reduce((a, b) => a * b, 1);
    return core.subarray(entry.offset, entry.offset + length);
  }

  function shapeOf(name) {
    const entry = manifest.core.tensors[name];
    if (!entry) throw new Error(`unknown tensor: ${name}`);
    return entry.shape;
  }

  const { hiddenSize } = manifest;

  return {
    manifest,
    vocabList,
    tokenizer: createTokenizer(vocabList),
    tensor,
    shapeOf,

    /** Row `id` of the embedding table, as a view (do not mutate). */
    wordEmbedding(id) {
      return wordEmbeddings.subarray(id * hiddenSize, (id + 1) * hiddenSize);
    },

    /** The whole [vocabSize, hiddenSize] table, flat row-major. */
    wordEmbeddingTable: wordEmbeddings,
  };
}
