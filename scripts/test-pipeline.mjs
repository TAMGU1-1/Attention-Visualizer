// End-to-end check of the browser-side BERT code, run under Node.
//
// The masked-language-model predictions are the real test: if the weight
// transpose, LayerNorm, GELU, or the residual connections were wrong, the
// top-k words would be noise rather than plausible completions.
//
// Usage: node scripts/test-pipeline.mjs

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createBertModel } from "../src/lib/bertModel.js";
import { runBertAttention } from "../src/lib/bertPipeline.js";
import { gelu, layerNorm, multiplyVectorByFlatMatrix } from "../src/lib/linalg.js";

const directory = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  "public",
  "models",
  "bert-tiny"
);

async function loadBuffer(name) {
  const file = await readFile(path.join(directory, name));
  return file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
}

const model = createBertModel({
  manifest: JSON.parse(await readFile(path.join(directory, "manifest.json"), "utf-8")),
  vocabText: await readFile(path.join(directory, "vocab.txt"), "utf-8"),
  coreBuffer: await loadBuffer("core.f32.bin"),
  embeddingBuffer: await loadBuffer("word-embeddings.f16.bin"),
});

let failures = 0;
function check(label, condition, detail = "") {
  console.log(`${condition ? "  ok  " : "FAIL  "}${label}${detail && ` — ${detail}`}`);
  if (!condition) failures += 1;
}

// ---------------------------------------------------------------- tokenizer
console.log("tokenizer\n");

const tokenizerCases = [
  ["The cat sat on the chair.", ["[CLS]", "the", "cat", "sat", "on", "the", "chair", ".", "[SEP]"]],
  ["visualizing", ["[CLS]", "visual", "##izing", "[SEP]"]],
  ["Café", ["[CLS]", "cafe", "[SEP]"]],
  ["don't", ["[CLS]", "don", "'", "t", "[SEP]"]],
];

for (const [text, expected] of tokenizerCases) {
  const actual = model.tokenizer.encode(text).map((entry) => entry.token);
  check(
    JSON.stringify(text),
    JSON.stringify(actual) === JSON.stringify(expected),
    actual.join(" ")
  );
}

// unknown ids must never appear for ordinary English
const ids = model.tokenizer.encode("The quick brown fox jumps over the lazy dog.");
check(
  "no [UNK] for common English",
  ids.every((entry) => entry.id !== model.tokenizer.unknownId)
);

// ---------------------------------------------------------------- attention
console.log("\nattention\n");

const result = runBertAttention(model, "The cat sat on the chair.", {
  layerIndex: 0,
  headIndex: 0,
});

check("token count", result.tokens.length === 9, `${result.tokens.length} tokens`);
check(
  "query/key/value are head-sized",
  result.tokens[0].query.length === model.manifest.headDim,
  `headDim ${model.manifest.headDim}`
);

const rowSums = result.attentionMatrix.map((row) =>
  row.reduce((a, b) => a + b, 0)
);
check(
  "attention rows sum to 1",
  rowSums.every((sum) => Math.abs(sum - 1) < 1e-6),
  `worst |sum-1| = ${Math.max(...rowSums.map((s) => Math.abs(s - 1))).toExponential(2)}`
);

check(
  "scaled = raw / sqrt(headDim)",
  result.rawScores.every((row, i) =>
    row.every(
      (value, j) =>
        Math.abs(value / Math.sqrt(model.manifest.headDim) - result.scaledScores[i][j]) < 1e-5
    )
  )
);

// heads must actually differ, otherwise the slicing is broken
const head1 = runBertAttention(model, "The cat sat on the chair.", {
  layerIndex: 0,
  headIndex: 1,
});
check(
  "head 0 and head 1 differ",
  head1.attentionMatrix.some((row, i) =>
    row.some((value, j) => Math.abs(value - result.attentionMatrix[i][j]) > 1e-3)
  )
);

// layer 1 must see a different hidden state than layer 0
const layer1 = runBertAttention(model, "The cat sat on the chair.", {
  layerIndex: 1,
  headIndex: 0,
});
check(
  "layer 1 input differs from layer 0 input",
  layer1.tokens.some((token, i) =>
    token.embedding.some(
      (value, d) => Math.abs(value - result.tokens[i].embedding[d]) > 1e-3
    )
  )
);

// position embeddings are actually applied
const identical = runBertAttention(model, "dog dog", { layerIndex: 0, headIndex: 0 });
check(
  "same word at different positions gets different embeddings",
  identical.tokens[1].embedding.some(
    (value, d) => Math.abs(value - identical.tokens[2].embedding[d]) > 1e-3
  )
);

// ---------------------------------------------------------------- MLM head
console.log("\nmasked language model (full 2-layer forward + MLM head)\n");

function predictMask(text, topK = 8) {
  const output = runBertAttention(model, text, {
    layerIndex: model.manifest.numLayers - 1,
    headIndex: 0,
  });

  const maskIndex = output.tokens.findIndex((token) => token.token === "[MASK]");
  if (maskIndex === -1) throw new Error("no [MASK] in input");

  const { hiddenSize, layerNormEps, vocabSize } = model.manifest;

  const transformed = multiplyVectorByFlatMatrix(
    Float32Array.from(output.tokens[maskIndex].finalHidden),
    model.tensor("mlm.dense.weight"),
    hiddenSize,
    hiddenSize,
    { bias: model.tensor("mlm.dense.bias") }
  );

  for (let i = 0; i < transformed.length; i += 1) {
    transformed[i] = gelu(transformed[i]);
  }

  const normalized = layerNorm(
    transformed,
    model.tensor("mlm.ln.weight"),
    model.tensor("mlm.ln.bias"),
    layerNormEps
  );

  // The MLM decoder is weight-tied to the input embedding table.
  const bias = model.tensor("mlm.bias");
  const scored = [];

  for (let v = 0; v < vocabSize; v += 1) {
    const row = model.wordEmbedding(v);
    let logit = bias[v];
    for (let d = 0; d < hiddenSize; d += 1) logit += normalized[d] * row[d];
    scored.push([v, logit]);
  }

  scored.sort((a, b) => b[1] - a[1]);
  return scored.slice(0, topK).map(([v]) => model.vocabList[v]);
}

// bert-tiny is 2 layers / 128 hidden — far too small for factual recall, so
// these assert the right *category*, not the right fact. ("the capital of
// france is [MASK]" yields countries, not "paris".)
const mlmCases = [
  [
    "the capital of france is [MASK].",
    ["paris", "france", "spain", "germany", "italy", "belgium", "canada", "algeria", "commune"],
  ],
  ["she is a [MASK].", ["woman", "man", "girl", "child", "female", "singer", "student", "teacher"]],
  ["i ate an [MASK] for breakfast.", ["egg", "apple", "orange", "hour", "meal", "lunch", "breakfast"]],
];

for (const [text, acceptable] of mlmCases) {
  const predictions = predictMask(text);
  const hit = predictions.some((word) => acceptable.includes(word));
  check(text, hit, `top-8: ${predictions.join(", ")}`);
}

console.log(
  failures === 0
    ? "\nall checks passed"
    : `\n${failures} FAILURES`
);
process.exit(failures === 0 ? 0 : 1);
