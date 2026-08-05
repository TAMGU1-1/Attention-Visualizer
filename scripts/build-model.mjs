// Exports a pretrained BERT checkpoint into assets the browser can load directly.
//
//   node scripts/build-model.mjs [modelId] [outputName]
//
// Produces public/models/<outputName>/
//   manifest.json            config + float offsets into core.f32.bin
//   core.f32.bin             positional/LayerNorm/attention/FFN/MLM-head weights
//   word-embeddings.f16.bin  the vocab embedding table (half precision)
//   vocab.txt                WordPiece vocabulary
//
// PyTorch stores nn.Linear weights as [out_features, in_features]. Everything
// written here is transposed to [in, out] so the app's multiplyVectorByMatrix
// (which walks rows as inputs) can be used unchanged.

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { readSafetensors, encodeFloat16 } from "./lib/safetensors.mjs";
import { downloadModelFile } from "./lib/download.mjs";

const modelId = process.argv[2] ?? "google/bert_uncased_L-2_H-128_A-2";
const outputName = process.argv[3] ?? "bert-tiny";

const projectRoot = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  ".."
);
const outputDirectory = path.join(projectRoot, "public", "models", outputName);

function transpose(tensor) {
  const [rows, columns] = tensor.shape;
  const out = new Float32Array(rows * columns);

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < columns; c += 1) {
      out[c * rows + r] = tensor.data[r * columns + c];
    }
  }

  return { shape: [columns, rows], data: out };
}

const config = JSON.parse(
  (await downloadModelFile(modelId, "config.json")).toString("utf-8")
);
const vocabText = (await downloadModelFile(modelId, "vocab.txt")).toString(
  "utf-8"
);
const tensors = readSafetensors(
  await downloadModelFile(modelId, "model.safetensors")
);

function get(name) {
  const tensor = tensors.get(name);
  if (!tensor) throw new Error(`missing tensor: ${name}`);
  return tensor;
}

// nn.Linear -> [in, out]
function linear(name) {
  return transpose(get(`${name}.weight`));
}
function bias(name) {
  return get(`${name}.bias`);
}

const hiddenSize = config.hidden_size;
const numHeads = config.num_attention_heads;
const numLayers = config.num_hidden_layers;

const core = [];

function push(name, tensor) {
  core.push({ name, shape: tensor.shape, data: tensor.data });
}

push("position_embeddings", get("bert.embeddings.position_embeddings.weight"));
push(
  "token_type_embeddings",
  get("bert.embeddings.token_type_embeddings.weight")
);
push("embeddings_ln.weight", get("bert.embeddings.LayerNorm.weight"));
push("embeddings_ln.bias", get("bert.embeddings.LayerNorm.bias"));

for (let layer = 0; layer < numLayers; layer += 1) {
  const prefix = `bert.encoder.layer.${layer}`;

  for (const [source, target] of [
    ["attention.self.query", "query"],
    ["attention.self.key", "key"],
    ["attention.self.value", "value"],
    ["attention.output.dense", "attn_out"],
    ["intermediate.dense", "ffn_in"],
    ["output.dense", "ffn_out"],
  ]) {
    push(`layer${layer}.${target}.weight`, linear(`${prefix}.${source}`));
    push(`layer${layer}.${target}.bias`, bias(`${prefix}.${source}`));
  }

  for (const [source, target] of [
    ["attention.output.LayerNorm", "attn_ln"],
    ["output.LayerNorm", "ffn_ln"],
  ]) {
    push(`layer${layer}.${target}.weight`, get(`${prefix}.${source}.weight`));
    push(`layer${layer}.${target}.bias`, get(`${prefix}.${source}.bias`));
  }
}

// MLM head — lets the app decode a hidden state back into vocabulary logits.
push("mlm.dense.weight", linear("cls.predictions.transform.dense"));
push("mlm.dense.bias", bias("cls.predictions.transform.dense"));
push("mlm.ln.weight", get("cls.predictions.transform.LayerNorm.weight"));
push("mlm.ln.bias", get("cls.predictions.transform.LayerNorm.bias"));
push("mlm.bias", get("cls.predictions.bias"));

const totalFloats = core.reduce((sum, entry) => sum + entry.data.length, 0);
const coreBuffer = new Float32Array(totalFloats);
const coreIndex = {};

let offset = 0;
for (const entry of core) {
  coreBuffer.set(entry.data, offset);
  coreIndex[entry.name] = { shape: entry.shape, offset };
  offset += entry.data.length;
}

const wordEmbeddings = get("bert.embeddings.word_embeddings.weight");
const half = new Uint16Array(wordEmbeddings.data.length);
for (let i = 0; i < wordEmbeddings.data.length; i += 1) {
  half[i] = encodeFloat16(wordEmbeddings.data[i]);
}

const vocab = vocabText.split("\n");
if (vocab.at(-1) === "") vocab.pop();

const manifest = {
  modelId,
  hiddenSize,
  numHeads,
  numLayers,
  headDim: hiddenSize / numHeads,
  vocabSize: vocab.length,
  maxPositions: config.max_position_embeddings,
  layerNormEps: config.layer_norm_eps ?? 1e-12,
  core: { file: "core.f32.bin", dtype: "f32", tensors: coreIndex },
  wordEmbeddings: {
    file: "word-embeddings.f16.bin",
    dtype: "f16",
    shape: wordEmbeddings.shape,
  },
  vocabFile: "vocab.txt",
};

await mkdir(outputDirectory, { recursive: true });
await writeFile(
  path.join(outputDirectory, "manifest.json"),
  JSON.stringify(manifest, null, 2)
);
await writeFile(
  path.join(outputDirectory, "core.f32.bin"),
  Buffer.from(coreBuffer.buffer)
);
await writeFile(
  path.join(outputDirectory, "word-embeddings.f16.bin"),
  Buffer.from(half.buffer)
);
await writeFile(path.join(outputDirectory, "vocab.txt"), vocab.join("\n"));

const mb = (bytes) => `${(bytes / 1024 / 1024).toFixed(2)} MB`;

console.log(`\nwrote public/models/${outputName}/`);
console.log(`  core.f32.bin            ${mb(coreBuffer.byteLength)}`);
console.log(`  word-embeddings.f16.bin ${mb(half.byteLength)}`);
console.log(`  vocab.txt               ${vocab.length} tokens`);
console.log(
  `  ${hiddenSize}d · ${numLayers} layers · ${numHeads} heads · head_dim ${
    hiddenSize / numHeads
  }`
);
