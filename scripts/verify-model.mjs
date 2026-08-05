// Sanity-checks the exported assets against the original checkpoint.
// Usage: node scripts/verify-model.mjs [modelId] [outputName]

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { readSafetensors, decodeFloat16 } from "./lib/safetensors.mjs";
import { downloadModelFile } from "./lib/download.mjs";

const modelId = process.argv[2] ?? "google/bert_uncased_L-2_H-128_A-2";
const outputName = process.argv[3] ?? "bert-tiny";

const directory = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  "public",
  "models",
  outputName
);

const tensors = readSafetensors(
  await downloadModelFile(modelId, "model.safetensors")
);
const manifest = JSON.parse(
  await readFile(path.join(directory, "manifest.json"), "utf-8")
);

const coreBytes = await readFile(path.join(directory, "core.f32.bin"));
const core = new Float32Array(
  coreBytes.buffer.slice(
    coreBytes.byteOffset,
    coreBytes.byteOffset + coreBytes.byteLength
  )
);

const halfBytes = await readFile(
  path.join(directory, "word-embeddings.f16.bin")
);
const half = new Uint16Array(
  halfBytes.buffer.slice(
    halfBytes.byteOffset,
    halfBytes.byteOffset + halfBytes.byteLength
  )
);

let failures = 0;
function check(label, condition, detail = "") {
  console.log(`${condition ? "  ok  " : "FAIL  "}${label}${detail && ` — ${detail}`}`);
  if (!condition) failures += 1;
}

// 1. fp16 round-trip error on the word embedding table
const original = tensors.get("bert.embeddings.word_embeddings.weight").data;
let maxError = 0;
for (let i = 0; i < original.length; i += 1) {
  maxError = Math.max(maxError, Math.abs(decodeFloat16(half[i]) - original[i]));
}
check(
  "word embeddings fp16 round-trip",
  maxError < 0.005,
  `max abs error ${maxError.toExponential(2)}`
);

// 2. transpose: exported [in, out] must equal original [out, in]
function checkTranspose(exportedName, originalName) {
  const entry = manifest.core.tensors[exportedName];
  const source = tensors.get(originalName);
  const [inDim, outDim] = entry.shape;

  let worst = 0;
  for (let i = 0; i < inDim; i += 1) {
    for (let o = 0; o < outDim; o += 1) {
      const exported = core[entry.offset + i * outDim + o];
      const expected = source.data[o * inDim + i];
      worst = Math.max(worst, Math.abs(exported - expected));
    }
  }

  check(`transpose ${exportedName}`, worst === 0, `[${inDim}, ${outDim}]`);
}

checkTranspose(
  "layer0.query.weight",
  "bert.encoder.layer.0.attention.self.query.weight"
);
checkTranspose(
  "layer1.attn_out.weight",
  "bert.encoder.layer.1.attention.output.dense.weight"
);
checkTranspose("layer0.ffn_in.weight", "bert.encoder.layer.0.intermediate.dense.weight");
checkTranspose("mlm.dense.weight", "cls.predictions.transform.dense.weight");

// 3. straight copies
function checkCopy(exportedName, originalName) {
  const entry = manifest.core.tensors[exportedName];
  const source = tensors.get(originalName).data;

  let worst = 0;
  for (let i = 0; i < source.length; i += 1) {
    worst = Math.max(worst, Math.abs(core[entry.offset + i] - source[i]));
  }

  check(`copy ${exportedName}`, worst === 0, `${source.length} floats`);
}

checkCopy(
  "position_embeddings",
  "bert.embeddings.position_embeddings.weight"
);
checkCopy("layer0.query.bias", "bert.encoder.layer.0.attention.self.query.bias");
checkCopy("mlm.bias", "cls.predictions.bias");

// 4. manifest coverage
const declared = Object.values(manifest.core.tensors).reduce(
  (max, entry) =>
    Math.max(max, entry.offset + entry.shape.reduce((a, b) => a * b, 1)),
  0
);
check(
  "core buffer fully described by manifest",
  declared === core.length,
  `${declared} / ${core.length} floats`
);

console.log(failures === 0 ? "\nall checks passed" : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
