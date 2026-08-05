// Prints the tensor names/shapes of a Hugging Face safetensors checkpoint.
// Usage: node scripts/inspect-model.mjs [modelId]

import { readSafetensors } from "./lib/safetensors.mjs";
import { downloadModelFile } from "./lib/download.mjs";

const modelId = process.argv[2] ?? "google/bert_uncased_L-2_H-128_A-2";

const buffer = await downloadModelFile(modelId, "model.safetensors");
const tensors = readSafetensors(buffer);

console.log(`${modelId} — ${tensors.size} tensors\n`);

let totalParameters = 0;
for (const [name, tensor] of tensors) {
  const count = tensor.shape.reduce((a, b) => a * b, 1);
  totalParameters += count;
  console.log(`${name.padEnd(60)} [${tensor.shape.join(", ")}]`);
}

console.log(`\ntotal parameters: ${totalParameters.toLocaleString()}`);
