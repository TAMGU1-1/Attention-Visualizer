// Downloads (and caches) files from the Hugging Face Hub.
// Cached under scripts/.cache/ so repeated builds do not re-download.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const CACHE_DIRECTORY = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  ".cache"
);

export async function downloadModelFile(modelId, fileName) {
  await mkdir(CACHE_DIRECTORY, { recursive: true });

  const cachePath = path.join(
    CACHE_DIRECTORY,
    `${modelId.replace(/[/\\]/g, "_")}__${fileName}`
  );

  if (existsSync(cachePath)) {
    return readFile(cachePath);
  }

  const url = `https://huggingface.co/${modelId}/resolve/main/${fileName}`;
  process.stdout.write(`downloading ${url} ... `);

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} for ${url}`);
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  await writeFile(cachePath, buffer);

  console.log(`${(buffer.length / 1024 / 1024).toFixed(1)} MB`);
  return buffer;
}
