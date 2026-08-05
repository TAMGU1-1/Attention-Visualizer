// Do W_Q, W_K and W_V actually map words differently?
//
// Three measurements:
//   1. how much the nearest-neighbour lists overlap between the three spaces,
//      calibrated against two unrelated random projections;
//   2. how strongly the pairwise similarity structure agrees between them;
//   3. whether W_Q and W_K are individually meaningful at all.
//
// Usage: node scripts/analyze-qkv.mjs [layer] [head]

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createBertModel } from "../src/lib/bertModel.js";
import { layerNorm } from "../src/lib/linalg.js";

const layerIndex = Number(process.argv[2] ?? 0);
const headIndex = Number(process.argv[3] ?? 0);

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

const { hiddenSize, headDim, layerNormEps } = model.manifest;
const POSITION = 1;

// ---------------------------------------------------------------- sample set
// Common English words only: skip specials, [unused], continuation pieces and
// anything non-Latin, then take a slice from the frequency-ordered region.
const sample = [];
for (let id = 2000; id < model.vocabList.length && sample.length < 3000; id += 1) {
  const token = model.vocabList[id];
  if (/^[a-z]{3,}$/.test(token)) sample.push(id);
}
console.log(`vocabulary sample: ${sample.length} common words`);
console.log(`layer ${layerIndex}, head ${headIndex}, head_dim ${headDim}\n`);

// Embedding-layer output for each sampled word at a fixed position.
const positionEmbeddings = model.tensor("position_embeddings");
const typeEmbeddings = model.tensor("token_type_embeddings");
const lnWeight = model.tensor("embeddings_ln.weight");
const lnBias = model.tensor("embeddings_ln.bias");
const positionVector = positionEmbeddings.subarray(
  POSITION * hiddenSize,
  (POSITION + 1) * hiddenSize
);

const inputs = sample.map((id) => {
  const embedding = model.wordEmbedding(id);
  const summed = new Float32Array(hiddenSize);
  for (let d = 0; d < hiddenSize; d += 1) {
    summed[d] = embedding[d] + positionVector[d] + typeEmbeddings[d];
  }
  return layerNorm(summed, lnWeight, lnBias, layerNormEps);
});

// ---------------------------------------------------------------- projections
/** Slice one head's columns out of a fused [hidden, hidden] projection. */
function headWeight(kind) {
  const full = model.tensor(`layer${layerIndex}.${kind}.weight`);
  const start = headIndex * headDim;
  const out = new Float32Array(hiddenSize * headDim);

  for (let i = 0; i < hiddenSize; i += 1) {
    for (let c = 0; c < headDim; c += 1) {
      out[i * headDim + c] = full[i * hiddenSize + start + c];
    }
  }

  return out;
}

function project(weight) {
  return inputs.map((input) => {
    const out = new Float32Array(headDim);
    for (let i = 0; i < hiddenSize; i += 1) {
      const value = input[i];
      if (value === 0) continue;
      for (let c = 0; c < headDim; c += 1) {
        out[c] += value * weight[i * headDim + c];
      }
    }
    return out;
  });
}

function randomWeight(seed) {
  let state = seed;
  const random = () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296 - 0.5;
  };

  const out = new Float32Array(hiddenSize * headDim);
  for (let i = 0; i < out.length; i += 1) out[i] = random() * 0.1;
  return out;
}

const spaces = {
  Q: project(headWeight("query")),
  K: project(headWeight("key")),
  V: project(headWeight("value")),
  "rand-A": project(randomWeight(12345)),
  "rand-B": project(randomWeight(98765)),
};

// ---------------------------------------------------------------- helpers
function cosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let d = 0; d < a.length; d += 1) {
    dot += a[d] * b[d];
    na += a[d] * a[d];
    nb += b[d] * b[d];
  }
  const denominator = Math.sqrt(na) * Math.sqrt(nb);
  return denominator === 0 ? 0 : dot / denominator;
}

function neighbours(vectors, index, limit = 10) {
  const scored = [];
  for (let i = 0; i < vectors.length; i += 1) {
    if (i === index) continue;
    scored.push([i, cosine(vectors[index], vectors[i])]);
  }
  scored.sort((a, b) => b[1] - a[1]);
  return scored.slice(0, limit).map(([i]) => i);
}

function jaccard(a, b) {
  const setA = new Set(a);
  const shared = b.filter((value) => setA.has(value)).length;
  return shared / (a.length + b.length - shared);
}

// ---------------------------------------------- 1. neighbour-list overlap
const probes = [];
for (let i = 0; i < sample.length && probes.length < 60; i += 47) probes.push(i);

const pairs = [
  ["Q", "K"],
  ["Q", "V"],
  ["K", "V"],
  ["rand-A", "rand-B"],
];

console.log("1. top-10 neighbour overlap (Jaccard, 1.0 = identical lists)\n");

const overlapResults = {};
for (const [left, right] of pairs) {
  let total = 0;
  for (const probe of probes) {
    total += jaccard(
      neighbours(spaces[left], probe),
      neighbours(spaces[right], probe)
    );
  }
  const mean = total / probes.length;
  overlapResults[`${left}-${right}`] = mean;
  console.log(`   ${left.padEnd(6)} vs ${right.padEnd(6)}  ${mean.toFixed(3)}`);
}

console.log(
  `\n   (rand-A vs rand-B is the "unrelated projections" baseline)\n`
);

// ---------------------------------------------- 2. similarity-structure agreement
function pearson(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;

  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i += 1) {
    const a = xs[i] - mx;
    const b = ys[i] - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  return num / Math.sqrt(dx * dy);
}

const wordPairs = [];
for (let i = 0; i < 4000; i += 1) {
  const a = (i * 7919) % sample.length;
  const b = (i * 104729 + 13) % sample.length;
  if (a !== b) wordPairs.push([a, b]);
}

const similarityBySpace = {};
for (const name of Object.keys(spaces)) {
  similarityBySpace[name] = wordPairs.map(([a, b]) =>
    cosine(spaces[name][a], spaces[name][b])
  );
}

console.log("2. agreement of the pairwise similarity structure (Pearson r)\n");
for (const [left, right] of pairs) {
  const r = pearson(similarityBySpace[left], similarityBySpace[right]);
  console.log(`   ${left.padEnd(6)} vs ${right.padEnd(6)}  r = ${r.toFixed(3)}`);
}

// ---------------------------------------------- 3. are Q and K individually meaningful?
//
// Attention uses x_i W_Q W_K^T x_j^T. Substituting W_Q -> W_Q R and
// W_K -> W_K R^-T leaves that product — and therefore every attention score —
// untouched, while moving individual Q vectors somewhere else entirely.
console.log("\n3. is W_Q meaningful on its own?\n");

function invert(matrix, n) {
  const a = Array.from({ length: n }, (_, r) =>
    Array.from({ length: 2 * n }, (_, c) =>
      c < n ? matrix[r * n + c] : c - n === r ? 1 : 0
    )
  );

  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let r = col + 1; r < n; r += 1) {
      if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    }
    [a[col], a[pivot]] = [a[pivot], a[col]];

    const scale = a[col][col];
    for (let c = 0; c < 2 * n; c += 1) a[col][c] /= scale;

    for (let r = 0; r < n; r += 1) {
      if (r === col) continue;
      const factor = a[r][col];
      if (factor === 0) continue;
      for (let c = 0; c < 2 * n; c += 1) a[r][c] -= factor * a[col][c];
    }
  }

  const out = new Float64Array(n * n);
  for (let r = 0; r < n; r += 1) {
    for (let c = 0; c < n; c += 1) out[r * n + c] = a[r][c + n];
  }
  return out;
}

// A random well-conditioned R, plus its inverse-transpose.
let seed = 424242;
const random = () => {
  seed = (seed * 1664525 + 1013904223) % 4294967296;
  return seed / 4294967296 - 0.5;
};

const R = new Float64Array(headDim * headDim);
for (let r = 0; r < headDim; r += 1) {
  for (let c = 0; c < headDim; c += 1) {
    R[r * headDim + c] = (r === c ? 1 : 0) + random() * 0.3;
  }
}
const Rinv = invert(R, headDim);

function applyRight(vectors, matrix) {
  return vectors.map((vector) => {
    const out = new Float32Array(headDim);
    for (let i = 0; i < headDim; i += 1) {
      const value = vector[i];
      for (let c = 0; c < headDim; c += 1) out[c] += value * matrix[i * headDim + c];
    }
    return out;
  });
}

// q' = q R ; k' = k R^-T  (i.e. multiply k by the transpose of R^-1)
const RinvTransposed = new Float64Array(headDim * headDim);
for (let r = 0; r < headDim; r += 1) {
  for (let c = 0; c < headDim; c += 1) {
    RinvTransposed[r * headDim + c] = Rinv[c * headDim + r];
  }
}

const transformedQ = applyRight(spaces.Q, R);
const transformedK = applyRight(spaces.K, RinvTransposed);

let worstScoreDrift = 0;
for (const [a, b] of wordPairs.slice(0, 500)) {
  let before = 0;
  let after = 0;
  for (let d = 0; d < headDim; d += 1) {
    before += spaces.Q[a][d] * spaces.K[b][d];
    after += transformedQ[a][d] * transformedK[b][d];
  }
  worstScoreDrift = Math.max(worstScoreDrift, Math.abs(before - after));
}

let neighbourChange = 0;
for (const probe of probes) {
  neighbourChange += jaccard(
    neighbours(spaces.Q, probe),
    neighbours(transformedQ, probe)
  );
}
neighbourChange /= probes.length;

console.log(
  `   after W_Q → W_Q·R and W_K → W_K·R⁻ᵀ (random invertible R):`
);
console.log(
  `     attention scores changed by at most  ${worstScoreDrift.toExponential(2)}`
);
console.log(
  `     but Q-space neighbour overlap fell to ${neighbourChange.toFixed(3)}`
);
console.log(
  `\n   => attention is invariant, yet "Q's neighbours" is not. Only the`
);
console.log(`      product W_Q·W_Kᵀ is determined; W_Q alone is not.`);
