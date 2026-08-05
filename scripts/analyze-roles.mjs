// Does Q really act as the "query" and K as the "key"?
//
// Every attention score has the form
//
//     score(i, j) = x̃_i · M · x̃_j^T ,   M = [[W_Q W_K^T , W_Q b_K^T],
//                                            [b_Q W_K^T  , b_Q · b_K ]]
//
// (x̃ = x with a 1 appended, so the biases are included exactly).
//
// If M were symmetric then score(i,j) = score(j,i) for every pair, i.e. asking
// and being asked would be the same thing and swapping Q with K would change
// nothing. So the conventional "Q = search term, K = label" story is testable:
// it requires M to be asymmetric.
//
// Swapping Q and K is exactly transposing M, which is why one measurement
// answers both questions.
//
// Usage: node scripts/analyze-roles.mjs ["sentence"]

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createBertModel } from "../src/lib/bertModel.js";
import { runBertAttention } from "../src/lib/bertPipeline.js";
import { softmax } from "../src/lib/linalg.js";

const sentence = process.argv[2] ?? "The cat sat on the mat.";
const probeLayer = Number(process.argv[3] ?? 0);
const probeHead = Number(process.argv[4] ?? 0);

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

const { hiddenSize, headDim, numLayers, numHeads } = model.manifest;

// ------------------------------------------------------------ 1. is M symmetric?
/** Augmented QK matrix for one head, biases included: size (hidden+1)². */
function qkMatrix(layer, head) {
  const queryWeight = model.tensor(`layer${layer}.query.weight`);
  const keyWeight = model.tensor(`layer${layer}.key.weight`);
  const queryBias = model.tensor(`layer${layer}.query.bias`);
  const keyBias = model.tensor(`layer${layer}.key.bias`);

  const start = head * headDim;
  const n = hiddenSize + 1;
  const M = new Float64Array(n * n);

  const q = (row, c) => queryWeight[row * hiddenSize + start + c];
  const k = (row, c) => keyWeight[row * hiddenSize + start + c];
  const qb = (c) => queryBias[start + c];
  const kb = (c) => keyBias[start + c];

  for (let a = 0; a < hiddenSize; a += 1) {
    for (let b = 0; b < hiddenSize; b += 1) {
      let sum = 0;
      for (let c = 0; c < headDim; c += 1) sum += q(a, c) * k(b, c);
      M[a * n + b] = sum;
    }

    let withKeyBias = 0;
    let withQueryBias = 0;
    for (let c = 0; c < headDim; c += 1) {
      withKeyBias += q(a, c) * kb(c);
      withQueryBias += qb(c) * k(a, c);
    }
    M[a * n + hiddenSize] = withKeyBias;
    M[hiddenSize * n + a] = withQueryBias;
  }

  let biasDot = 0;
  for (let c = 0; c < headDim; c += 1) biasDot += qb(c) * kb(c);
  M[hiddenSize * n + hiddenSize] = biasDot;

  return { M, n };
}

/** ‖antisymmetric part‖ / ‖M‖ — 0 = perfectly symmetric, ~0.707 = random. */
function asymmetryRatio(M, n) {
  let total = 0;
  let anti = 0;

  for (let a = 0; a < n; a += 1) {
    for (let b = 0; b < n; b += 1) {
      const value = M[a * n + b];
      const half = (value - M[b * n + a]) / 2;
      total += value * value;
      anti += half * half;
    }
  }

  return Math.sqrt(anti) / Math.sqrt(total);
}

console.log("1. QK 행렬 M = W_Q·W_Kᵀ 의 비대칭성\n");
console.log("   0.000 = 완전 대칭 (Q와 K가 구별되지 않음)");
console.log("   0.707 = 무작위 행렬 수준 (역할이 완전히 갈림)\n");

for (let layer = 0; layer < numLayers; layer += 1) {
  for (let head = 0; head < numHeads; head += 1) {
    const { M, n } = qkMatrix(layer, head);
    console.log(
      `   layer ${layer}, head ${head}:  ${asymmetryRatio(M, n).toFixed(3)}`
    );
  }
}

// ------------------------------------------------------------ 2. observed asymmetry
const result = runBertAttention(model, sentence, {
  layerIndex: probeLayer,
  headIndex: probeHead,
});
const tokens = result.tokens.map((entry) => entry.token);

console.log(
  `\n\n2. 실제 문장에서의 비대칭  —  "${sentence}" (layer ${probeLayer}, head ${probeHead})\n`
);
console.log(`   토큰: ${tokens.join(" ")}\n`);

const attention = result.attentionMatrix;
const asymmetricPairs = [];

for (let i = 0; i < tokens.length; i += 1) {
  for (let j = 0; j < tokens.length; j += 1) {
    if (i === j) continue;
    asymmetricPairs.push({
      i,
      j,
      forward: attention[i][j],
      backward: attention[j][i],
      gap: attention[i][j] - attention[j][i],
    });
  }
}

asymmetricPairs.sort((a, b) => b.gap - a.gap);

console.log("   가장 일방적인 쌍 (A가 B를 보지만 B는 A를 안 봄):\n");
for (const pair of asymmetricPairs.slice(0, 6)) {
  console.log(
    `     ${tokens[pair.i].padEnd(7)} → ${tokens[pair.j].padEnd(7)} ` +
      `${(pair.forward * 100).toFixed(1).padStart(5)}%   ` +
      `반대 방향 ${(pair.backward * 100).toFixed(1).padStart(5)}%`
  );
}

// ------------------------------------------------------------ 3. swap Q and K
console.log("\n\n3. Q와 K를 맞바꾸면?\n");

const scale = Math.sqrt(headDim);
const swappedScores = tokens.map((_, i) =>
  tokens.map((_, j) => {
    // k_i · q_j instead of q_i · k_j
    let sum = 0;
    for (let d = 0; d < headDim; d += 1) {
      sum += result.tokens[i].key[d] * result.tokens[j].query[d];
    }
    return sum / scale;
  })
);
const swappedAttention = swappedScores.map((row) => softmax(row));

// total-variation distance per row: 0 = identical, 1 = disjoint
let totalVariation = 0;
for (let i = 0; i < tokens.length; i += 1) {
  let difference = 0;
  for (let j = 0; j < tokens.length; j += 1) {
    difference += Math.abs(attention[i][j] - swappedAttention[i][j]);
  }
  totalVariation += difference / 2;
}
totalVariation /= tokens.length;

console.log(
  `   attention 분포 변화량 (평균 total variation): ${totalVariation.toFixed(3)}`
);
console.log(`   0 = 아무것도 안 바뀜 (역할 구분 없음), 1 = 완전히 다른 분포\n`);

console.log("   토큰별 최대 주목 대상 변화:\n");
for (let i = 0; i < tokens.length; i += 1) {
  const before = attention[i].indexOf(Math.max(...attention[i]));
  const after = swappedAttention[i].indexOf(Math.max(...swappedAttention[i]));
  const changed = before !== after;

  console.log(
    `     ${tokens[i].padEnd(7)} ${tokens[before].padEnd(7)} → ${tokens[after].padEnd(7)}` +
      `${changed ? "   바뀜" : ""}`
  );
}

// ------------------------------------------------------------ 4. self vs other
console.log("\n\n4. 이 헤드는 무엇을 하고 있나\n");

let selfAttention = 0;
for (let i = 0; i < tokens.length; i += 1) selfAttention += attention[i][i];
selfAttention /= tokens.length;

console.log(`   자기 자신에 준 주목의 평균: ${(selfAttention * 100).toFixed(1)}%`);
console.log(
  `   (높으면 '같은 단어 찾기' 헤드 — 검색어와 제목이 사실상 같은 것을 가리킴)`
);
