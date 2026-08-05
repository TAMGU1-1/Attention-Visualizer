// Exercises the vocabulary-lens features and times them.
// Usage: node scripts/test-vocab-lens.mjs

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createBertModel } from "../src/lib/bertModel.js";
import { runBertAttention } from "../src/lib/bertPipeline.js";
import {
  logitLens,
  nearestInProjection,
  projectVocabulary,
  searchVocabularyByQuery,
} from "../src/lib/vocabLens.js";

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

function time(label, fn) {
  const start = performance.now();
  const value = fn();
  const elapsed = performance.now() - start;
  console.log(`       ${label}: ${elapsed.toFixed(0)} ms`);
  return { value, elapsed };
}

const sentence = "The cat sat on the mat.";
const result = runBertAttention(model, sentence, { layerIndex: 0, headIndex: 0 });
const tokens = result.tokens.map((entry) => entry.token);
console.log(`sentence: ${sentence}\ntokens:   ${tokens.join(" ")}\n`);

const catIndex = tokens.indexOf("cat");
const sentenceIds = result.tokens.map((entry) => entry.id);

// ---------------------------------------------------------- timing / cost
console.log("vocabulary projection (30522 words through one head)\n");

const keys = time("project K", () =>
  projectVocabulary(model, { kind: "key", headIndex: 0, position: catIndex })
);
const queries = time("project Q", () =>
  projectVocabulary(model, { kind: "query", headIndex: 0, position: catIndex })
);

check(
  "projection is fast enough to run on click",
  keys.elapsed < 2000,
  `${keys.elapsed.toFixed(0)} ms`
);
check(
  "projected shape",
  keys.value.length === model.manifest.vocabSize * model.manifest.headDim
);

// The projection must reproduce the real pipeline for the actual token.
const catQuery = result.tokens[catIndex].query;
const projectedCatQuery = queries.value.subarray(
  sentenceIds[catIndex] * model.manifest.headDim,
  (sentenceIds[catIndex] + 1) * model.manifest.headDim
);
const worst = Math.max(
  ...catQuery.map((value, d) => Math.abs(value - projectedCatQuery[d]))
);
check(
  "vocabulary projection matches the live forward pass",
  worst < 1e-4,
  `max abs diff ${worst.toExponential(2)} on "cat"`
);

// ---------------------------------------------------------- (a) neighbours
console.log("\n(a) neighbours inside each space — token: cat\n");

for (const kind of ["query", "key", "value"]) {
  const projected = projectVocabulary(model, {
    kind,
    headIndex: 0,
    position: catIndex,
  });

  const target =
    kind === "query"
      ? result.tokens[catIndex].query
      : kind === "key"
        ? result.tokens[catIndex].key
        : result.tokens[catIndex].value;

  const neighbours = nearestInProjection(model, projected, target, {
    limit: 8,
    excludeIds: [sentenceIds[catIndex]],
  });

  console.log(
    `  ${kind.padEnd(5)}: ${neighbours
      .map((entry) => `${entry.token} (${entry.score.toFixed(3)})`)
      .join(", ")}`
  );
}

// ---------------------------------------------------------- (b) K search
console.log("\n(b) what is each Query looking for? (top of 30522)\n");

for (const index of result.tokens.map((_, i) => i)) {
  const projectedKeys = projectVocabulary(model, {
    kind: "key",
    headIndex: 0,
    position: index,
  });

  const matches = searchVocabularyByQuery(
    model,
    result.tokens[index].query,
    projectedKeys,
    { limit: 6 }
  );

  console.log(
    `  ${tokens[index].padEnd(6)} → ${matches.map((m) => m.token).join(", ")}`
  );
}

// Candidates have to be placed at *some* position to become Keys. For content
// words the word itself dominates, so the ranking barely moves — which is what
// makes the feature honest to present.
console.log("\n    position sensitivity of the top-8 (content word 'sat')\n");

const satIndex = tokens.indexOf("sat");
const rankings = [0, 1, 3, 8, 50].map((position) => {
  const projectedKeys = projectVocabulary(model, {
    kind: "key",
    headIndex: 0,
    position,
  });
  return searchVocabularyByQuery(model, result.tokens[satIndex].query, projectedKeys, {
    limit: 8,
  }).map((entry) => entry.token);
});

const reference = new Set(rankings[0]);
const overlaps = rankings
  .slice(1)
  .map((list) => list.filter((token) => reference.has(token)).length);

console.log(`      pos 0:  ${rankings[0].join(", ")}`);
console.log(`      pos 50: ${rankings.at(-1).join(", ")}`);
check(
  "content-word ranking is stable across positions",
  overlaps.every((overlap) => overlap >= 6),
  `overlap with pos 0: ${overlaps.join(", ")} / 8`
);

// ---------------------------------------------------------- (c) logit lens
console.log("\n(c) logit lens\n");

const finalLayer = runBertAttention(model, sentence, {
  layerIndex: model.manifest.numLayers - 1,
  headIndex: 0,
});

// finalHidden must be the output of the *whole* model regardless of which layer
// is being inspected, otherwise the logit lens reports the wrong thing.
check(
  "finalHidden is layer-selection independent",
  result.tokens.every((token, index) =>
    token.finalHidden.every(
      (value, d) => Math.abs(value - finalLayer.tokens[index].finalHidden[d]) < 1e-5
    )
  ),
  "captured at layer 0 vs last layer"
);

const readable = /^(##)?[\x21-\x7EÀ-ɏ]+$/;

check(
  "logit lens returns only readable tokens",
  logitLens(model, finalLayer.tokens[1].finalHidden, { limit: 20 }).every((entry) =>
    readable.test(entry.token)
  )
);

// The K-neighbour list is where unreadable rare tokens used to surface.
const keyNeighbours = nearestInProjection(
  model,
  projectVocabulary(model, { kind: "key", headIndex: 0, position: satIndex }),
  result.tokens[satIndex].key,
  { limit: 20 }
);
check(
  "K neighbours return only readable tokens",
  keyNeighbours.every((entry) => readable.test(entry.token)),
  keyNeighbours
    .slice(0, 6)
    .map((entry) => entry.token)
    .join(", ")
);

for (let index = 0; index < finalLayer.tokens.length; index += 1) {
  const predictions = logitLens(model, finalLayer.tokens[index].finalHidden, {
    limit: 5,
  });

  console.log(
    `  ${tokens[index].padEnd(6)} → ${predictions
      .map((p) => `${p.token} ${(p.probability * 100).toFixed(1)}%`)
      .join(", ")}`
  );
}

// The lens on the final hidden state of a token should usually recover that
// same token — that is what "weight tying + MLM objective" means.
let recovered = 0;
for (let index = 1; index < finalLayer.tokens.length - 1; index += 1) {
  const top = logitLens(model, finalLayer.tokens[index].finalHidden, { limit: 1 });
  if (top[0].token === tokens[index]) recovered += 1;
}
check(
  "\nlogit lens recovers the input token at most positions",
  recovered >= Math.ceil((finalLayer.tokens.length - 2) * 0.5),
  `${recovered}/${finalLayer.tokens.length - 2}`
);

// ---------------------------------------------------------- [MASK] filling
console.log("\n[MASK] prediction\n");

const maskCases = [
  "the cat sat on the [MASK].",
  "she is a [MASK].",
  "paris is the capital of [MASK].",
];

for (const text of maskCases) {
  const output = runBertAttention(model, text, {
    layerIndex: model.manifest.numLayers - 1,
    headIndex: 0,
  });
  const maskIndex = output.tokens.findIndex((entry) => entry.token === "[MASK]");
  const predictions = logitLens(model, output.tokens[maskIndex].finalHidden, {
    limit: 5,
  });

  console.log(
    `  ${text}\n    ${predictions
      .map((p) => `${p.token} ${(p.probability * 100).toFixed(1)}%`)
      .join(", ")}`
  );
}

const probabilitySum = logitLens(model, finalLayer.tokens[1].finalHidden, {
  limit: 5,
}).reduce((sum, entry) => sum + entry.probability, 0);
check(
  "top-5 probabilities are a fraction of the full distribution",
  probabilitySum > 0 && probabilitySum <= 1,
  `sum ${probabilitySum.toFixed(3)}`
);

console.log(failures === 0 ? "\nall checks passed" : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
