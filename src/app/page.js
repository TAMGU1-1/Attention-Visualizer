"use client";

import { useState, useEffect, useCallback, useMemo, useRef } from "react";

import {
  cosineSimilarity,
  dotProduct,
  multiplyVectorByMatrix,
  softmax,
} from "@/lib/linalg";
import { loadBertModel } from "@/lib/bertModel";
import { runBertAttention } from "@/lib/bertPipeline";
import {
  MINIMUM_WORD_LENGTH,
  logitLens,
  projectVocabulary,
  readValueAsWords,
  searchVocabularyByKey,
  searchVocabularyByQuery,
} from "@/lib/vocabLens";

// GloVe mode: 50d pretrained embeddings, but untrained (random) Wq/Wk/Wv.
const EMBEDDING_DIMENSION = 50;
const QKV_DIMENSION = 8;

function dimensionLabels(length) {
  return Array.from({ length }, (_, index) => `d${index}`);
}

const QKV_HELPER_TEXT = {
  query: "이 토큰이 '무엇을 찾는지'를 나타내는 벡터 — 검색어에 해당합니다.",
  key: "이 토큰이 '무엇을 가졌는지'를 알리는 벡터 — 책 제목에 해당합니다.",
  value: "이 토큰이 '실제로 전달할 내용' — 책 내용에 해당합니다.",
};

function createRandomVector(length) {
  return Array.from({ length }, () =>
    Number((Math.random() * 2 - 1).toFixed(2))
  );
}

function createRandomMatrix(rows, columns) {
  return Array.from({ length: rows }, () =>
    createRandomVector(columns)
  );
}

function findNearestEmbeddingTokens(
  data,
  selectedIndex,
  limit = 5
) {
  return data
    .map((item, index) => ({
      token: item.token,
      index,
      similarity: cosineSimilarity(
        data[selectedIndex].embedding, //QKV가 랜덤이라서 일단 임베딩 벡터로 유사도 결과내는걸로 함
        item.embedding
      ),
    }))
    .filter((item) => item.index !== selectedIndex)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit);
}

function calculateRawScores(data) {
  return data.map((queryItem) =>
    data.map((keyItem) => dotProduct(queryItem.query, keyItem.key))
  );
}

function scaleScores(rawScores) {
  return rawScores.map((row) =>
    row.map((score) => score / Math.sqrt(QKV_DIMENSION))
  );
}

function applySoftmaxRows(scaledScores) {
  return scaledScores.map((row) => softmax(row));
}

function calculateOutputProjection(outputs, outputWeight) {
  return outputs.map((output) =>
    multiplyVectorByMatrix(output, outputWeight)
  );
}

function calculateAttentionOutputs(data, attentionMatrix) {
  return attentionMatrix.map((attentionRow) =>
    Array.from({ length: QKV_DIMENSION }, (_, dimensionIndex) =>
      attentionRow.reduce(
        (sum, attentionWeight, tokenIndex) =>
          sum + attentionWeight * data[tokenIndex].value[dimensionIndex],
        0
      )
    )
  );
}

function VectorList({
  data,
  vectorName,
  emptyMessage,
  textClassName,
}) {
  if (data.length === 0) {
    return <p className="text-gray-400">{emptyMessage}</p>;
  }

  return (
    <div className="space-y-4">
      {data.map((item, index) => (
        <div
          key={`${vectorName}-${item.token}-${index}`}
          className="rounded-lg bg-gray-50 p-3"
        >
          <p className="mb-1 font-semibold">{item.token}</p>

          <p
            className={`break-all font-mono text-sm ${textClassName}`}
          >
            [{item[vectorName].join(", ")}]
          </p>
        </div>
      ))}
    </div>
  );
}

function MatrixHeatmap({
  matrix,
  rowLabels,
  colLabels,
  cornerLabel = "",
  mode = "diverging",
  selectedRowIndex = null,
  onSelectRow,
  compact = false,
  precision = 2,
  helperText,
  collapseAfter = null,
  edgeColumns = 6,
}) {
  const [expanded, setExpanded] = useState(false);

  if (!matrix || matrix.length === 0) {
    return <p className="text-gray-400">계산하면 값이 표시됩니다.</p>;
  }

  // 색 농도는 항상 "행렬 전체" 기준으로 정규화한다.
  // 보이는 열만으로 계산하면 접었다 펼 때 같은 값의 색이 달라진다.
  const maxAbs =
    mode === "diverging"
      ? Math.max(
          1e-6,
          ...matrix.flat().map((value) => Math.abs(value))
        )
      : 1;

  const columnCount = colLabels.length;

  const canCollapse =
    collapseAfter !== null && columnCount > Math.max(collapseAfter, edgeColumns * 2);

  const isCollapsed = canCollapse && !expanded;
  const hiddenCount = isCollapsed ? columnCount - edgeColumns * 2 : 0;

  const visibleColumns = isCollapsed
    ? [
        ...Array.from({ length: edgeColumns }, (_, i) => i),
        "gap",
        ...Array.from(
          { length: edgeColumns },
          (_, i) => columnCount - edgeColumns + i
        ),
      ]
    : Array.from({ length: columnCount }, (_, i) => i);

  function cellStyle(value) {
    if (mode === "sequential") {
      const opacity = Math.min(0.15 + value * 1.5, 1);

      return {
        backgroundColor: `rgba(37, 99, 235, ${opacity})`,
        color: value >= 0.4 ? "white" : "#111827",
      };
    }

    const intensity = Math.abs(value) / maxAbs;
    const opacity = Math.min(0.12 + intensity * 0.85, 1);

    return {
      backgroundColor:
        value >= 0
          ? `rgba(37, 99, 235, ${opacity})`
          : `rgba(225, 29, 72, ${opacity})`,
      color: intensity > 0.55 ? "white" : "#111827",
    };
  }

  // Numbers only survive while the matrix is narrow. Expanded to 128 columns
  // they are noise — the colour pattern is the only readable signal, so cells
  // shrink to strips and the value moves into the tooltip.
  const showValues = !canCollapse || !expanded;

  const cellPadding = !showValues
    ? "px-0 py-2"
    : compact
      ? "px-2 py-1.5 text-xs"
      : "px-4 py-3 text-sm";

  return (
    <div>
      {(helperText || canCollapse) && (
        <div className="mb-3 flex items-start justify-between gap-4">
          <p className="text-sm text-gray-500">{helperText}</p>

          {canCollapse && (
            <button
              type="button"
              onClick={() => setExpanded((value) => !value)}
              className="shrink-0 rounded-md border border-gray-300 px-2.5 py-1 text-xs font-medium text-gray-600 transition hover:border-blue-400 hover:text-blue-700"
            >
              {expanded
                ? `숫자로 보기 (${edgeColumns * 2}열)`
                : `${columnCount}차원 전체를 색으로`}
            </button>
          )}
        </div>
      )}

      <div className="relative">
        <div className="overflow-x-auto rounded-lg border border-gray-200">
          <table className="min-w-max border-collapse text-center">
            <thead>
              <tr>
                <th
                  className={`border border-gray-200 bg-slate-50 font-normal italic text-gray-400 ${cellPadding}`}
                >
                  {cornerLabel}
                </th>

                {visibleColumns.map((columnIndex) =>
                  columnIndex === "gap" ? (
                    <th
                      key="col-gap"
                      onClick={() => setExpanded(true)}
                      title={`Show ${hiddenCount} hidden dimensions`}
                      className={`cursor-pointer border border-gray-200 bg-slate-100 font-mono font-normal text-gray-500 transition hover:bg-blue-50 hover:text-blue-700 ${cellPadding}`}
                    >
                      ⋯ +{hiddenCount} ⋯
                    </th>
                  ) : (
                    <th
                      key={`col-${columnIndex}`}
                      className={`border border-gray-200 bg-slate-50 font-semibold text-gray-600 ${cellPadding} ${
                        showValues ? "" : "w-2 text-[0px]"
                      }`}
                    >
                      {colLabels[columnIndex]}
                    </th>
                  )
                )}
              </tr>
            </thead>

            <tbody>
              {matrix.map((row, rowIndex) => (
                <tr
                  key={`row-${rowIndex}`}
                  onClick={
                    onSelectRow ? () => onSelectRow(rowIndex) : undefined
                  }
                  className={onSelectRow ? "cursor-pointer" : ""}
                >
                  <th
                    className={`border border-gray-200 font-semibold ${cellPadding} ${
                      selectedRowIndex === rowIndex
                        ? "bg-blue-50 text-blue-700"
                        : `bg-slate-50 text-gray-700 ${
                            onSelectRow ? "hover:bg-blue-50" : ""
                          }`
                    }`}
                  >
                    {rowLabels[rowIndex]}
                  </th>

                  {visibleColumns.map((columnIndex) =>
                    columnIndex === "gap" ? (
                      <td
                        key={`cell-${rowIndex}-gap`}
                        className={`border border-gray-200 bg-slate-100 font-mono text-gray-400 ${cellPadding}`}
                      >
                        ⋯
                      </td>
                    ) : (
                      <td
                        key={`cell-${rowIndex}-${columnIndex}`}
                        className={`border-gray-200 font-mono transition ${cellPadding} ${
                          showValues ? "border" : "w-2 border-y"
                        } ${
                          selectedRowIndex === rowIndex
                            ? "ring-2 ring-inset ring-blue-300"
                            : ""
                        }`}
                        style={cellStyle(row[columnIndex])}
                        title={`${rowLabels[rowIndex]} · ${colLabels[columnIndex]}: ${row[
                          columnIndex
                        ].toFixed(4)}`}
                      >
                        {showValues ? row[columnIndex].toFixed(precision) : ""}
                      </td>
                    )
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {compact && (
          <div className="pointer-events-none absolute inset-y-0 right-0 w-10 rounded-r-lg bg-gradient-to-l from-white to-transparent" />
        )}
      </div>
    </div>
  );
}

function FlowConnector({ label }) {
  return (
    <div className="flex flex-col items-center py-2">
      <div className="h-5 w-px bg-gradient-to-b from-gray-300 to-gray-400" />

      <div className="my-1 rounded-full border border-gray-300 bg-white px-3 py-1 font-mono text-xs text-gray-600 shadow-sm">
        {label}
      </div>

      <svg
        width="14"
        height="9"
        viewBox="0 0 14 9"
        className="text-gray-400"
      >
        <path d="M0 0 L7 9 L14 0 Z" fill="currentColor" />
      </svg>
    </div>
  );
}

function FlowStage({ title, formula, children }) {
  return (
    <div className="rounded-lg border border-gray-200 p-5">
      <div className="mb-4">
        <h2 className="text-xl font-bold">{title}</h2>

        {formula && (
          <p className="mt-1 font-mono text-sm tracking-tight text-gray-500">
            {formula}
          </p>
        )}
      </div>

      {children}
    </div>
  );
}



function ResultCard({ index, title, hint, action, children }) {
  return (
    <section className="rounded-xl border border-gray-200 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-bold">
            <span className="mr-2 text-blue-600">{index}</span>
            {title}
          </h3>

          {hint && <p className="mt-1 text-sm text-gray-500">{hint}</p>}
        </div>

        {action}
      </div>

      <div className="mt-4">{children}</div>
    </section>
  );
}

function TokenStrip({ tokens, selectedIndex, onSelect }) {
  if (tokens.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {tokens.map((entry, index) => (
        <button
          key={`${entry.token}-${index}`}
          type="button"
          onClick={() => onSelect(index)}
          className={`rounded-md px-2.5 py-1 font-mono text-sm transition ${
            selectedIndex === index
              ? "bg-blue-600 text-white"
              : entry.isSpecial
                ? "bg-gray-100 text-gray-400 hover:bg-blue-50 hover:text-blue-700"
                : "bg-gray-100 text-gray-700 hover:bg-blue-50 hover:text-blue-700"
          }`}
        >
          {entry.token}
        </button>
      ))}
    </div>
  );
}

function CollapsibleSection({ title, subtitle, children, defaultOpen = false }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="mt-8 rounded-xl border border-gray-200">
      <button
        type="button"
        onClick={() => setIsOpen((value) => !value)}
        className="flex w-full items-center justify-between gap-4 px-5 py-4 text-left transition hover:bg-slate-50"
      >
        <span>
          <span className="text-lg font-bold">{title}</span>
          {subtitle && (
            <span className="ml-3 text-sm font-normal text-gray-500">
              {subtitle}
            </span>
          )}
        </span>

        <span className="shrink-0 text-sm font-medium text-blue-700">
          {isOpen ? "접기 ▴" : "펼치기 ▾"}
        </span>
      </button>

      {isOpen && <div className="border-t border-gray-200 p-5">{children}</div>}
    </div>
  );
}

function TabBar({ tabs, active, onChange }) {
  return (
    <div className="flex flex-wrap gap-2">
      {tabs.map((tab) => (
        <button
          key={tab.key}
          type="button"
          onClick={() => onChange(tab.key)}
          className={`rounded-md border px-3 py-1.5 text-sm font-medium transition ${
            active === tab.key
              ? "border-blue-600 bg-blue-600 text-white"
              : "border-gray-300 bg-white text-gray-600 hover:bg-blue-50"
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

function ScoreBar({ label, score, width, tone = "blue", suffix }) {
  const toneClassName =
    tone === "orange" ? "bg-orange-500" : tone === "green" ? "bg-emerald-500" : "bg-blue-600";

  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-medium">{label}</span>
        <span className="font-mono text-sm text-gray-500">{suffix}</span>
      </div>

      <div className="mt-1 h-2 overflow-hidden rounded-full bg-gray-200">
        <div
          className={`h-full rounded-full ${toneClassName}`}
          style={{ width: `${Math.max(Math.min(width, 100), 1)}%` }}
        />
      </div>
    </div>
  );
}

/**
 * A "##" result is a word fragment, and on its own it reads like noise. Glued
 * to the selected token it usually spells a real word — cat + ##ania →
 * catania, visual + ##ization → visualization — which is the whole point when a
 * head is hunting for the piece that follows (or precedes) a token.
 */
function joinFragment(readingKey, selectedToken, candidate) {
  const selectedIsFragment = selectedToken.startsWith("##");
  const candidateIsFragment = candidate.startsWith("##");

  // "what am I looking for" — the candidate would come after me
  if (readingKey === "query" && candidateIsFragment && !selectedIsFragment) {
    return selectedToken + candidate.slice(2);
  }

  // "who is looking for me" — the candidate would come before me
  if (readingKey === "key" && selectedIsFragment && !candidateIsFragment) {
    return candidate + selectedToken.slice(2);
  }

  return null;
}

// Each of Q, K and V gets read back into words through the product it actually
// participates in, so none of the three depends on how the model happens to be
// parameterised: Q and K through W_Q·W_Kᵀ, V through W_V·W_O.
const VECTOR_READINGS = [
  {
    key: "query",
    label: "Q",
    title: "내가 찾는 것",
    detail: "이 토큰의 Query에 가장 잘 걸리는 Key를 가진 단어",
    tone: "blue",
  },
  {
    key: "key",
    label: "K",
    title: "나를 찾는 것",
    detail: "이 토큰의 Key를 가장 잘 집어내는 Query를 가진 단어",
    tone: "orange",
  },
  {
    key: "value",
    label: "V",
    title: "내가 전하는 것",
    detail: "주목받았을 때 이 토큰이 실제로 넘기는 내용 (v·W_O를 단어로 디코딩)",
    tone: "green",
  },
];

function VocabularyLens({ model, tokenData, selectedQueryIndex, layerIndex, headIndex }) {
  const [hideFragments, setHideFragments] = useState(true);
  const [computed, setComputed] = useState(null);
  const [isComputing, setIsComputing] = useState(false);
  const cache = useRef(new Map());

  const selectedToken = tokenData[selectedQueryIndex];

  // Results are tagged with the inputs that produced them, so a stale result is
  // discarded during render rather than cleared by an effect.
  const signature = `${hideFragments}|${layerIndex}|${headIndex}|${selectedQueryIndex}|${tokenData
    .map((entry) => entry.id)
    .join(",")}`;
  const readings = computed?.signature === signature ? computed.readings : null;

  if (!model || !selectedToken) return null;

  const isLayerZero = layerIndex === 0;

  function projectedFor(kind) {
    const cacheKey = `${kind}-${headIndex}-${selectedQueryIndex}`;

    if (!cache.current.has(cacheKey)) {
      cache.current.set(
        cacheKey,
        projectVocabulary(model, {
          kind,
          headIndex,
          position: selectedQueryIndex,
        })
      );
    }

    return cache.current.get(cacheKey);
  }

  function compute() {
    setIsComputing(true);

    // Yield one frame so the button can repaint before the ~1s sweep.
    requestAnimationFrame(() => {
      const minimumLength = hideFragments ? MINIMUM_WORD_LENGTH : 1;
      const options = { limit: 6, minimumLength };

      setComputed({
        signature,
        readings: {
          // "what am I looking for" — my Query against every word's Key
          query: searchVocabularyByQuery(
            model,
            selectedToken.query,
            projectedFor("key"),
            options
          ),
          // "who is looking for me" — every word's Query against my Key
          key: searchVocabularyByKey(
            model,
            selectedToken.key,
            projectedFor("query"),
            options
          ),
          // "what do I hand over" — v·W_O decoded through the MLM head
          value: readValueAsWords(model, selectedToken.value, {
            layerIndex,
            headIndex,
            limit: 6,
          }),
        },
      });

      setIsComputing(false);
    });
  }

  return (
    <ResultCard
      index="2"
      title={
        <>
          <span className="font-mono text-blue-700">{selectedToken.token}</span>{" "}
          의 Q · K · V 를 단어로 읽으면
        </>
      }
      hint={
        <>
          <span className="font-medium text-gray-600">
            세 벡터가 각각 어떤 단어를 가리키는지 읽어냅니다.
          </span>{" "}
          Q와 K는 어휘 30,522개와 대조해서, V는 전달 내용을 직접 디코딩해서
          구합니다.
        </>
      }
    >
      {!isLayerZero ? (
        <p className="rounded-md bg-amber-50 p-3 text-sm text-amber-800">
          layer 0에서만 가능합니다. 깊은 층에서는 한 토큰의 입력이 문장 전체에
          의존해서, 어휘의 단어를 하나씩 넣어보는 방식이 성립하지 않습니다.
        </p>
      ) : readings === null ? (
        <button
          type="button"
          onClick={compute}
          disabled={isComputing}
          className="rounded-lg border border-blue-600 px-4 py-2 text-sm font-semibold text-blue-700 transition hover:bg-blue-50 disabled:cursor-wait disabled:opacity-60"
        >
          {isComputing ? "어휘 30,522개 훑는 중…" : "Q · K · V 읽어내기"}
        </button>
      ) : (
        <>
          <div className="space-y-5">
            {VECTOR_READINGS.map((reading) => {
              const entries = readings[reading.key];
              const isValue = reading.key === "value";
              const maximum = isValue
                ? entries[0]?.probability || 1
                : Math.abs(entries[0]?.score || 1);

              return (
                <div key={reading.key}>
                  <div className="mb-2 flex flex-wrap items-baseline gap-x-2">
                    <span
                      className={`rounded px-1.5 py-0.5 font-mono text-sm font-bold ${
                        reading.tone === "blue"
                          ? "bg-blue-100 text-blue-700"
                          : reading.tone === "orange"
                            ? "bg-orange-100 text-orange-700"
                            : "bg-emerald-100 text-emerald-700"
                      }`}
                    >
                      {reading.label}
                    </span>

                    <span className="font-semibold">{reading.title}</span>

                    <span className="text-xs text-gray-400">
                      {reading.detail}
                    </span>
                  </div>

                  <div className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
                    {entries.map((entry, index) => {
                      const joined = joinFragment(
                        reading.key,
                        selectedToken.token,
                        entry.token
                      );

                      return (
                        <ScoreBar
                          key={`${reading.key}-${entry.id}`}
                          label={
                            joined ? (
                              <>
                                {index + 1}. {joined}{" "}
                                <span className="font-mono text-xs font-normal text-gray-400">
                                  {entry.token}
                                </span>
                              </>
                            ) : (
                              `${index + 1}. ${entry.token}`
                            )
                          }
                          width={
                            ((isValue
                              ? entry.probability
                              : Math.abs(entry.score)) /
                              maximum) *
                            100
                          }
                          tone={reading.tone}
                          suffix={
                            isValue
                              ? `${(entry.probability * 100).toFixed(1)}%`
                              : entry.score.toFixed(2)
                          }
                        />
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>

          <label className="mt-5 flex items-center gap-2 text-sm text-gray-600">
            <input
              type="checkbox"
              checked={hideFragments}
              onChange={(event) => setHideFragments(event.target.checked)}
              className="h-4 w-4"
            />
            3글자 이하 단어 조각 숨기기
            <span className="text-xs text-gray-400">
              (cy, fe, peg 같은 조각이 상위를 차지하는 것을 막습니다)
            </span>
          </label>
        </>
      )}
    </ResultCard>
  );
}

function LogitLensPanel({ model, tokenData, selectedQueryIndex, layerIndex }) {
  const selectedToken = tokenData[selectedQueryIndex];

  // At layer 0 the input *is* the word embedding, so decoding it just returns
  // the same word at ~100% — a tautology, not a reading. Only worth showing
  // once an actual layer has transformed it.
  const showLayerInput = layerIndex > 0;

  const readings = useMemo(() => {
    if (!model || !selectedToken) return null;

    return {
      layerInput: showLayerInput
        ? logitLens(model, selectedToken.embedding, { limit: 6 })
        : null,
      final: logitLens(model, selectedToken.finalHidden, { limit: 6 }),
    };
  }, [model, selectedToken, showLayerInput]);

  if (!readings) return null;

  const isMask = selectedToken.token === "[MASK]";

  return (
    <ResultCard
      index="3"
      title={
        isMask ? (
          <>
            <span className="font-mono text-emerald-700">[MASK]</span> 자리에
            들어갈 단어
          </>
        ) : (
          <>
            이 자리의 벡터를 단어로 되돌리면
          </>
        )
      }
      hint={
        isMask
          ? "모델이 실제로 빈칸을 채운 결과입니다."
          : "은닉 벡터를 어휘 분포로 디코딩합니다. 문장에 [MASK]를 넣고 그 토큰을 고르면 빈칸 채우기가 됩니다."
      }
    >
      <div className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
        {[
          showLayerInput && [
            `layer ${layerIndex} 입력 시점`,
            readings.layerInput,
            "orange",
          ],
          ["전 층 통과 후", readings.final, "green"],
        ]
          .filter(Boolean)
          .map(([title, entries, tone]) => (
            <div key={title}>
              <h4 className="mb-2 text-sm font-semibold text-gray-700">
                {title}
              </h4>

              <div className="space-y-2">
                {entries.map((entry, index) => (
                  <ScoreBar
                    key={entry.id}
                    label={`${index + 1}. ${entry.token}`}
                    width={entry.probability * 100}
                    tone={tone}
                    suffix={`${(entry.probability * 100).toFixed(1)}%`}
                  />
                ))}
              </div>
            </div>
          ))}
      </div>

      {!showLayerInput && (
        <p className="mt-4 text-xs text-gray-400">
          layer 1 이상을 고르면 &ldquo;층에 들어갈 때&rdquo; 시점의 판독도 함께
          나옵니다. layer 0 입력은 단어 임베딩 그 자체라 자기 자신이 나올 수밖에
          없어서 생략했습니다.
        </p>
      )}
    </ResultCard>
  );
}

function ModelControls({
  modelSource,
  onChangeModelSource,
  manifest,
  layerIndex,
  onChangeLayer,
  headIndex,
  onChangeHead,
  loadingStage,
}) {
  const isBert = modelSource === "bert";

  const selectClassName =
    "rounded-md border border-gray-300 bg-white px-2 py-1 text-sm font-mono outline-none focus:border-blue-500";

  return (
    <div className="mb-6 rounded-lg border border-gray-200 bg-slate-50 p-4">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-gray-700">Weights</span>

          <div className="flex overflow-hidden rounded-md border border-gray-300">
            {[
              ["bert", "Pretrained BERT"],
              ["glove", "GloVe + random"],
            ].map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => onChangeModelSource(value)}
                className={`px-3 py-1.5 text-sm font-medium transition ${
                  modelSource === value
                    ? "bg-blue-600 text-white"
                    : "bg-white text-gray-600 hover:bg-blue-50"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {isBert && manifest && (
          <>
            <label className="flex items-center gap-2 text-sm text-gray-700">
              Layer
              <select
                className={selectClassName}
                value={layerIndex}
                onChange={(event) => onChangeLayer(Number(event.target.value))}
              >
                {Array.from({ length: manifest.numLayers }, (_, index) => (
                  <option key={index} value={index}>
                    {index}
                  </option>
                ))}
              </select>
            </label>

            <label className="flex items-center gap-2 text-sm text-gray-700">
              Head
              <select
                className={selectClassName}
                value={headIndex}
                onChange={(event) => onChangeHead(Number(event.target.value))}
              >
                {Array.from({ length: manifest.numHeads }, (_, index) => (
                  <option key={index} value={index}>
                    {index}
                  </option>
                ))}
              </select>
            </label>
          </>
        )}

        {loadingStage && (
          <span className="text-sm text-gray-500">
            {loadingStage} 불러오는 중…
          </span>
        )}
      </div>

      <p className="mt-3 text-sm text-gray-500">
        {isBert ? (
          manifest ? (
            <>
              <span className="font-mono">{manifest.modelId}</span> —{" "}
              {manifest.hiddenSize}차원 · {manifest.numLayers}개 층 ·{" "}
              {manifest.numHeads}개 헤드 · head_dim {manifest.headDim}. 모든
              값이 학습된 체크포인트에서 나오므로 같은 문장은 항상 같은 결과가
              됩니다.
            </>
          ) : (
            "학습된 가중치를 받는 중입니다 (약 9MB, 이후에는 캐시를 씁니다)…"
          )
        ) : (
          <>
            GloVe 임베딩은 학습된 것이지만 W<sub>Q</sub>/W<sub>K</sub>/W
            <sub>V</sub>는 실행할 때마다 새로 뽑는 난수입니다 — 계산은 진짜지만
            attention 패턴에는 의미가 없습니다. 학습 전/후 비교용입니다.
          </>
        )}
      </p>
    </div>
  );
}

function AttentionDistribution({
  data,
  attentionMatrix,
  selectedQueryIndex,
}) {
  if (
    data.length === 0 ||
    attentionMatrix.length === 0 ||
    selectedQueryIndex === null
  ) {
    return null;
  }

  const selectedRow = attentionMatrix[selectedQueryIndex];
  const ranked = [...selectedRow.keys()].sort(
    (a, b) => selectedRow[b] - selectedRow[a]
  );

  return (
    <ResultCard
      index="1"
      title={
        <>
          <span className="font-mono text-blue-700">
            {data[selectedQueryIndex].token}
          </span>{" "}
          이 주목하는 곳
        </>
      }
      hint="이 토큰의 Query가 문장 안 각 토큰에 배분한 비율입니다. 합은 100%."
    >
      <div className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
        {ranked.map((index) => (
          <ScoreBar
            key={`bar-${index}`}
            label={data[index].token}
            width={selectedRow[index] * 100}
            tone={index === selectedQueryIndex ? "green" : "blue"}
            suffix={`${(selectedRow[index] * 100).toFixed(1)}%`}
          />
        ))}
      </div>
    </ResultCard>
  );
}

function OutputSimilarWords({
  data,
  selectedQueryIndex,
}) {
  if (
    data.length === 0 ||
    selectedQueryIndex === null
  ) {
    return null;
  }

  const similarWords =
    findNearestEmbeddingTokens(
      data,
      selectedQueryIndex
    );

  return (
    <div className="mt-8 rounded-lg border p-5">
      <h2 className="text-xl font-bold">
        Attention Output Similar Words
      </h2>

      <p className="mt-2 text-sm text-gray-500">
        Each token&apos;s embedding is compared with all other token embeddings
        using cosine similarity to find the most similar words. Note this
        compares the vectors that <em>enter</em> attention, not the projected
        output.
      </p>

      <p className="mt-2 text-sm text-gray-500">
        Selected Token :
        <span className="font-semibold text-blue-700">
          {" "}
          {data[selectedQueryIndex].token}
        </span>
      </p>

      <div className="mt-5 space-y-3">
        {similarWords.map((item, index) => (
          <div
            key={`${item.token}-${item.index}`}
            className="rounded-lg bg-gray-50 p-3"
          >
            <div className="flex justify-between">
              <span>
                {index + 1}. {item.token}
              </span>

              <span className="font-mono">
                {item.similarity.toFixed(4)}
              </span>
            </div>

            <div className="mt-2 h-2 overflow-hidden rounded-full bg-gray-200">
              <div
                className="h-full rounded-full bg-orange-500"
                style={{
                  width: `${((item.similarity + 1) / 2) * 100}%`,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Home() {
  const [sentence, setSentence] = useState("The cat sat on the chair.");
  const [tokenData, setTokenData] = useState([]);
  const [rawScores, setRawScores] = useState([]);
  const [scaledScores, setScaledScores] = useState([]);
  const [attentionMatrix, setAttentionMatrix] = useState([]);
  const [selectedQueryIndex, setSelectedQueryIndex] =
    useState(null);
  const [error, setError] = useState("");

  // "bert" = pretrained weights, "glove" = GloVe embeddings + random Wq/Wk/Wv.
  const [modelSource, setModelSource] = useState("bert");
  const [layerIndex, setLayerIndex] = useState(0);
  const [headIndex, setHeadIndex] = useState(0);

  const [qkvTab, setQkvTab] = useState("query");
  const [scoreTab, setScoreTab] = useState("raw");

  // 처음 딱 한 번만 사전 담은 useState 실행해서 받아온 걸 저장
  const [embeddings, setEmbeddings] = useState(null);
  const [bertModel, setBertModel] = useState(null);
  const [loadingStage, setLoadingStage] = useState(null);

  // Each mode's assets are fetched only when that mode is first selected —
  // the BERT weights are ~9MB and the GloVe dictionary ~4MB.
  useEffect(() => {
    let cancelled = false;

    if (modelSource === "glove" && !embeddings) {
      fetch("/glove-50d.json")
        .then((res) => res.json())
        .then((dict) => {
          if (cancelled) return;
          setEmbeddings(dict);
          setLoadingStage(null);
        })
        .catch(() => {
          if (!cancelled) setLoadingStage(null);
        });
    }

    if (modelSource === "bert" && !bertModel) {
      loadBertModel(undefined, (stage) => {
        if (!cancelled) setLoadingStage(stage);
      })
        .then((model) => {
          if (cancelled) return;
          setBertModel(model);
          setLoadingStage(null);
        })
        .catch((cause) => {
          if (cancelled) return;
          setError(`Could not load the BERT weights: ${cause.message}`);
          setLoadingStage(null);
        });
    }

    return () => {
      cancelled = true;
    };
  }, [modelSource, embeddings, bertModel]);

  const isReady =
    modelSource === "bert" ? bertModel !== null : embeddings !== null;

  // Derived rather than stored, so the effect above never calls setState
  // synchronously just to show the first label.
  const loadingLabel = isReady
    ? null
    : (loadingStage ??
      (modelSource === "bert" ? "manifest" : "GloVe dictionary"));

  function clearResults() {
    setTokenData([]);
    setRawScores([]);
    setScaledScores([]);
    setAttentionMatrix([]);
    setSelectedQueryIndex(null);
  }

  const calculateWithBert = useCallback(
    (text, layer, head) => {
      const result = runBertAttention(bertModel, text, {
        layerIndex: layer,
        headIndex: head,
      });

      setTokenData(
        result.tokens.map((entry) => ({
          ...entry,
          isOOV: entry.id === bertModel.tokenizer.unknownId,
        }))
      );
      setRawScores(result.rawScores);
      setScaledScores(result.scaledScores);
      setAttentionMatrix(result.attentionMatrix);
      setSelectedQueryIndex((current) =>
        current === null || current >= result.tokens.length ? 0 : current
      );
    },
    [bertModel]
  );

  // Switching layer/head re-runs the sentence already on screen. Done in the
  // change handler rather than an effect — it is a user action, not a sync.
  function recalculateWith(layer, head) {
    if (!bertModel || tokenData.length === 0) return;
    calculateWithBert(sentence, layer, head);
  }

  function handleCalculate() {
    const trimmedSentence = sentence.trim();

    if (trimmedSentence === "") {
      clearResults();
      setError("문장을 입력하세요.");
      return;
    }

    setError("");

    if (modelSource === "bert") {
      calculateWithBert(trimmedSentence, layerIndex, headIndex);
      return;
    }

    const tokens = trimmedSentence.split(/\s+/);

    const queryWeight = createRandomMatrix(
      EMBEDDING_DIMENSION,
      QKV_DIMENSION
    );

    const keyWeight = createRandomMatrix(
      EMBEDDING_DIMENSION,
      QKV_DIMENSION
    );

    const valueWeight = createRandomMatrix(
      EMBEDDING_DIMENSION,
      QKV_DIMENSION
    );

    const outputWeight = createRandomMatrix(
      QKV_DIMENSION,
      EMBEDDING_DIMENSION
    );

    const calculatedData = tokens.map((token) => {
      const lookupKey = token
        .toLowerCase()
        .replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, ""); //양 끝 문장부호 제거 후 GloVe 소문자 사전에서 조회
      const found = embeddings[lookupKey]; //사전에서 벡터 꺼내기
      const embedding = found ?? createRandomVector(EMBEDDING_DIMENSION);
      const isOOV = !found;

      const query = multiplyVectorByMatrix(
        embedding,
        queryWeight
      );

      const key = multiplyVectorByMatrix(
        embedding,
        keyWeight
      );

      const value = multiplyVectorByMatrix(
        embedding,
        valueWeight
      );

      return {
        token,
        isOOV,
        embedding,
        query,
        key,
        value,
      };
    });

    const calculatedRawScores = calculateRawScores(calculatedData);
    const calculatedScaledScores = scaleScores(calculatedRawScores);
    const calculatedAttentionMatrix = applySoftmaxRows(
      calculatedScaledScores
    );

    const attentionOutputs = calculateAttentionOutputs(
      calculatedData,
      calculatedAttentionMatrix
    );

    const projectedOutputs = calculateOutputProjection(
      attentionOutputs,
      outputWeight
    );

    const finalData = calculatedData.map((item, index) => ({
      ...item,
      output: attentionOutputs[index],
      projectedOutput: projectedOutputs[index],
    }));

    setTokenData(finalData);
    setRawScores(calculatedRawScores);
    setScaledScores(calculatedScaledScores);
    setAttentionMatrix(calculatedAttentionMatrix);
    setSelectedQueryIndex(0);
  }

  const tokenLabels = tokenData.map((item) =>
    item.isOOV ? `${item.token} *` : item.token
  );
  const hasOOVTokens = tokenData.some((item) => item.isOOV);

  const isBert = modelSource === "bert";

  // Dimensions come from the data itself: 50/8 for GloVe, 128/64 for bert-tiny.
  const embeddingDimension =
    tokenData[0]?.embedding.length ??
    (isBert ? bertModel?.manifest.hiddenSize ?? 128 : EMBEDDING_DIMENSION);
  const qkvDimension =
    tokenData[0]?.query.length ??
    (isBert ? bertModel?.manifest.headDim ?? 64 : QKV_DIMENSION);

  const embeddingColumnLabels = dimensionLabels(embeddingDimension);
  const qkvColumnLabels = dimensionLabels(qkvDimension);

  return (
    <main className="min-h-screen bg-gray-100 p-6 md:p-10">
      <div className="mx-auto max-w-6xl rounded-xl bg-white p-6 shadow-lg md:p-8">
        <h1 className="mb-3 text-center text-4xl font-bold">
          Transformer Visualizer
        </h1>

        <p className="mb-6 text-center text-gray-600">
          문장을 입력하면 Transformer의 Self-Attention 계산을 단계별로 보여줍니다.
        </p>

        <ModelControls
          modelSource={modelSource}
          onChangeModelSource={(next) => {
            setModelSource(next);
            setLayerIndex(0);
            setHeadIndex(0);
            clearResults();
            setError("");
          }}
          manifest={bertModel?.manifest}
          layerIndex={layerIndex}
          onChangeLayer={(next) => {
            setLayerIndex(next);
            recalculateWith(next, headIndex);
          }}
          headIndex={headIndex}
          onChangeHead={(next) => {
            setHeadIndex(next);
            recalculateWith(layerIndex, next);
          }}
          loadingStage={loadingLabel}
        />

        <textarea
          className="w-full rounded-lg border border-gray-300 p-4 outline-none focus:border-blue-500"
          rows={2}
          placeholder="영어 문장을 입력하세요. 예: The cat sat on the chair."
          value={sentence}
          onChange={(event) => setSentence(event.target.value)}
        />

        {isBert && (
          <div className="mt-2 space-y-1 text-sm text-gray-500">
            <p>
              문장 어디에나 <span className="font-mono">[MASK]</span>를 넣을 수
              있습니다 —{" "}
              <button
                type="button"
                onClick={() => setSentence("The cat sat on the [MASK].")}
                className="font-mono text-blue-700 underline decoration-dotted underline-offset-2"
              >
                The cat sat on the [MASK].
              </button>{" "}
              그 토큰을 고르면 모델이 빈칸을 채웁니다.
            </p>

            <p>
              Query가 실제로 &ldquo;검색&rdquo;하는 것을 보려면 —{" "}
              <button
                type="button"
                onClick={() => {
                  setSentence("The visualizing kitten sat.");
                  setLayerIndex(0);
                  setHeadIndex(1);
                }}
                className="font-mono text-blue-700 underline decoration-dotted underline-offset-2"
              >
                The visualizing kitten sat.
              </button>{" "}
              (layer 0 · head 1) — <span className="font-mono">visual</span>이
              뒤에 붙을 조각 <span className="font-mono">##izing</span>을
              찾아냅니다.
            </p>
          </div>
        )}

        {error && (
          <p className="mt-2 text-sm text-red-600">{error}</p>
        )}

        <button
          onClick={handleCalculate}
          disabled={!isReady}
          className="mt-4 rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-300"
        >
          {isReady ? "Attention 계산" : "모델 불러오는 중…"}
        </button>

        {tokenData.length > 0 && selectedQueryIndex !== null && (
          <div className="mt-8 border-t border-gray-200 pt-6">
            <p className="mb-2 text-sm font-semibold text-gray-700">
              토큰을 눌러 추적할 대상을 바꾸세요
            </p>

            <TokenStrip
              tokens={tokenData}
              selectedIndex={selectedQueryIndex}
              onSelect={setSelectedQueryIndex}
            />

            <div className="mt-6 space-y-4">
              <AttentionDistribution
                data={tokenData}
                attentionMatrix={attentionMatrix}
                selectedQueryIndex={selectedQueryIndex}
              />

              {isBert ? (
                <>
                  <VocabularyLens
                    model={bertModel}
                    tokenData={tokenData}
                    selectedQueryIndex={selectedQueryIndex}
                    layerIndex={layerIndex}
                    headIndex={headIndex}
                  />

                  <LogitLensPanel
                    model={bertModel}
                    tokenData={tokenData}
                    selectedQueryIndex={selectedQueryIndex}
                    layerIndex={layerIndex}
                  />
                </>
              ) : (
                <OutputSimilarWords
                  data={tokenData}
                  selectedQueryIndex={selectedQueryIndex}
                />
              )}
            </div>
          </div>
        )}

        {tokenData.length > 0 && (
        <CollapsibleSection
          title="계산 과정 보기"
          subtitle="위 결과가 나오기까지의 7단계 행렬"
        >
        <div className="flex flex-col">
          <FlowStage
            title="Input Embedding"
            formula={
              isBert ? (
                <>
                  LayerNorm(E<sub>token</sub> + E<sub>position</sub> + E
                  <sub>segment</sub>) · dimension {embeddingDimension}
                  {layerIndex > 0 && ` → hidden state entering layer ${layerIndex}`}
                </>
              ) : (
                <>GloVe embedding · dimension {embeddingDimension}</>
              )
            }
          >
            <MatrixHeatmap
              matrix={tokenData.map((item) => item.embedding)}
              rowLabels={tokenLabels}
              colLabels={embeddingColumnLabels}
              cornerLabel="token \\ dim"
              compact
              collapseAfter={16}
              selectedRowIndex={selectedQueryIndex}
              onSelectRow={setSelectedQueryIndex}
              helperText={
                isBert
                  ? layerIndex > 0
                    ? `Hidden state entering layer ${layerIndex} — the output of every earlier layer (attention + feed-forward).`
                    : "Token, position and segment embeddings summed, then LayerNorm. [CLS]/[SEP] are BERT's own special tokens."
                  : hasOOVTokens
                    ? "각 행이 토큰 하나의 학습된 임베딩 벡터입니다. * = GloVe 1만 단어 사전에 없어서 난수 벡터로 대체된 단어."
                    : "각 행이 토큰 하나의 학습된 임베딩 벡터입니다."
              }
            />
          </FlowStage>

          <FlowConnector label="× Wq / Wk / Wv" />

          <FlowStage
            title={
              isBert
                ? `Query · Key · Value — layer ${layerIndex}, head ${headIndex}`
                : "Query · Key · Value"
            }
            formula={
              isBert ? (
                <>
                  Q = X·W<sub>Q</sub> + b<sub>Q</sub> &nbsp; K = X·W<sub>K</sub>{" "}
                  + b<sub>K</sub> &nbsp; V = X·W<sub>V</sub> + b<sub>V</sub>{" "}
                  &nbsp;·&nbsp; sliced to head {headIndex} ({qkvDimension} of{" "}
                  {embeddingDimension} dims)
                </>
              ) : (
                <>
                  Q = X·W<sub>Q</sub> &nbsp; K = X·W<sub>K</sub> &nbsp; V =
                  X·W<sub>V</sub>
                </>
              )
            }
          >
            <TabBar
              tabs={[
                { key: "query", label: "Query (Q)" },
                { key: "key", label: "Key (K)" },
                { key: "value", label: "Value (V)" },
              ]}
              active={qkvTab}
              onChange={setQkvTab}
            />

            <div className="mt-4">
              <MatrixHeatmap
                matrix={tokenData.map((item) => item[qkvTab])}
                rowLabels={tokenLabels}
                colLabels={qkvColumnLabels}
                cornerLabel="tok \\ dim"
                collapseAfter={16}
                selectedRowIndex={selectedQueryIndex}
                onSelectRow={setSelectedQueryIndex}
                helperText={QKV_HELPER_TEXT[qkvTab]}
              />
            </div>
          </FlowStage>

          <FlowConnector
            label={
              <>
                Q · K<sup>T</sup> ÷ √d<sub>k</sub>
              </>
            }
          />

          <FlowStage
            title="Attention Scores"
            formula={
              <>
                score<sub>ij</sub> = q<sub>i</sub> · k<sub>j</sub> &nbsp;→&nbsp;
                ÷ √{qkvDimension}
              </>
            }
          >
            <TabBar
              tabs={[
                { key: "raw", label: "Raw" },
                { key: "scaled", label: `÷ √${qkvDimension}` },
              ]}
              active={scoreTab}
              onChange={setScoreTab}
            />

            <div className="mt-4">
              <MatrixHeatmap
                matrix={scoreTab === "raw" ? rawScores : scaledScores}
                rowLabels={tokenLabels}
                colLabels={tokenLabels}
                cornerLabel="Q \\ K"
                selectedRowIndex={selectedQueryIndex}
                onSelectRow={setSelectedQueryIndex}
                helperText={
                  scoreTab === "raw"
                    ? "모든 Query와 모든 Key의 내적. 아직 스케일링 전입니다."
                    : `√${qkvDimension}로 나눈 값. 다음 단계의 Softmax가 한쪽으로 쏠리는 것을 막습니다.`
                }
              />
            </div>
          </FlowStage>

          <FlowConnector label="Softmax (row-wise)" />

          <FlowStage
            title="Attention Weights"
            formula={
              <>
                Attention(Q, K, V) = Softmax(QK<sup>T</sup> / √d
                <sub>k</sub>)V
              </>
            }
          >
            <MatrixHeatmap
              matrix={attentionMatrix}
              rowLabels={tokenLabels}
              colLabels={tokenLabels}
              cornerLabel="Q \\ K"
              mode="sequential"
              precision={4}
              selectedRowIndex={selectedQueryIndex}
              onSelectRow={setSelectedQueryIndex}
              helperText="행이 Query 토큰, 열이 Key 토큰. 각 행의 합은 1이라 그대로 비율로 읽힙니다."
            />
          </FlowStage>

          <FlowConnector label="× V (weighted sum)" />

          <FlowStage
            title="Attention Output"
            formula={<>Output = Attention Weights × V</>}
          >
            <MatrixHeatmap
              matrix={tokenData.map((item) => item.output)}
              rowLabels={tokenLabels}
              colLabels={qkvColumnLabels}
              cornerLabel="tok \\ dim"
              collapseAfter={16}
              selectedRowIndex={selectedQueryIndex}
              onSelectRow={setSelectedQueryIndex}
            />
          </FlowStage>

          <FlowConnector
            label={
              <>
                × W<sub>O</sub> (output projection)
              </>
            }
          />

          <FlowStage
            title="Output Projection"
            formula={
              <>
                Projected = Output · W<sub>O</sub> (dimension{" "}
                {embeddingDimension})
              </>
            }
          >
            <MatrixHeatmap
              matrix={tokenData.map((item) => item.projectedOutput)}
              rowLabels={tokenLabels}
              colLabels={embeddingColumnLabels}
              cornerLabel="tok \\ dim"
              compact
              collapseAfter={16}
              selectedRowIndex={selectedQueryIndex}
              onSelectRow={setSelectedQueryIndex}
              helperText={
                isBert
                  ? `head ${headIndex}의 W_O 부분이 ${qkvDimension}차원 출력을 ${embeddingDimension}차원으로 되돌립니다. 실제 모델에서는 모든 head의 결과를 합친 뒤 bias·residual·LayerNorm이 붙습니다.`
                  : "Attention 출력을 임베딩 차원으로 되돌린 값입니다."
              }
            />
          </FlowStage>
        </div>
        </CollapsibleSection>
        )}
      </div>
    </main>
  );
}
