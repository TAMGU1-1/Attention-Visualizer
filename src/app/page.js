"use client";

import { useState, useEffect } from "react";

const EMBEDDING_DIMENSION = 50;
const QKV_DIMENSION = 8;

const EMBEDDING_COLUMN_LABELS = Array.from(
  { length: EMBEDDING_DIMENSION },
  (_, index) => `d${index}`
);

const QKV_COLUMN_LABELS = Array.from(
  { length: QKV_DIMENSION },
  (_, index) => `d${index}`
);

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

function multiplyVectorByMatrix(vector, matrix) {
  return matrix[0].map((_, columnIndex) => {
    const result = vector.reduce(
      (sum, value, rowIndex) =>
        sum + value * matrix[rowIndex][columnIndex],
      0
    );

    return Number(result.toFixed(2));
  });
}

function dotProduct(vectorA, vectorB) {
  return vectorA.reduce(
    (sum, value, index) => sum + value * vectorB[index],
    0
  );
}

function vectorNorm(vector) {
  return Math.sqrt(
    vector.reduce((sum, value) => sum + value * value, 0)
  );
}

function cosineSimilarity(vectorA, vectorB) {
  const denominator =
    vectorNorm(vectorA) * vectorNorm(vectorB);

  if (denominator === 0) {
    return 0;
  }

  return dotProduct(vectorA, vectorB) / denominator;
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

function softmax(values) {
  const maximumValue = Math.max(...values);

  const exponentials = values.map((value) =>
    Math.exp(value - maximumValue)
  );

  const exponentialSum = exponentials.reduce(
    (sum, value) => sum + value,
    0
  );

  return exponentials.map((value) =>
    Number((value / exponentialSum).toFixed(4))
  );
}

function calculateRawScores(data) {
  return data.map((queryItem) =>
    data.map((keyItem) =>
      Number(dotProduct(queryItem.query, keyItem.key).toFixed(2))
    )
  );
}

function scaleScores(rawScores) {
  return rawScores.map((row) =>
    row.map((score) =>
      Number((score / Math.sqrt(QKV_DIMENSION)).toFixed(2))
    )
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
  return attentionMatrix.map((attentionRow) => {
    return Array.from(
      { length: QKV_DIMENSION },
      (_, dimensionIndex) => {
        const result = attentionRow.reduce(
          (sum, attentionWeight, tokenIndex) =>
            sum +
            attentionWeight *
              data[tokenIndex].value[dimensionIndex],
          0
        );

        return Number(result.toFixed(2));
      }
    );
  });
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
}) {
  if (!matrix || matrix.length === 0) {
    return <p className="text-gray-400">Values will appear here.</p>;
  }

  const maxAbs =
    mode === "diverging"
      ? Math.max(
          1e-6,
          ...matrix.flat().map((value) => Math.abs(value))
        )
      : 1;

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

  const cellPadding = compact
    ? "px-2 py-1.5 text-xs"
    : "px-4 py-3 text-sm";

  return (
    <div>
      {helperText && (
        <p className="mb-3 text-sm text-gray-500">{helperText}</p>
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

                {colLabels.map((label, index) => (
                  <th
                    key={`col-${label}-${index}`}
                    className={`border border-gray-200 bg-slate-50 font-semibold text-gray-600 ${cellPadding}`}
                  >
                    {label}
                  </th>
                ))}
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

                  {row.map((value, columnIndex) => (
                    <td
                      key={`cell-${rowIndex}-${columnIndex}`}
                      className={`border border-gray-200 font-mono transition ${cellPadding} ${
                        selectedRowIndex === rowIndex
                          ? "ring-2 ring-inset ring-blue-300"
                          : ""
                      }`}
                      style={cellStyle(value)}
                      title={`${rowLabels[rowIndex]} · ${colLabels[columnIndex]}: ${value}`}
                    >
                      {value.toFixed(precision)}
                    </td>
                  ))}
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

  return (
    <div className="mt-8 rounded-lg bg-gray-50 p-5">
      <h3 className="mb-1 text-lg font-bold">
        Attention Distribution
      </h3>

      <p className="mb-5 text-sm text-gray-500">
        Query token:{" "}
        <span className="font-semibold text-blue-700">
          {data[selectedQueryIndex].token}
        </span>
      </p>

      <div className="space-y-4">
        {selectedRow.map((weight, index) => (
          <div key={`bar-${data[index].token}-${index}`}>
            <div className="mb-1 flex items-center justify-between gap-4">
              <span className="font-medium">
                {data[index].token}
              </span>

              <span className="font-mono text-sm">
                {(weight * 100).toFixed(2)}%
              </span>
            </div>

            <div className="h-4 overflow-hidden rounded-full bg-gray-200">
              <div
                className="h-full rounded-full bg-blue-600 transition-all duration-300"
                style={{
                  width: `${Math.max(weight * 100, 1)}%`,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
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
  Each token's embedding is compared with all other token
  embeddings using cosine similarity to find the most similar words.
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
            key={item.token}
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
// 처음 딱 한 번만 사전 담은 useState 실행해서 받아온 걸 저장
  const [embeddings, setEmbeddings] = useState(null);

  useEffect(() => {
    fetch("/glove-50d.json")
      .then((res) => res.json())
      .then((dict) => setEmbeddings(dict));
  }, []);

  function handleCalculate() {
    const trimmedSentence = sentence.trim();

    if (trimmedSentence === "") {
      setTokenData([]);
      setRawScores([]);
      setScaledScores([]);
      setAttentionMatrix([]);
      setSelectedQueryIndex(null);
      setError("Please enter a sentence.");
      return;
    }

    setError("");

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

  return (
    <main className="min-h-screen bg-gray-100 p-6 md:p-10">
      <div className="mx-auto max-w-6xl rounded-xl bg-white p-6 shadow-lg md:p-8">
        <h1 className="mb-3 text-center text-4xl font-bold">
          Transformer Visualizer
        </h1>

        <p className="mb-8 text-center text-gray-600">
          Enter a sentence to visualize the attention mechanism of
          a Transformer model.
        </p>

        <textarea
          className="w-full rounded-lg border border-gray-300 p-4 outline-none focus:border-blue-500"
          rows={3}
          placeholder="Enter a sentence, e.g. The cat sat on the chair."
          value={sentence}
          onChange={(event) => setSentence(event.target.value)}
        />

        {error && (
          <p className="mt-2 text-sm text-red-600">{error}</p>
        )}

        <button
          onClick={handleCalculate}
          className="mt-4 rounded-lg bg-blue-600 px-6 py-3 font-semibold text-white hover:bg-blue-700"
        >
          Calculate Attention
        </button>

        {tokenData.length > 0 && selectedQueryIndex !== null && (
          <p className="mt-6 text-sm text-gray-500">
            Selected token:{" "}
            <span className="font-semibold text-blue-700">
              {tokenData[selectedQueryIndex].token}
            </span>{" "}
            — click any row in the heatmaps below to trace a different
            token through the pipeline.
          </p>
        )}

        <div className="mt-6 flex flex-col">
          <FlowStage
            title="Input Embedding"
            formula={
              <>GloVe embedding · dimension {EMBEDDING_DIMENSION}</>
            }
          >
            <MatrixHeatmap
              matrix={tokenData.map((item) => item.embedding)}
              rowLabels={tokenLabels}
              colLabels={EMBEDDING_COLUMN_LABELS}
              cornerLabel="token \\ dim"
              compact
              selectedRowIndex={selectedQueryIndex}
              onSelectRow={setSelectedQueryIndex}
              helperText={
                hasOOVTokens
                  ? "Each row is a token's pretrained embedding vector. * = word not found in the 10k-word GloVe vocabulary — a random vector was used instead."
                  : "Each row is a token's pretrained embedding vector."
              }
            />
          </FlowStage>

          <FlowConnector label="× Wq / Wk / Wv" />

          <FlowStage
            title="Query · Key · Value"
            formula={
              <>
                Q = X·W<sub>Q</sub> &nbsp; K = X·W<sub>K</sub> &nbsp; V =
                X·W<sub>V</sub>
              </>
            }
          >
            <div className="flex flex-col gap-6">
              <div>
                <h3 className="mb-2 font-semibold text-green-700">
                  Query (Q)
                </h3>

                <MatrixHeatmap
                  matrix={tokenData.map((item) => item.query)}
                  rowLabels={tokenLabels}
                  colLabels={QKV_COLUMN_LABELS}
                  cornerLabel="tok \\ dim"
                  selectedRowIndex={selectedQueryIndex}
                  onSelectRow={setSelectedQueryIndex}
                />
              </div>

              <div>
                <h3 className="mb-2 font-semibold text-red-700">
                  Key (K)
                </h3>

                <MatrixHeatmap
                  matrix={tokenData.map((item) => item.key)}
                  rowLabels={tokenLabels}
                  colLabels={QKV_COLUMN_LABELS}
                  cornerLabel="tok \\ dim"
                  selectedRowIndex={selectedQueryIndex}
                  onSelectRow={setSelectedQueryIndex}
                />
              </div>

              <div>
                <h3 className="mb-2 font-semibold text-purple-700">
                  Value (V)
                </h3>

                <MatrixHeatmap
                  matrix={tokenData.map((item) => item.value)}
                  rowLabels={tokenLabels}
                  colLabels={QKV_COLUMN_LABELS}
                  cornerLabel="tok \\ dim"
                  selectedRowIndex={selectedQueryIndex}
                  onSelectRow={setSelectedQueryIndex}
                />
              </div>
            </div>
          </FlowStage>

          <FlowConnector
            label={
              <>
                Q · K<sup>T</sup>
              </>
            }
          />

          <FlowStage
            title="Raw Attention Scores"
            formula={
              <>
                score<sub>ij</sub> = q<sub>i</sub> · k<sub>j</sub>
              </>
            }
          >
            <MatrixHeatmap
              matrix={rawScores}
              rowLabels={tokenLabels}
              colLabels={tokenLabels}
              cornerLabel="Q \\ K"
              selectedRowIndex={selectedQueryIndex}
              onSelectRow={setSelectedQueryIndex}
              helperText="Dot product of every Query with every Key, before scaling."
            />
          </FlowStage>

          <FlowConnector
            label={
              <>
                ÷ √d<sub>k</sub>
              </>
            }
          />

          <FlowStage
            title="Scaled Scores"
            formula={
              <>
                scaled<sub>ij</sub> = score<sub>ij</sub> / √
                {QKV_DIMENSION}
              </>
            }
          >
            <MatrixHeatmap
              matrix={scaledScores}
              rowLabels={tokenLabels}
              colLabels={tokenLabels}
              cornerLabel="Q \\ K"
              selectedRowIndex={selectedQueryIndex}
              onSelectRow={setSelectedQueryIndex}
            />
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
              helperText="Each row is a Query token and each column is a Key token. Click a row to inspect its attention distribution."
            />

            <AttentionDistribution
              data={tokenData}
              attentionMatrix={attentionMatrix}
              selectedQueryIndex={selectedQueryIndex}
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
              colLabels={QKV_COLUMN_LABELS}
              cornerLabel="tok \\ dim"
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
                {EMBEDDING_DIMENSION})
              </>
            }
          >
            <MatrixHeatmap
              matrix={tokenData.map((item) => item.projectedOutput)}
              rowLabels={tokenLabels}
              colLabels={EMBEDDING_COLUMN_LABELS}
              cornerLabel="tok \\ dim"
              compact
              selectedRowIndex={selectedQueryIndex}
              onSelectRow={setSelectedQueryIndex}
              helperText="Attention Output projected back into embedding-space dimensions."
            />

            <OutputSimilarWords
              data={tokenData}
              selectedQueryIndex={selectedQueryIndex}
            />
          </FlowStage>
        </div>
      </div>
    </main>
  );
}
