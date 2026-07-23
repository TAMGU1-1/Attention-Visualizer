"use client";

import { useState } from "react";

const EMBEDDING_DIMENSION = 8;
const QKV_DIMENSION = 8;

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

function findNearestTokens(
  data,
  vectorName,
  selectedIndex,
  limit = 3
) {
  return data
    .map((item, index) => ({
      token: item.token,
      index,
      similarity: cosineSimilarity(
        data[selectedIndex][vectorName],
        item[vectorName]
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

function calculateAttentionMatrix(data) {
  return data.map((queryItem) => {
    const scores = data.map((keyItem) => {
      const score = dotProduct(
        queryItem.query,
        keyItem.key
      );

      return score / Math.sqrt(QKV_DIMENSION);
    });

    return softmax(scores);
  });
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

function AttentionHeatmap({
  data,
  attentionMatrix,
  selectedQueryIndex,
  onSelectQuery,
}) {
  if (data.length === 0 || attentionMatrix.length === 0) {
    return (
      <p className="text-gray-400">
        The attention score matrix will appear here.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <p className="mb-4 text-sm text-gray-500">
        Each row is a Query token and each column is a Key token.
        Click a row to inspect its attention distribution.
      </p>

      <table className="min-w-max border-collapse text-center">
        <thead>
          <tr>
            <th className="border bg-gray-100 px-4 py-3 text-sm">
              Q \ K
            </th>

            {data.map((item, index) => (
              <th
                key={`column-${item.token}-${index}`}
                className="border bg-gray-100 px-4 py-3 text-sm"
              >
                {item.token}
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          {attentionMatrix.map((row, rowIndex) => (
            <tr
              key={`row-${rowIndex}`}
              onClick={() => onSelectQuery(rowIndex)}
              className="cursor-pointer"
            >
              <th
                className={`border px-4 py-3 text-sm ${
                  selectedQueryIndex === rowIndex
                    ? "bg-blue-100 text-blue-800"
                    : "bg-gray-100"
                }`}
              >
                {data[rowIndex].token}
              </th>

              {row.map((weight, columnIndex) => {
                const opacity = Math.min(
                  0.15 + weight * 1.5,
                  1
                );

                return (
                  <td
                    key={`cell-${rowIndex}-${columnIndex}`}
                    className={`border px-4 py-4 font-mono text-sm transition ${
                      selectedQueryIndex === rowIndex
                        ? "ring-2 ring-inset ring-blue-300"
                        : ""
                    }`}
                    style={{
                      backgroundColor: `rgba(37, 99, 235, ${opacity})`,
                      color: weight >= 0.4 ? "white" : "#111827",
                    }}
                    title={`${data[rowIndex].token} attends to ${data[columnIndex].token}: ${weight}`}
                  >
                    {weight.toFixed(4)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
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

function SimilarTokens({
  data,
  selectedQueryIndex,
}) {
  if (
    data.length === 0 ||
    selectedQueryIndex === null
  ) {
    return null;
  }

  const selectedToken = data[selectedQueryIndex].token;

  const querySimilarTokens = findNearestTokens(
    data,
    "query",
    selectedQueryIndex
  );

  const keySimilarTokens = findNearestTokens(
    data,
    "key",
    selectedQueryIndex
  );

  const valueSimilarTokens = findNearestTokens(
    data,
    "value",
    selectedQueryIndex
  );

  function SimilarityList({
    title,
    items,
    textClassName,
  }) {
    return (
      <div className="rounded-lg border p-5">
        <h3 className={`mb-4 text-lg font-bold ${textClassName}`}>
          {title}
        </h3>

        {items.length === 0 ? (
          <p className="text-sm text-gray-400">
            At least two words are required.
          </p>
        ) : (
          <div className="space-y-3">
            {items.map((item, index) => (
              <div
                key={`${title}-${item.token}-${item.index}`}
                className="rounded-lg bg-gray-50 p-3"
              >
                <div className="flex items-center justify-between gap-4">
                  <span className="font-medium">
                    {index + 1}. {item.token}
                  </span>

                  <span className="font-mono text-sm">
                    {item.similarity.toFixed(4)}
                  </span>
                </div>

                <div className="mt-2 h-2 overflow-hidden rounded-full bg-gray-200">
                  <div
                    className="h-full rounded-full bg-blue-600"
                    style={{
                      width: `${Math.max(
                        ((item.similarity + 1) / 2) * 100,
                        1
                      )}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="mt-8 rounded-lg border p-5">
      <div className="mb-5">
        <h2 className="text-xl font-bold">
          Similar Words by Q, K, V
        </h2>

        <p className="mt-1 text-sm text-gray-500">
          Selected token:{" "}
          <span className="font-semibold text-blue-700">
            {selectedToken}
          </span>
        </p>

        <p className="mt-2 text-sm text-gray-500">
          Similarity is calculated using cosine similarity
          between the selected token and the other tokens in
          the sentence.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        <SimilarityList
          title="Similar Query Words"
          items={querySimilarTokens}
          textClassName="text-green-700"
        />

        <SimilarityList
          title="Similar Key Words"
          items={keySimilarTokens}
          textClassName="text-red-700"
        />

        <SimilarityList
          title="Similar Value Words"
          items={valueSimilarTokens}
          textClassName="text-purple-700"
        />
      </div>

      <p className="mt-4 text-xs text-gray-400">
        Because the current embeddings and weight matrices are
        randomly generated, these results represent similarity
        inside the generated vector space rather than real
        linguistic meaning.
      </p>
    </div>
  );
}

export default function Home() {
  const [sentence, setSentence] = useState("");
  const [tokenData, setTokenData] = useState([]);
  const [attentionMatrix, setAttentionMatrix] = useState([]);
  const [selectedQueryIndex, setSelectedQueryIndex] =
    useState(null);
  const [error, setError] = useState("");

  function handleCalculate() {
    const trimmedSentence = sentence.trim();

    if (trimmedSentence === "") {
      setTokenData([]);
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

    const calculatedData = tokens.map((token) => {
      const embedding = createRandomVector(
        EMBEDDING_DIMENSION
      );

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
        embedding,
        query,
        key,
        value,
      };
    });

    const calculatedAttentionMatrix =
      calculateAttentionMatrix(calculatedData);

    const attentionOutputs = calculateAttentionOutputs(
      calculatedData,
      calculatedAttentionMatrix
    );

    const finalData = calculatedData.map((item, index) => ({
      ...item,
      output: attentionOutputs[index],
    }));

    setTokenData(finalData);
    setAttentionMatrix(calculatedAttentionMatrix);
    setSelectedQueryIndex(0);
  }

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
          placeholder="Example: The cat sat on the mat."
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

        <div className="mt-10 rounded-lg border p-5">
          <div className="mb-4">
            <h2 className="text-xl font-bold">
              Token Embeddings
            </h2>

            <p className="mt-1 text-sm text-gray-500">
              Embedding dimension: {EMBEDDING_DIMENSION}
            </p>
          </div>

          <VectorList
            data={tokenData}
            vectorName="embedding"
            emptyMessage="Embeddings will appear here."
            textClassName="text-blue-700"
          />
        </div>

        <div className="mt-8 grid grid-cols-1 gap-5 lg:grid-cols-3">
          <div className="min-h-60 rounded-lg border p-5">
            <div className="mb-4">
              <h2 className="text-xl font-bold">Query (Q)</h2>

              <p className="mt-1 text-sm text-gray-500">
                Q = Embedding × WQ
              </p>
            </div>

            <VectorList
              data={tokenData}
              vectorName="query"
              emptyMessage="Query vectors will appear here."
              textClassName="text-green-700"
            />
          </div>

          <div className="min-h-60 rounded-lg border p-5">
            <div className="mb-4">
              <h2 className="text-xl font-bold">Key (K)</h2>

              <p className="mt-1 text-sm text-gray-500">
                K = Embedding × WK
              </p>
            </div>

            <VectorList
              data={tokenData}
              vectorName="key"
              emptyMessage="Key vectors will appear here."
              textClassName="text-red-700"
            />
          </div>

          <div className="min-h-60 rounded-lg border p-5">
            <div className="mb-4">
              <h2 className="text-xl font-bold">Value (V)</h2>

              <p className="mt-1 text-sm text-gray-500">
                V = Embedding × WV
              </p>
            </div>

            <VectorList
              data={tokenData}
              vectorName="value"
              emptyMessage="Value vectors will appear here."
              textClassName="text-purple-700"
            />
          </div>
        </div>

        <div className="mt-8 min-h-96 rounded-lg border p-5">
          <div className="mb-5">
            <h2 className="text-xl font-bold">
              Attention Heatmap
            </h2>

            <p className="mt-1 text-sm text-gray-500">
              Attention(Q, K, V) = Softmax(QKᵀ / √dₖ)V
            </p>
          </div>

          <AttentionHeatmap
            data={tokenData}
            attentionMatrix={attentionMatrix}
            selectedQueryIndex={selectedQueryIndex}
            onSelectQuery={setSelectedQueryIndex}
          />

          <AttentionDistribution
            data={tokenData}
            attentionMatrix={attentionMatrix}
            selectedQueryIndex={selectedQueryIndex}
          />
        </div>

        <SimilarTokens
  data={tokenData}
  selectedQueryIndex={selectedQueryIndex}
/>

        <div className="mt-8 rounded-lg border p-5">
          <div className="mb-4">
            <h2 className="text-xl font-bold">
              Attention Output
            </h2>

            <p className="mt-1 text-sm text-gray-500">
              Output = Attention Weights × Value
            </p>
          </div>

          <VectorList
            data={tokenData}
            vectorName="output"
            emptyMessage="Attention output vectors will appear here."
            textClassName="text-orange-700"
          />
        </div>
      </div>
    </main>
  );
}