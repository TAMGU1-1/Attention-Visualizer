// Shared vector/matrix helpers.
//
// Nothing here rounds. Rounding during the forward pass compounds across
// layers; formatting is the display layer's job (MatrixHeatmap `precision`).

export function dotProduct(vectorA, vectorB) {
  let sum = 0;
  for (let i = 0; i < vectorA.length; i += 1) {
    sum += vectorA[i] * vectorB[i];
  }
  return sum;
}

export function vectorNorm(vector) {
  return Math.sqrt(dotProduct(vector, vector));
}

export function cosineSimilarity(vectorA, vectorB) {
  const denominator = vectorNorm(vectorA) * vectorNorm(vectorB);
  return denominator === 0 ? 0 : dotProduct(vectorA, vectorB) / denominator;
}

export function softmax(values) {
  const maximumValue = Math.max(...values);

  const exponentials = Array.from(values, (value) =>
    Math.exp(value - maximumValue)
  );

  const exponentialSum = exponentials.reduce((sum, value) => sum + value, 0);

  return exponentials.map((value) => value / exponentialSum);
}

/** matrix is an array of rows: result[col] = Σ_row vector[row] * matrix[row][col] */
export function multiplyVectorByMatrix(vector, matrix) {
  return matrix[0].map((_, columnIndex) =>
    vector.reduce(
      (sum, value, rowIndex) => sum + value * matrix[rowIndex][columnIndex],
      0
    )
  );
}

/**
 * Same operation for a flat row-major [inDim, outDim] matrix, optionally
 * restricted to a slice of output columns (used to pull one attention head
 * out of a fused Q/K/V projection).
 */
export function multiplyVectorByFlatMatrix(
  vector,
  flatMatrix,
  inDim,
  outDim,
  { bias = null, columnStart = 0, columnCount = outDim } = {}
) {
  const result = new Float32Array(columnCount);

  for (let i = 0; i < inDim; i += 1) {
    const value = vector[i];
    if (value === 0) continue;

    const rowOffset = i * outDim + columnStart;
    for (let c = 0; c < columnCount; c += 1) {
      result[c] += value * flatMatrix[rowOffset + c];
    }
  }

  if (bias) {
    for (let c = 0; c < columnCount; c += 1) {
      result[c] += bias[columnStart + c];
    }
  }

  return result;
}

export function addInPlace(target, addend) {
  for (let i = 0; i < target.length; i += 1) {
    target[i] += addend[i];
  }
  return target;
}

/** BERT-style LayerNorm over the last dimension. */
export function layerNorm(vector, weight, bias, epsilon = 1e-12) {
  const length = vector.length;

  let mean = 0;
  for (let i = 0; i < length; i += 1) mean += vector[i];
  mean /= length;

  let variance = 0;
  for (let i = 0; i < length; i += 1) {
    const delta = vector[i] - mean;
    variance += delta * delta;
  }
  variance /= length;

  const scale = 1 / Math.sqrt(variance + epsilon);
  const result = new Float32Array(length);

  for (let i = 0; i < length; i += 1) {
    result[i] = (vector[i] - mean) * scale * weight[i] + bias[i];
  }

  return result;
}

/** Exact GELU (erf form), matching BERT's "gelu" activation. */
export function gelu(value) {
  return 0.5 * value * (1 + erf(value / Math.SQRT2));
}

// Abramowitz & Stegun 7.1.26 — max abs error ~1.5e-7, well below fp32 display needs.
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const absoluteX = Math.abs(x);

  const t = 1 / (1 + 0.3275911 * absoluteX);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-absoluteX * absoluteX);

  return sign * y;
}
