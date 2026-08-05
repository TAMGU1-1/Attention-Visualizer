// Minimal safetensors reader.
//
// Format: [8-byte LE uint64 = header length][JSON header][raw tensor data]
// The JSON header maps tensor name -> { dtype, shape, data_offsets: [start, end] },
// where the offsets are relative to the start of the data section.

const DTYPE_READERS = {
  F32: (buffer, start, end) =>
    new Float32Array(
      buffer.buffer.slice(buffer.byteOffset + start, buffer.byteOffset + end)
    ),
  F16: (buffer, start, end) => {
    const raw = new Uint16Array(
      buffer.buffer.slice(buffer.byteOffset + start, buffer.byteOffset + end)
    );
    const out = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) {
      out[i] = decodeFloat16(raw[i]);
    }
    return out;
  },
};

export function decodeFloat16(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x03ff;

  if (exponent === 0) {
    return sign * fraction * 2 ** -24;
  }
  if (exponent === 0x1f) {
    return fraction ? NaN : sign * Infinity;
  }
  return sign * (fraction + 1024) * 2 ** (exponent - 25);
}

export function encodeFloat16(value) {
  if (Number.isNaN(value)) return 0x7e00;
  if (value === Infinity) return 0x7c00;
  if (value === -Infinity) return 0xfc00;
  if (value === 0) return Object.is(value, -0) ? 0x8000 : 0;

  const sign = value < 0 ? 0x8000 : 0;
  const magnitude = Math.abs(value);

  if (magnitude >= 65504) return sign | 0x7bff; // clamp to max finite fp16

  let exponent = Math.floor(Math.log2(magnitude));
  let fraction = magnitude / 2 ** exponent - 1;

  // log2 rounding can land one ulp off; normalise back into [1, 2).
  if (fraction < 0) {
    exponent -= 1;
    fraction = magnitude / 2 ** exponent - 1;
  } else if (fraction >= 1) {
    exponent += 1;
    fraction = magnitude / 2 ** exponent - 1;
  }

  if (exponent < -14) {
    // subnormal
    return sign | Math.round(magnitude / 2 ** -24);
  }

  let mantissa = Math.round(fraction * 1024);
  if (mantissa === 1024) {
    mantissa = 0;
    exponent += 1;
  }
  if (exponent > 15) return sign | 0x7bff;

  return sign | ((exponent + 15) << 10) | mantissa;
}

export function readSafetensors(buffer) {
  const view = new DataView(
    buffer.buffer,
    buffer.byteOffset,
    buffer.byteLength
  );

  const headerLength = Number(view.getBigUint64(0, true));
  const headerJson = new TextDecoder().decode(
    buffer.subarray(8, 8 + headerLength)
  );
  const header = JSON.parse(headerJson);
  const dataStart = 8 + headerLength;

  const tensors = new Map();

  for (const [name, meta] of Object.entries(header)) {
    if (name === "__metadata__") continue;

    const reader = DTYPE_READERS[meta.dtype];
    if (!reader) {
      throw new Error(`Unsupported dtype ${meta.dtype} for tensor ${name}`);
    }

    const [start, end] = meta.data_offsets;
    tensors.set(name, {
      shape: meta.shape,
      data: reader(buffer, dataStart + start, dataStart + end),
    });
  }

  return tensors;
}
