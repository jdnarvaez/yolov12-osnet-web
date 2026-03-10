// Cosine similarity — assumes both vectors are already L2-normalised.
export function dotProduct(a: Float32Array, b: Float32Array): number {
  let s = 0;

  for (let i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }

  return s;
}
