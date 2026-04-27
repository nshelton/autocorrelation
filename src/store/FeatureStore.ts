const EMPTY = new Float32Array(0);

export class FeatureStore {
  private buffers = new Map<string, Float32Array>();

  set(key: string, buf: Float32Array): void {
    this.buffers.set(key, buf);
  }

  get(key: string): Float32Array {
    return this.buffers.get(key) ?? EMPTY;
  }
}
