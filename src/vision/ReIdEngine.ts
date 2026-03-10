import type { Detection } from '@/types/types';
import * as ort from 'onnxruntime-web/webgpu';
import { createSession } from './utils/createSession';
import { dotProduct } from './utils/dotProduct';

const REID_H = 256; // input height
const REID_W = 128; // input width
const REID_DIM = 512; // OSNet x0.25 embedding dimension
const REID_MEAN = [0.485, 0.456, 0.406] as const; // ImageNet mean (RGB)
const REID_STD = [0.229, 0.224, 0.225] as const; // ImageNet std  (RGB)
const REID_SIM_THRESHOLD = 0.65; // min cosine similarity to reuse a track
const REID_MAX_AGE = 60 * 60 * 10; // inference cycles before a track expires
// Max normalised center-to-center distance (0–√2) for a gallery candidate to
// be considered. Prevents an ID from jumping across the frame.
const REID_MAX_DIST = 0.4;

// cx/cy are normalised center coordinates (0–1) of the detection when the
// entry was last updated, used for the spatial gate.
export interface GalleryEntry {
  embedding: Float32Array;
  lastCycle: number;
  cx: number;
  cy: number;
}

export class ReIdEngine {
  private session: ort.InferenceSession | null = null;
  private classGalleries = new Map<string, Map<number, GalleryEntry>>();
  private nextTrackId = 0;
  private reidCycle = 0;
  private executionProviders: ort.InferenceSession.ExecutionProviderConfig[];
  private uri;
  private lastComputationTimeMs = 0;
  private runningInference: boolean = false;
  private cropCanvas = new OffscreenCanvas(REID_W, REID_H);
  private cropCtx = this.cropCanvas.getContext('2d', {
    willReadFrequently: true,
    alpha: false,
  }) as OffscreenCanvasRenderingContext2D;

  constructor({
    uri,
    executionProviders = ['webgpu'],
  }: {
    uri: string;
    executionProviders?: ort.InferenceSession.ExecutionProviderConfig[];
  }) {
    this.uri = uri;
    this.executionProviders = executionProviders;
  }

  getDimensions(): { width: number; height: number } {
    return { width: REID_W, height: REID_H };
  }

  async intialize(): Promise<void> {
    if (this.session) {
      return;
    }

    try {
      this.session = await createSession(this.uri, this.executionProviders);
      console.log('[worker] Re-ID session loaded');
    } catch (err) {
      console.warn('[worker] Re-ID model failed to load (tracking disabled):', err);
    }
  }

  getClassGallery(className: string): Map<number, GalleryEntry> {
    let cg = this.classGalleries.get(className);

    if (!cg) {
      cg = new Map();
      this.classGalleries.set(className, cg);
    }

    return cg;
  }

  getLastComputationTimeMs(): number {
    return this.lastComputationTimeMs;
  }

  // Crops each detection from srcCanvas, runs OSNet in a single batch, matches
  // embeddings to the per-class gallery, and writes trackId back onto each Detection.
  async assignTrackIds(detections: Detection[], srcCanvas: OffscreenCanvas): Promise<Detection[]> {
    const start = performance.now();

    if (this.runningInference) {
      return detections;
    }

    this.runningInference = true;

    try {
      if (!this.session || detections.length === 0) {
        return detections;
      }

      const cycle = this.reidCycle++;
      const N = detections.length;
      const planeSize = REID_H * REID_W;
      const reidData = new Float32Array(N * 3 * planeSize);

      for (let n = 0; n < N; n++) {
        const { x, y, width, height } = detections[n];
        this.cropCtx.clearRect(0, 0, REID_W, REID_H);
        this.cropCtx.drawImage(srcCanvas, x, y, width, height, 0, 0, REID_W, REID_H);
        const pixels = this.cropCtx.getImageData(0, 0, REID_W, REID_H).data;
        const base = n * 3 * planeSize;

        for (let i = 0; i < planeSize; i++) {
          const p = i * 4;
          reidData[base + i] = (pixels[p] / 255 - REID_MEAN[0]) / REID_STD[0];
          reidData[base + planeSize + i] = (pixels[p + 1] / 255 - REID_MEAN[1]) / REID_STD[1];
          reidData[base + 2 * planeSize + i] = (pixels[p + 2] / 255 - REID_MEAN[2]) / REID_STD[2];
        }
      }

      const inputTensor = new ort.Tensor('float32', reidData, [N, 3, REID_H, REID_W]);
      let results: Record<string, ort.Tensor> | null = null;

      try {
        results = (await this.session.run({
          [this.session.inputNames[0]]: inputTensor,
        })) as Record<string, ort.Tensor>;

        const raw = results[this.session.outputNames[0]].data as Float32Array;

        // L2-normalise each embedding and copy into its own buffer for gallery storage
        const embeddings: Float32Array[] = [];

        for (let n = 0; n < N; n++) {
          const emb = raw.slice(n * REID_DIM, (n + 1) * REID_DIM);
          let norm = 0;

          for (let i = 0; i < REID_DIM; i++) {
            norm += emb[i] * emb[i];
          }

          norm = Math.sqrt(norm);

          if (norm > 1e-6) {
            for (let i = 0; i < REID_DIM; i++) {
              emb[i] /= norm;
            }
          }

          embeddings.push(emb);
        }

        // Expire stale tracks across all class galleries
        for (const cg of this.classGalleries.values()) {
          for (const [id, entry] of cg) {
            if (cycle - entry.lastCycle > REID_MAX_AGE) {
              cg.delete(id);
            }
          }
        }

        // Greedy nearest-neighbour matching within the same class + spatial gate
        const claimed = new Set<number>();

        for (let n = 0; n < N; n++) {
          const det = detections[n];
          const cg = this.getClassGallery(det.class);

          // Normalised center of this detection (0–1 in each axis)
          const ncx = (det.x + det.width / 2) / srcCanvas.width;
          const ncy = (det.y + det.height / 2) / srcCanvas.height;

          let bestId = -1,
            bestSim = REID_SIM_THRESHOLD;

          for (const [id, entry] of cg) {
            if (claimed.has(id)) {
              continue;
            }

            // Spatial gate — skip candidates whose last known position is too far away
            const dx = ncx - entry.cx;
            const dy = ncy - entry.cy;

            if (Math.sqrt(dx * dx + dy * dy) > REID_MAX_DIST) {
              continue;
            }

            const sim = dotProduct(embeddings[n], entry.embedding);

            if (sim > bestSim) {
              bestSim = sim;
              bestId = id;
            }
          }

          if (bestId === -1) {
            bestId = this.nextTrackId++;
          }

          claimed.add(bestId);
          cg.set(bestId, { embedding: embeddings[n], lastCycle: cycle, cx: ncx, cy: ncy });
          detections[n].trackId = bestId;
        }
      } finally {
        inputTensor.dispose();

        if (results) {
          for (const t of Object.values(results)) {
            t.dispose();
          }
        }
      }
    } finally {
      this.lastComputationTimeMs = performance.now() - start;
      this.runningInference = false;
    }

    return detections;
  }
}
