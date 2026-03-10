import { Detection } from '@/types/types';
import * as ort from 'onnxruntime-web/webgpu';
import metadata from '../models/yolov12n.json';
import { createSession } from './utils/createSession';
import { iou } from './utils/iou';

export class YoloEngine {
  private executionProviders: ort.InferenceSession.ExecutionProviderConfig[];
  private uri;
  private session: ort.InferenceSession | null = null;
  private tensorData: Float32Array;
  private inputName = '';
  private outputName = '';
  private lastComputationTimeMs = 0;
  private preprocessCanvas: OffscreenCanvas;
  private preprocessCtx: OffscreenCanvasRenderingContext2D;
  private inputWidth: number;
  private inputHeight: number;
  private runningInference: boolean = false;
  private confidenceThreshold: number;
  private nmsThreshold: number;
  private classes: string[];

  private inferenceFrameCanvas = new OffscreenCanvas(1, 1);

  private inferenceFrameCtx = this.inferenceFrameCanvas.getContext('2d', {
    willReadFrequently: true,
    alpha: false,
  }) as OffscreenCanvasRenderingContext2D;

  constructor({
    uri,
    executionProviders = ['webgpu', 'wasm', 'cpu'],
    inputWidth = metadata.inputSize[0],
    inputHeight = metadata.inputSize[1],
    nmsThreshold = metadata.nmsThreshold,
    confidenceThreshold = metadata.confidenceThreshold,
    classes = metadata.classes,
  }: {
    uri: string;
    executionProviders?: ort.InferenceSession.ExecutionProviderConfig[];
    inputWidth?: number;
    inputHeight?: number;
    nmsThreshold?: number;
    confidenceThreshold?: number;
    classes?: string[];
  }) {
    this.uri = uri;
    this.executionProviders = executionProviders;
    this.preprocessCanvas = new OffscreenCanvas(inputWidth, inputHeight);
    this.nmsThreshold = nmsThreshold;
    this.confidenceThreshold = confidenceThreshold;
    this.classes = classes;

    this.preprocessCtx = this.preprocessCanvas.getContext('2d', {
      willReadFrequently: true,
      alpha: false,
    }) as OffscreenCanvasRenderingContext2D;

    this.tensorData = new Float32Array(inputWidth * inputHeight * 3);
    this.inputWidth = inputWidth;
    this.inputHeight = inputHeight;
  }

  getPreprocessedCanvas(): OffscreenCanvas {
    return this.preprocessCanvas;
  }

  getDimensions(): { width: number; height: number } {
    return { width: this.inputWidth, height: this.inputHeight };
  }

  getLastComputationTimeMs(): number {
    return this.lastComputationTimeMs;
  }

  async intialize(): Promise<void> {
    if (this.session) {
      return;
    }

    try {
      this.session = await createSession(this.uri, this.executionProviders);
      this.inputName = this.session.inputNames[0];
      this.outputName = this.session.outputNames[0];
      console.log('[worker] YOLO session loaded');
    } catch (err) {
      console.warn('[worker] YOLO model failed to load (object detection disabled):', err);
    }
  }

  preprocessFrame(): { offsetX: number; offsetY: number; scale: number } {
    const srcW = this.inferenceFrameCanvas.width;
    const srcH = this.inferenceFrameCanvas.height;
    const scale = Math.min(this.inputWidth / srcW, this.inputHeight / srcH);
    const dw = srcW * scale;
    const dh = srcH * scale;
    const offsetX = (this.inputWidth - dw) / 2;
    const offsetY = (this.inputHeight - dh) / 2;

    this.preprocessCtx.clearRect(0, 0, this.inputWidth, this.inputHeight);
    this.preprocessCtx.drawImage(this.inferenceFrameCanvas, 0, 0, srcW, srcH, offsetX, offsetY, dw, dh);

    const pixels = this.preprocessCtx.getImageData(0, 0, this.inputWidth, this.inputHeight).data;
    const INV255 = 1 / 255;
    const planeSize = this.inputWidth * this.inputHeight;

    for (let i = 0; i < planeSize; i++) {
      const s = i * 4;
      this.tensorData[i] = pixels[s] * INV255;
      this.tensorData[i + planeSize] = pixels[s + 1] * INV255;
      this.tensorData[i + planeSize * 2] = pixels[s + 2] * INV255;
    }

    return { offsetX, offsetY, scale };
  }

  postprocess(
    data: Float32Array | null,
    dims: readonly number[],
    frameW: number,
    frameH: number,
    offsetX: number,
    offsetY: number,
    scale: number,
  ): Detection[] {
    if (!data) {
      return [];
    }

    const nmsUsed = new Set<number>();
    const numDetections = dims[1];
    const raw: Detection[] = [];

    for (let i = 0; i < numDetections; i++) {
      const s = i * 6;
      const confidence = data[s + 4];

      if (confidence < this.confidenceThreshold) {
        continue;
      }

      const classId = Math.round(data[s + 5]);
      const tx1 = (data[s] - offsetX) / scale;
      const ty1 = (data[s + 1] - offsetY) / scale;
      const tx2 = (data[s + 2] - offsetX) / scale;
      const ty2 = (data[s + 3] - offsetY) / scale;

      const x = Math.max(0, tx1);
      const y = Math.max(0, ty1);
      const width = Math.min(Math.max(0, tx2 - tx1), frameW - x);
      const height = Math.min(Math.max(0, ty2 - ty1), frameH - y);

      raw.push({ x, y, width, height, confidence, class: this.classes[classId] ?? `class_${classId}` });
    }

    raw.sort((a, b) => b.confidence - a.confidence);

    const kept: Detection[] = [];

    for (let i = 0; i < raw.length; i++) {
      if (nmsUsed.has(i)) {
        continue;
      }

      kept.push(raw[i]);

      for (let j = i + 1; j < raw.length; j++) {
        if (!nmsUsed.has(j) && iou(raw[i], raw[j]) > this.nmsThreshold) {
          nmsUsed.add(j);
        }
      }

      nmsUsed.add(i);
    }

    return kept;
  }

  async runInference(canvas: OffscreenCanvas): Promise<Detection[] | undefined> {
    if (!this.session || this.runningInference) {
      return undefined;
    }

    this.runningInference = true;

    const start = performance.now();

    if (this.inferenceFrameCanvas.width !== canvas.width || this.inferenceFrameCanvas.height !== canvas.height) {
      this.inferenceFrameCanvas.width = canvas.width;
      this.inferenceFrameCanvas.height = canvas.height;
    }

    this.inferenceFrameCtx.drawImage(canvas, 0, 0);

    const { offsetX, offsetY, scale } = this.preprocessFrame();
    const { width: inputWidth, height: inputHeight } = this.getDimensions();
    const inputTensor = new ort.Tensor('float32', this.tensorData, [1, 3, inputHeight, inputWidth]);
    let results: Record<string, ort.Tensor> | null = null;

    try {
      results = (await this.session.run({ [this.inputName]: inputTensor })) as Record<string, ort.Tensor>;
      const outputTensor = results[this.outputName];
      let outputData: Float32Array | null = null;

      if (outputTensor) {
        outputData = (await outputTensor.getData()) as Float32Array;
      }

      return this.postprocess(
        outputData,
        outputTensor?.dims ?? [],
        canvas.width,
        canvas.height,
        offsetX,
        offsetY,
        scale,
      );
    } finally {
      inputTensor.dispose();

      if (results) {
        for (const t of Object.values(results)) {
          t.dispose();
        }
      }

      this.lastComputationTimeMs = performance.now() - start;
      this.runningInference = false;
    }
  }
}
