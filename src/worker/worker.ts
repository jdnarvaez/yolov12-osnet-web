import { ReIdEngine } from '@/vision/ReIdEngine';
import { YoloEngine } from '@/vision/YoloEngine';
import type { Detection } from '../types/types';

let inferenceEnabled = true;
let inferenceInProgress = false;
let detections: Detection[] = [];

const COLOR_MAP: Record<string, string> = {
  person: '#3B82F6',
  car: '#EF4444',
  truck: '#F59E0B',
  bus: '#10B981',
  motorcycle: '#8B5CF6',
  bicycle: '#EC4899',
  'traffic light': '#F97316',
  'stop sign': '#DC2626',
  'fire hydrant': '#EF4444',
  dog: '#14B8A6',
  cat: '#6366F1',
  bird: '#22C55E',
  chair: '#A855F7',
  bottle: '#06B6D4',
  cup: '#84CC16',
  laptop: '#64748B',
  phone: '#6366F1',
  book: '#F59E0B',
  umbrella: '#EC4899',
};

const getClassColor = (className: string): string => {
  if (COLOR_MAP[className]) {
    return COLOR_MAP[className];
  }

  let hash = 0;

  for (let i = 0; i < className.length; i++) {
    hash = className.charCodeAt(i) + ((hash << 5) - hash);
  }

  COLOR_MAP[className] = `hsl(${Math.abs(hash) % 360}, 70%, 55%)`;

  return COLOR_MAP[className];
};

function drawDetections(ctx: OffscreenCanvasRenderingContext2D, detections: Detection[]): void {
  for (const { x, y, width, height, confidence, class: className, trackId } of detections) {
    const color = getClassColor(className);
    const label = (trackId !== undefined ? `#${trackId} ${className}` : className).toUpperCase();
    const confidenceText = `(${(confidence * 100).toFixed(1)}%)`;

    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);

    ctx.font = 'bold 2.5rem Arial';
    const measuredText = ctx.measureText(label);
    const textHeight = measuredText.actualBoundingBoxAscent + measuredText.actualBoundingBoxDescent;
    const labelW0 = measuredText.width;
    ctx.font = '1.5rem monospace';
    const confidenceW = ctx.measureText(confidenceText).width;

    const labelW = labelW0 + confidenceW + 16;
    const labelH = 2 * textHeight;
    const labelX = x;
    const labelY = Math.max(0, y - labelH - 0.25 * textHeight);

    ctx.fillStyle = color;
    ctx.fillRect(labelX, labelY, labelW, labelH);

    ctx.fillRect(x, y, 16, 16);
    ctx.fillRect(x + width - 16, y, 16, 16);
    ctx.fillRect(x, y + height - 16, 16, 16);
    ctx.fillRect(x + width - 16, y + height - 16, 16, 16);

    ctx.textBaseline = 'middle';
    ctx.fillStyle = 'white';
    ctx.font = 'bold 2.5rem Arial';
    ctx.fillText(label, labelX + 8, labelY + labelH / 2);
    ctx.font = '1.5rem monospace';
    ctx.fillText(confidenceText, labelX + labelW0 + 8, labelY + labelH / 2);
  }
}

const reIdEngine: ReIdEngine | null = new ReIdEngine({
  uri: new URL(`../models/osnet_x0_25.onnx`, import.meta.url).href,
});

const yoloEngine: YoloEngine | null = new YoloEngine({
  uri: new URL(`../models/yolov12n.onnx`, import.meta.url).href,
});

self.onmessage = async (event: MessageEvent<{ readable?: ReadableStream<VideoFrame>; inferenceEnabled?: boolean }>) => {
  if (event.data.inferenceEnabled !== undefined) {
    inferenceEnabled = event.data.inferenceEnabled;
    console.log('[worker] inferenceEnabled set to', inferenceEnabled);
  }

  if (!event.data.readable) {
    return;
  }

  const { readable } = event.data;

  try {
    await yoloEngine.intialize();
    await reIdEngine.intialize();
    console.log('Models loaded');
  } catch (err) {
    console.error('[worker] failed to load models:', err);
    self.postMessage({ type: 'error', message: 'Failed to load models' });
    return;
  }

  const displayCanvas = new OffscreenCanvas(1, 1);
  const displayCtx = displayCanvas.getContext('2d', { alpha: false }) as OffscreenCanvasRenderingContext2D;

  // Display loop — renders every camera frame at the native camera rate.
  // Inference runs asynchronously; the most recent detections are overlaid on
  // each display frame so video is always smooth regardless of inference speed.
  const reader = readable.getReader();

  try {
    while (true) {
      const { done, value: videoFrame } = await reader.read();

      if (done) {
        console.log('[worker] video feed ended');
        break;
      }

      const w = videoFrame.displayWidth;
      const h = videoFrame.displayHeight;

      if (displayCanvas.width !== w || displayCanvas.height !== h) {
        displayCanvas.width = w;
        displayCanvas.height = h;
      }

      const t0 = performance.now();
      // mirror the video frame horizontally
      displayCtx.save();
      displayCtx.translate(w, 0);
      displayCtx.scale(-1, 1);
      displayCtx.drawImage(videoFrame, 0, 0);
      displayCtx.restore();
      const t1 = performance.now();

      if (inferenceEnabled && !inferenceInProgress) {
        inferenceInProgress = true;

        yoloEngine
          .runInference(displayCanvas)
          .then((newDetections: Detection[] | undefined) => {
            if (!newDetections) {
              inferenceInProgress = false;
              return;
            }

            reIdEngine
              .assignTrackIds(newDetections, yoloEngine.getPreprocessedCanvas())
              .then((reIdDetections: Detection[]) => {
                detections = reIdDetections;
                drawDetections(displayCtx, detections);
              })
              .finally(() => {
                inferenceInProgress = false;
              });
          })
          .catch((err) => {
            console.error('[worker] inference error:', err);
            inferenceInProgress = false;
          });

        drawDetections(displayCtx, detections);
      } else if (inferenceEnabled) {
        drawDetections(displayCtx, detections);
      }

      videoFrame.close();

      displayCtx.fillStyle = 'white';
      displayCtx.font = 'bold 2rem "Courier New"';
      const lastTimingStr = `yolo=${(yoloEngine!.getLastComputationTimeMs()).toFixed(1)}ms reid=${(reIdEngine!.getLastComputationTimeMs()).toFixed(1)}ms`;
      displayCtx.fillText(`draw=${(t1 - t0).toFixed(1)}ms  ${inferenceEnabled ? lastTimingStr : ''}`, 16, 16);
      const bitmap = displayCanvas.transferToImageBitmap();
      self.postMessage({ type: 'frame', bitmap, detections }, { transfer: [bitmap] });
    }
  } catch (err) {
    console.error('[worker] frame pipeline error:', err);
  } finally {
    reader.releaseLock();
  }
};
