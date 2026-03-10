import type { Detection } from '@/types/types';

export function iou(a: Detection, b: Detection): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);

  if (x2 <= x1 || y2 <= y1) {
    return 0;
  }

  const inter = (x2 - x1) * (y2 - y1);
  return inter / (a.width * a.height + b.width * b.height - inter);
}
