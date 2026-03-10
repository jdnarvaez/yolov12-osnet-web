import { CameraButton } from '@/components/CameraButton';
import { Button, Card, Spinner } from '@heroui/react';
import { Bot, BotOff, Camera } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

type MediaStreamTrackProcessorCtor = new (options: {
  track: MediaStreamTrack;
}) => {
  readable: ReadableStream<VideoFrame>;
};

function App() {
  const cameraCanvasRef = useRef<HTMLCanvasElement>(null);
  const [inferenceEnabled, setInferenceEnabled] = useState(true);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [receivedFrame, setReceivedFrame] = useState(false);

  const worker = useMemo(() => new Worker(new URL('./worker/worker', import.meta.url), { type: 'module' }), []);

  useEffect(() => {
    return () => {
      worker.terminate();
    };
  }, [worker]);

  useEffect(() => {
    const onMessage = (event: MessageEvent<{ type: string; bitmap: ImageBitmap }>) => {
      const { type, bitmap } = event.data;

      if (!bitmap || type !== 'frame') {
        return;
      }

      try {
        const canvas = cameraCanvasRef.current;

        if (!canvas) {
          return;
        }

        setReceivedFrame(true);

        const ctx = canvas.getContext('2d');

        if (!ctx) {
          return;
        }

        if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
        }

        ctx.drawImage(bitmap, 0, 0);
      } finally {
        bitmap.close();
      }
    };

    worker.addEventListener('message', onMessage);

    return () => {
      worker.removeEventListener('message', onMessage);
    };
  }, [worker]);

  useEffect(() => {
    if (!stream || !worker) {
      return;
    }

    const track = stream.getVideoTracks()[0];

    if (track.readyState !== 'live') {
      setStream(null);
      return;
    }

    const MediaStreamTrackProcessorConstructor = (
      globalThis as typeof globalThis & {
        MediaStreamTrackProcessor?: MediaStreamTrackProcessorCtor;
      }
    ).MediaStreamTrackProcessor;

    if (!MediaStreamTrackProcessorConstructor) {
      console.error('MediaStreamTrackProcessor API is not available in this browser');
      return;
    }

    // Transfer VideoFrames to the worker; it renders each frame (with detections)
    // onto an OffscreenCanvas and posts back an ImageBitmap for display
    const processor = new MediaStreamTrackProcessorConstructor({ track });
    worker.postMessage({ readable: processor.readable }, [processor.readable]);

    return () => {
      if (cameraCanvasRef.current) {
        const ctx = cameraCanvasRef.current.getContext('2d');
        ctx?.clearRect(0, 0, cameraCanvasRef.current.width, cameraCanvasRef.current.height);
      }

      try {
        stream.getTracks().forEach((track) => {
          track.stop();
        });
      } catch (error) {
        console.error('Error stopping camera stream:', error);
      }
    };
  }, [worker, stream]);

  const handleCameraStart = useCallback((stream: MediaStream) => {
    setStream(stream);
  }, []);

  const handleCameraStop = useCallback(() => {
    setStream(null);
    setReceivedFrame(false);
  }, []);

  const toggleInference = useCallback(() => {
    setInferenceEnabled((prev) => {
      const newInferenceEnabled = !prev;
      worker.postMessage({ inferenceEnabled: newInferenceEnabled });
      return newInferenceEnabled;
    });
  }, [worker]);

  return (
    <div className="min-h-screen max-h-screen overflow-hidden flex items-center justify-center">
      <div className="h-full w-full max-h-screen p-4 overflow-hidden flex">
        <Card className="h-full w-full max-h-screen">
          <Card.Content className="p-6">
            <div className="w-full">
              <div className="space-y-6">
                {/* Camera preview — worker renders each frame (with detections) to this canvas */}
                <div className="relative bg-card rounded-xl overflow-hidden min-h-[400px] flex items-center justify-center">
                  {!stream ? (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                      <Camera className="h-16 w-16 mx-auto mb-4 opacity-50" />
                      <p className="text-lg">Video Feed Not Started</p>
                      <p className="text-sm mt-2">Click &quot;Start Camera&quot; below to begin</p>
                    </div>
                  ) : null}
                  <canvas ref={cameraCanvasRef} className={`w-full h-auto object-contain${!stream ? ' hidden' : ''}`} />
                  <AnimatePresence>
                    {stream && !receivedFrame ? (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="absolute flex top-0 right-0 bottom-0 left-0 items-center justify-center"
                      >
                        <Spinner size="xl" color="current" />
                      </motion.div>
                    ) : null}
                  </AnimatePresence>
                </div>
                <div className="flex justify-center gap-2">
                  <CameraButton isCameraActive={!!stream} onStart={handleCameraStart} onStop={handleCameraStop} />
                  <Button onClick={toggleInference} variant="outline" className="px-4 py-2 gap-4">
                    {inferenceEnabled ? <BotOff className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                    {`Turn Detections ${!inferenceEnabled ? 'On' : 'Off'}`}
                  </Button>
                </div>
              </div>
            </div>
          </Card.Content>
        </Card>
      </div>
    </div>
  );
}

export default App;
