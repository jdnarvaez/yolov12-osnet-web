import { useVideoDevices } from '@/hooks/useMediaDevices';
import { Button, ButtonGroup, Dropdown, Label } from '@heroui/react';
import { useDisclosure } from '@reactuses/core';
import { Camera, CameraOff, ChevronDown } from 'lucide-react';
import { useCallback, useEffect, useRef } from 'react';

interface CameraButtonProps {
  isCameraActive: boolean;
  onStart: (stream: MediaStream) => void;
  onStop: () => void;
}

async function openStream(deviceId?: string): Promise<MediaStream> {
  const videoConstraints: MediaTrackConstraints = deviceId
    ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }
    : { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };

  try {
    return await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
  } catch {
    return await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId } } : true,
      audio: false,
    });
  }
}

export function CameraButton({ isCameraActive, onStart, onStop }: CameraButtonProps) {
  const { devices: videoDevices, refreshDevices: refreshVideoDevices } = useVideoDevices();
  const previewVideoRef = useRef<HTMLVideoElement>(null);
  const previewStreamRef = useRef<MediaStream | null>(null);
  const { isOpen, onOpenChange } = useDisclosure();

  const stopPreview = useCallback(() => {
    if (previewStreamRef.current) {
      previewStreamRef.current.getTracks().forEach((t) => {
        t.stop();
      });

      previewStreamRef.current = null;
    }

    if (previewVideoRef.current) {
      previewVideoRef.current.srcObject = null;
    }
  }, []);

  const startPreview = useCallback(
    async (deviceId: string) => {
      stopPreview();

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: deviceId } },
          audio: false,
        });

        previewStreamRef.current = stream;

        if (previewVideoRef.current) {
          previewVideoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.warn('[CameraButton] preview failed:', err);
      }
    },
    [stopPreview],
  );

  const handleSelectDevice = useCallback(
    async (deviceId: string) => {
      try {
        const stream = await openStream(deviceId);
        onStart(stream);
      } catch (err) {
        console.error('[CameraButton] failed to start camera with device:', err);
      }
    },
    [onStart],
  );

  useEffect(() => {
    if (videoDevices?.[0]?.deviceId) {
      startPreview(videoDevices[0].deviceId);
    }
  }, [videoDevices, startPreview]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: causes infinite render loop otherwise
  useEffect(() => {
    if (isOpen) {
      refreshVideoDevices().then((devices) => {
        if (devices?.length) {
          startPreview(devices[0].deviceId);
        }
      });
    }
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) {
      stopPreview();
    }
  }, [isOpen, stopPreview]);

  useEffect(() => () => stopPreview(), [stopPreview]);

  return (
    <ButtonGroup>
      <Button
        variant="outline"
        onClick={async () => {
          if (isCameraActive) {
            onStop();
            return;
          }

          const devices = await refreshVideoDevices();

          if (devices?.length) {
            handleSelectDevice(devices[0].deviceId);
          }
        }}
      >
        {isCameraActive ? <CameraOff className="h-4 w-4 mr-2" /> : <Camera className="h-4 w-4 mr-2" />}
        {isCameraActive ? 'Stop Camera' : 'Start Camera'}
      </Button>
      <Dropdown isOpen={isOpen} onOpenChange={onOpenChange}>
        <Button isIconOnly aria-label="More options" variant="outline">
          <ChevronDown />
        </Button>
        <Dropdown.Popover className="max-w-[290px]" placement="bottom end">
          <div className="flex p-4">
            <video
              ref={previewVideoRef}
              autoPlay
              muted
              playsInline
              className="w-full aspect-video rounded-md bg-black object-cover"
              style={{
                transform: 'scaleX(-1)',
              }}
            />
          </div>
          <Dropdown.Menu className="flex flex-col items-center gap-1">
            {videoDevices.map((device) => (
              <Dropdown.Item
                key={device.deviceId}
                className="flex flex-col items-center gap-1"
                id={device.deviceId}
                textValue={device.label}
                onMouseEnter={() => startPreview(device.deviceId)}
                onClick={() => handleSelectDevice(device.deviceId)}
              >
                <Label className="flex items-center">{device.label}</Label>
              </Dropdown.Item>
            ))}
          </Dropdown.Menu>
        </Dropdown.Popover>
      </Dropdown>
    </ButtonGroup>
  );
}
