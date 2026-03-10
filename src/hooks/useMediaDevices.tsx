import { useCallback, useState } from 'react';

export const useMediaDevices = (
  constraints?: MediaStreamConstraints,
  filter: (device: MediaDeviceInfo) => boolean = () => true,
): {
  devices: MediaDeviceInfo[];
  refreshDevices: () => Promise<MediaDeviceInfo[]>;
} => {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);

  const refreshDevices = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      const deviceList = await navigator.mediaDevices.enumerateDevices();
      const filteredDevices = deviceList.filter(filter);

      setDevices(filteredDevices);

      stream.getTracks().forEach((t) => {
        try {
          t.stop();
        } catch (error) {
          console.error('Error stopping track:', error);
        }
      });

      return filteredDevices;
    } catch (error) {
      console.error('Error getting devices:', error);
      return [];
    }
  }, [filter, constraints]);

  return {
    devices,
    refreshDevices,
  };
};

export const useVideoDevices = () =>
  useMediaDevices({ audio: false, video: true }, (device) => device.kind === 'videoinput');
