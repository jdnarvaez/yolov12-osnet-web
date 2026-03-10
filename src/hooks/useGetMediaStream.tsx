import { useCallback, useEffect, useState } from 'react';

const stopStream = (mediaStream?: MediaStream) => {
  mediaStream?.getTracks().forEach((t) => {
    t.stop();
  });
};

export const useGetMediaStream = (): [MediaStream | undefined, (device: MediaDeviceInfo) => void, () => void] => {
  const [mediaStream, setMediaStream] = useState<MediaStream>();

  useEffect(() => {
    return () => {
      stopStream(mediaStream);
    };
  }, [mediaStream]);

  const getMediaStream = (device: MediaDeviceInfo, audio?: boolean) => {
    (async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: device.deviceId }, audio });
      setMediaStream(stream);
    })();
  };

  const stopMediaStream = useCallback(() => {
    stopStream(mediaStream);
    setMediaStream(undefined);
  }, [mediaStream]);

  return [mediaStream, getMediaStream, stopMediaStream];
};
