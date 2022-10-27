import { useState } from "react";

// Check if webcam access is supported.
function getUserMediaSupported() {
  return !!(navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia);
}

const webcamSupported = getUserMediaSupported();

export default function ({
  videoRef,
  onVideoLoaded,
}) {

  const [videoPlaying, setVideoPlaying] = useState(true);

  function enableCam(event) {
    if (!webcamSupported) return;

    const video = videoRef.current;
    if (video) {
      const constraints = {
        video: true,
        width: 640,
        height: 480
      };

      // Activate the webcam stream.
      navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        video.srcObject = stream;
        video.addEventListener('loadeddata', function () {
          console.log("Video dim: ", video?.videoWidth, video?.videoHeight)
          setVideoPlaying(true);
          onVideoLoaded(video);
        });
      });
    }
  }

  return {
    enableCam,
    videoPlaying,
  };
}
