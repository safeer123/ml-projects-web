import React from "react";

import * as poseDetection from "@tensorflow-models/pose-detection";
import * as handposeDetection from '@tensorflow-models/handpose';
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs-core";

export default function useBodyPoseDetection() {

  const [detector, setDetector] = React.useState(null);
  const [handposeDetector, setHandposeDetector] = React.useState(null);

  React.useEffect(() => {
    setupTf();
  }, []);

  const setupTf = async () => {
    const flagConfig = {
      WEBGL_VERSION: 2,
      WEBGL_CPU_FORWARD: true,
      WEBGL_PACK: true,
      WEBGL_FORCE_F16_TEXTURES: false,
      WEBGL_RENDER_FLOAT32_CAPABLE: true,
      WEBGL_FLUSH_THRESHOLD: -1
    };
    tf.env().setFlags(flagConfig);
    await tf.setBackend("webgl");

    const det = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
        maxPoses: 1,
        type: "lightning",
        scoreThreshold: 0.3,
        customModel: "",
        enableTracking: false
      }
    );
    setDetector(det);

    const handposeDet = await handposeDetection.load();
    setHandposeDetector(handposeDet);
    // console.log(handposeDet)
  };

  return {
    detector,
    handposeDetector,
  };
}
