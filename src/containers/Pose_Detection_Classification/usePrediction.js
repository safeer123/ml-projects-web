import { useState } from "react";

import * as tf from "@tensorflow/tfjs";

import { keypointsToVecs } from "./utils";

const PREDICTION_INFO_TEXT_COLOR = "#9F9"

export default function ({
  predict,
  model,
  classNames,
  ctx2d,
}) {

  const [predictedIndex, setPredictedIndex] = useState(-1);

  function predictStep(keypoints) {
    if (!model) return;
    let predIndex = -1;
    if (predict) {

      const vecs = keypointsToVecs(keypoints)
      // console.log(vecs, vecs.length)

      tf.tidy(() => {
        const vecsTensor = tf.tensor(vecs)
        const l2norm = vecsTensor.norm("euclidean", 1).expandDims(1)
        const l2norm2d = l2norm.concat(l2norm, 1)
        const normalizedVecs = vecsTensor.div(l2norm2d).reshape([1, 32])

        // console.log("prediction input shape: ", normalizedVecs.shape)

        let prediction = model.predict(normalizedVecs).squeeze();
        let highestIndex = prediction.argMax().arraySync();
        let predictionArray = prediction.arraySync();
        predIndex = highestIndex;
        drawPredictionResult(classNames[highestIndex], Math.floor(predictionArray[highestIndex] * 100))
      });
    }
    if (predictedIndex !== predIndex) {
      setPredictedIndex(predIndex);
    }
  }

  function drawPredictionResult(className, confidence) {
    const txt = `Predicted: ${className} (${confidence}%)`;
    ctx2d.font = "24px Arial";
    ctx2d.fillStyle = PREDICTION_INFO_TEXT_COLOR;
    const { width, height } = ctx2d.canvas;
    // const halfTxtWidth = 0.5 * ctx2d.measureText(txt).width;
    ctx2d.fillText(txt, 50, 0.1 * height);
  }

  return {
    predictStep,
    predictedIndex,
  };
}
