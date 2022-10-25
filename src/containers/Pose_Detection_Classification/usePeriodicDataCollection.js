import { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { keypointsToVecs } from "./utils";

const TIMER_INFO_TEXT_COLOR = "#FFF"
const INTERVAL_MILLISEC = 250
const STOP_DATA_GATHER = -1;
const WAITING_TIME = 3000;
const DATA_COLLECTION_TIMEOUT = 10000;

export default function ({ 
  ctx2d 
}) {
  const [gatherDataState, setGatherDataState] = useState(STOP_DATA_GATHER);
  const [trainingDataInputs, setTrainingDataInputs] = useState([]);
  const [trainingDataOutputs, setTrainingDataOutputs] = useState([]);
  const [examplesCount, setExamplesCount] = useState([]);
  const [timeElapsed, setTimeElapsed] = useState(0);
  const [lastCollectedAt, setLastCollectedAt] = useState(-1);

  useEffect(() => {
    let timerRef;
    if (gatherDataState !== STOP_DATA_GATHER) {
      timerRef = setInterval(() => {
        setTimeElapsed(t => {
          if (t >= (WAITING_TIME + DATA_COLLECTION_TIMEOUT)) {
            setGatherDataState(STOP_DATA_GATHER)
            clearInterval(timerRef)
            return 0;
          }
          return t + INTERVAL_MILLISEC
        })
      }, INTERVAL_MILLISEC)
    }

    return () => clearInterval(timerRef)
  }, [gatherDataState])

  function dataGatherStep(keypoints) {
    if (gatherDataState === STOP_DATA_GATHER) return;

    drawTimerUpdates(timeElapsed);
    
    if (timeElapsed < WAITING_TIME ||
      lastCollectedAt === timeElapsed) return;

    const vecs = keypointsToVecs(keypoints)
    // console.log(vecs, vecs.length)

    const normalizedVecs = tf.tidy(() => {
      const vecsTensor = tf.tensor(vecs)
      const l2norm = vecsTensor.norm("euclidean", 1).expandDims(1)
      const l2norm2d = l2norm.concat(l2norm, 1)
      // const tnsr_normal = tns1.div(tns2)
      return vecsTensor.div(l2norm2d).squeeze()
    });

    // console.log(normalizedVecs.print())

    trainingDataInputs.push(normalizedVecs);
    trainingDataOutputs.push(gatherDataState);
    setTrainingDataInputs(trainingDataInputs)
    setTrainingDataOutputs(trainingDataOutputs)

    // Intialize array index element if currently undefined.
    if (examplesCount[gatherDataState] === undefined) {
      examplesCount[gatherDataState] = 0;
    }
    examplesCount[gatherDataState]++;
    setExamplesCount(examplesCount)

    setLastCollectedAt(timeElapsed);
  }

  function drawTimerUpdates(t) {
    const txt = t < WAITING_TIME ?
      `Waiting.. ${Math.round((WAITING_TIME - t)/1000)}` : 
      `Collecting data.. t=${Math.round((t - WAITING_TIME)/1000)}`;
  
    ctx2d.font = "48px Arial";
    ctx2d.fillStyle = TIMER_INFO_TEXT_COLOR;
    ctx2d.strokeStyle = "white";
    const { width, height } = ctx2d.canvas;
    const halfTxtWidth = 0.5 * ctx2d.measureText(txt).width;
    ctx2d.fillText(txt, 0.5 * width - halfTxtWidth, 0.5 * height);
  }

  /**
   * Handle Data Gather for button click.
   **/
  function startCollectingData(classNumber) {
    setTimeElapsed(0)
    setLastCollectedAt(-1);
    setGatherDataState(classNumber)
  }

  function resetDataset() {
    setExamplesCount([])
    for (let i = 0; i < trainingDataInputs.length; i++) {
      trainingDataInputs[i].dispose();
    }
    setTrainingDataInputs([])
    setTrainingDataOutputs([])
  }

  return {
    dataGatherStep,
    startCollectingData,
    examplesCount,
    trainingDataInputs,
    trainingDataOutputs,
    resetDataset,
  };
}
