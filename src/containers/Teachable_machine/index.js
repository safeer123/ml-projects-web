import { useState, useEffect, useRef } from "react";
import { Button, Input } from "antd";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import "./styles.css"

// Check if webcam access is supported.
function getUserMediaSupported() {
  return !!(navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia);
}

const webcamSupported = getUserMediaSupported();


const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const MOB_NET_URL =
  'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
let handle = null;
let handle2 = null;

export default function () {

  const [mobilenet, setMobilenet] = useState(undefined);
  const [gatherDataState, setGatherDataState] = useState(STOP_DATA_GATHER);
  const [model, setModel] = useState(null);
  const [videoPlaying, setVideoPlaying] = useState(false);
  const [trainingDataInputs, setTrainingDataInputs] = useState([]);
  const [trainingDataOutputs, setTrainingDataOutputs] = useState([]);
  const [examplesCount, setExamplesCount] = useState([]);
  const [predict, setPredict] = useState(false);
  const [stateText, setStateText] = useState('Awaiting TF.js load');
  const [classList, setClassList] = useState([
    {id: 0, name: "pen"},
    {id: 1, name: "bottle"},
    {id: 2, name: "spectacles"},
    {id: 3, name: "nothing"},
  ]);
  const [predictedIndex, setPredictedIndex] = useState(-1);

  const classNames = classList.map(c => c.name);

  const videoRef = useRef();

  useEffect(() => {
    loadMobileNetFeatureModel();
  }, [])

  useEffect(() => {
    if (gatherDataState !== STOP_DATA_GATHER) {
      handle = window.requestAnimationFrame(dataGatherLoop)
    }
    return () => window.cancelAnimationFrame(handle)
  }, [gatherDataState])

  useEffect(() => {
    if (predict) {
      handle2 = window.requestAnimationFrame(predictLoop)
    }
    return () => window.cancelAnimationFrame(handle2)
  }, [predict])

  /**
   * Loads the MobileNet model and warms it up so ready for use.
   **/
  async function loadMobileNetFeatureModel() {

    const mobileNet = await tf.loadGraphModel(MOB_NET_URL, { fromTFHub: true });
    setMobilenet(mobileNet);
    setStateText('MobileNet v3 loaded successfully!');

    // Warm up the model by passing zeros through it once.
    tf.tidy(function () {
      let answer = mobileNet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
      console.log(answer.shape);
    });
  }

  // create the model
  function createModel() {
    let model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [1024], units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: classNames.length, activation: 'softmax' }));

    model.summary();

    // Compile the model with the defined optimizer and specify a loss function to use.
    model.compile({
      // Adam changes the learning rate over time which is useful.
      optimizer: 'adam',
      // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
      // Else categoricalCrossentropy is used if more than 2 classes.
      loss: (classNames.length === 2) ? 'binaryCrossentropy' : 'categoricalCrossentropy',
      // As this is a classification problem you can record accuracy in the logs too!
      metrics: ['accuracy']
    });
    setModel(model);
  }

  // demosSection.classList.remove('invisible');

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
          setVideoPlaying(true);
        });
      });
    }
  }

  /**
   * Handle Data Gather for button mouseup/mousedown.
   **/
  function collectData(classNumber, isMouseDown) {
    const gathrDataState = isMouseDown ? classNumber : STOP_DATA_GATHER;
    setGatherDataState(gathrDataState)
    // console.log("mouse pressed: ", classNumber, isMouseDown, gathrDataState)
  }

  function dataGatherLoop() {
    const video = videoRef.current;
    if (!video) return;

    // console.log(gatherDataState)
    if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
      let imageFeatures = tf.tidy(function () {
        let videoFrameAsTensor = tf.browser.fromPixels(video);
        let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT,
          MOBILE_NET_INPUT_WIDTH], true);
        let normalizedTensorFrame = resizedTensorFrame.div(255);
        return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
      });

      trainingDataInputs.push(imageFeatures);
      trainingDataOutputs.push(gatherDataState);
      setTrainingDataInputs(trainingDataInputs)
      setTrainingDataOutputs(trainingDataOutputs)

      // Intialize array index element if currently undefined.
      if (examplesCount[gatherDataState] === undefined) {
        examplesCount[gatherDataState] = 0;
      }
      examplesCount[gatherDataState]++;
      setExamplesCount(examplesCount)

      let statusTxt = ''
      for (let n = 0; n < classNames.length; n++) {
        statusTxt += classNames[n] + ' data count: ' + examplesCount[n] + '. ';
      }
      setStateText(statusTxt);
    }

    handle = window.requestAnimationFrame(dataGatherLoop);
  }

  async function trainAndPredict() {
    setPredict(false)
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
    let oneHotOutputs = tf.oneHot(outputsAsTensor, classNames.length);
    let inputsAsTensor = tf.stack(trainingDataInputs);

    let results = await model.fit(inputsAsTensor, oneHotOutputs, {
      shuffle: true, batchSize: 5, epochs: 10,
      callbacks: { onEpochEnd: logProgress }
    });

    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();
    setPredict(true)
  }

  function logProgress(epoch, logs) {
    console.log('Data for epoch ' + epoch, logs);
  }

  function predictLoop() {
    const video = videoRef.current;
    if (!video || !model || !mobilenet) return;
    let predIndex = -1;
    if (predict) {
      tf.tidy(function () {
        let videoFrameAsTensor = tf.browser.fromPixels(video).div(255);
        let resizedTensorFrame = tf.image.resizeBilinear(
          videoFrameAsTensor,
          [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
          true);

        let imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
        let prediction = model.predict(imageFeatures).squeeze();
        let highestIndex = prediction.argMax().arraySync();
        let predictionArray = prediction.arraySync();
        predIndex = highestIndex;
        const statusTxt = 'Prediction: ' + classNames[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
        setStateText(statusTxt)
      });

      handle2 = window.requestAnimationFrame(predictLoop);
    }
    if (predictedIndex !== predIndex) {
      setPredictedIndex(predIndex);
    }
  }

  /**
   * Purge data and start over. Note this does not dispose of the loaded 
   * MobileNet model and MLP head tensors as you will need to reuse 
   * them to train a new model.
   **/
  function reset() {
    setPredict(false)
    setExamplesCount([])
    for (let i = 0; i < trainingDataInputs.length; i++) {
      trainingDataInputs[i].dispose();
    }
    setTrainingDataInputs([])
    setTrainingDataOutputs([])
    setStateText('No data collected')

    console.log('Tensors in memory: ' + tf.memory().numTensors);
  }

  function onChangeClassName(i, e) {
    // console.log(e.target.value)
    const { value } = e.target;
    const updatedClassList = [...classList];
    updatedClassList[i].name = value;
    setClassList(updatedClassList)
  }


  return (
    <div className="ml-teachable-machine-root">

      <h1>Teachable Machine</h1>

      <p id="status">{stateText}</p>

      <video ref={videoRef} id="webcam" autoPlay muted></video>

      <div>
        {
          classList.map((c, i) => (
            <div key={`${c.id}`} className="data-class-edit-item">
              <p>{i + 1}. Class Name: </p>
              <Input placeholder="Class Name" value={c.name} style={{ width: 200 }} onChange={(e) => onChangeClassName(i, e)} />
              <Button type="primary" onMouseUp={() => collectData(i, false)} onMouseDown={() => collectData(i, true)}>Gather Data</Button>
              { i in examplesCount && <p>Train sample count: {examplesCount[i]}</p> }
              { predict && i === predictedIndex && <p className="check-symbol-for-prediction">✓</p> }
            </div>
          ))
        }

      </div>
      <Button type="primary" id="createModel" onClick={createModel}>Create a model</Button>
      <Button type="primary" disabled={videoPlaying} id="enableCam" onClick={enableCam}>Enable Webcam</Button>
      <Button type="primary" onClick={trainAndPredict} id="train">Train &amp; Predict!</Button>
      <Button type="primary" onClick={reset} id="reset">Reset</Button>

    </div>
  );
}
