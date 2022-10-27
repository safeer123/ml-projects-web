import { useState, useEffect, useRef } from "react";
import { Button, Input } from "antd";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import useBodyPoseDetection from "./useBodyPoseDetection";
import useBodyPoseGraphicLayer from "./useBodyPoseGraphicLayer";
import useWebCam from "./useWebCam";
import usePeriodicDataCollection from "./usePeriodicDataCollection";
import usePrediction from "./usePrediction";

import ComponentLayout1 from "../../components/ComponentLayout1";
import Section from "../../components/Section";
import { objective_content } from "./projectWriteups";
import Page from "../../components/Page";

import "./styles.css"

const PAGE_TITLE = "Pose Classification"
const CLASS_NAME_INPUT_LABEL = "Pose Label"
const DATA_GATHER_BUTTON_LABEL = "Start Collecting Data"
const TRAIN_DATA_COUNT_LABEL = "Collected: "
const DOWNLOAD_FILE_NAME = "pose-classification-model"

const DEFAULT_POSE_LABEL_01 = "Crescent Moon"
const DEFAULT_POSE_LABEL_02 = "Triangle"
const DEFAULT_POSE_LABEL_03 = "High Cobra"
const DEFAULT_POSE_LABEL_04 = "Half Pigeon"

let handle = null;

export default function () {

  const [model, setModel] = useState(null);
  const [modelTrained, setModelTrained] = useState(false);

  const [predict, setPredict] = useState(false);
  const [ctx2d, setCtx2d] = useState(null);
  const [classList, setClassList] = useState([
    { id: 0, name: DEFAULT_POSE_LABEL_01 },
    { id: 1, name: DEFAULT_POSE_LABEL_02 },
    { id: 2, name: DEFAULT_POSE_LABEL_03 },
    { id: 3, name: DEFAULT_POSE_LABEL_04 },
  ]);

  const videoRef = useRef();
  const canvasRef = useRef();
  const classNames = classList.map(c => c.name);

  const { detector, handposeDetector } = useBodyPoseDetection();
  const { drawAnnotations } = useBodyPoseGraphicLayer({ ctx2d });
  const { enableCam, videoPlaying } = useWebCam({ videoRef, onVideoLoaded });

  const {
    dataGatherStep,
    startCollectingData,
    examplesCount,
    trainingDataInputs,
    trainingDataOutputs,
    resetDataset,
  } = usePeriodicDataCollection({ ctx2d });

  const {
    predictStep,
    predictedIndex,
  } = usePrediction({
    predict,
    model,
    classNames,
    ctx2d,
  })

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      setCtx2d(ctx);
    }
  }, [canvasRef])

  useEffect(() => {
    handle = window.requestAnimationFrame(mainLoop)
    return () => window.cancelAnimationFrame(handle)
  }, [detector, ctx2d, drawAnnotations, dataGatherStep])

  function mainLoop() {
    if (!videoPlaying) return;
    detectBodyPoses().then(poses => {
      if (poses && poses[0]) {
        const { keypoints } = poses[0];

        if (!ctx2d) return;
        const { width, height } = ctx2d.canvas;
        ctx2d.clearRect(0, 0, width, height);

        // Testing ctx2d
        // ctx2d.fillStyle = "rgba(0, 255, 0, 1)";
        // ctx2d.fillRect(50, 50, 10, 10);

        drawAnnotations({ keypoints })

        // data collection
        dataGatherStep(keypoints)

        // prediction
        predictStep(keypoints)
      }
    })

    handle = window.requestAnimationFrame(mainLoop);
  }

  async function detectBodyPoses() {
    if (!detector) return;

    const video = videoRef.current;
    if (!video) return;

    return await detector.estimatePoses(video);
  }

  function onVideoLoaded(video) {
    if (canvasRef?.current) {
      canvasRef.current.width = video?.videoWidth;
      canvasRef.current.height = video?.videoHeight;
    }
  }

  // create the model
  function createModel() {
    let model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [32], units: 8, activation: 'relu' }));
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

    tfvis.show.modelSummary({ name: "Model Summary", tab: "Model" }, model);
  }

  async function trainAndPredict() {
    setPredict(false)
    console.log("train 1: ", trainingDataInputs.length, trainingDataInputs[0].shape)
    console.log("train 2: ", trainingDataOutputs.length)
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
    let oneHotOutputs = tf.oneHot(outputsAsTensor, classNames.length);
    let inputsAsTensor = tf.stack(trainingDataInputs);
    inputsAsTensor = tf.reshape(inputsAsTensor, [-1, 32])
    console.log("train X shape: ", inputsAsTensor.shape)
    console.log("train Y shape: ", oneHotOutputs.shape)

    let results = await model.fit(inputsAsTensor, oneHotOutputs, {
      shuffle: true, batchSize: 5, epochs: 10,
      callbacks: tfvis.show.fitCallbacks(
        { name: "Training Performance", tab: "Training" },
        ["loss", "acc"],
        { height: 200, callbacks: ["onEpochEnd"] }
      )
    });

    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();
    setModelTrained(true);
    setPredict(true)
    tfvis.visor().close()
  }

  function logProgress(epoch, logs) {
    console.log('Data for epoch ' + epoch, logs);
  }

  /**
   * Purge data and start over. Note this does not dispose of the loaded 
   * MobileNet model and MLP head tensors as you will need to reuse 
   * them to train a new model.
   **/
  function reset() {
    setPredict(false)
    resetDataset()
    setModel(null)
    console.log('Tensors in memory: ' + tf.memory().numTensors);
  }

  function onChangeClassName(i, e) {
    // console.log(e.target.value)
    const { value } = e.target;
    const updatedClassList = [...classList];
    updatedClassList[i].name = value;
    setClassList(updatedClassList)
  }

  function launchModelViz() {
    tfvis.visor().open()
  }

  async function downloadModel() {
    await model.save(`downloads://${DOWNLOAD_FILE_NAME}`);
  }


  let dataCollectionControlClass = (modelTrained || !videoPlaying) ? 'disabled' : ''
  return (
    <Page className="ml-pose-classification-root">

      <h1>{PAGE_TITLE}</h1>

      <Section title="Objective" description={objective_content} />

      <ComponentLayout1
        A={
          <div className="webcam-video-window-root">
            <video ref={videoRef} id="webcam" autoPlay muted></video>
            <canvas ref={canvasRef} id="overlay-canvas" />
            {
              !videoPlaying &&
              <div className="webcam-video-control-overlay-root">
                <Button type="primary" disabled={videoPlaying} id="enableCam" onClick={enableCam}>Enable Webcam</Button>
              </div>
            }
          </div>
        }
        B={
          <div className={`data-collection-window-root ${dataCollectionControlClass}`}>
            {
              classList.map((c, i) => (
                <div key={`${c.id}`} className="data-class-edit-item">
                  <p>{i + 1}. {CLASS_NAME_INPUT_LABEL}: </p>
                  <Input placeholder="Class Name" value={c.name} style={{ width: 200 }} onChange={(e) => onChangeClassName(i, e)} />
                  <Button type="primary" onClick={() => startCollectingData(i)} >{DATA_GATHER_BUTTON_LABEL}</Button>
                  {i in examplesCount && <p>{TRAIN_DATA_COUNT_LABEL}: {examplesCount[i]}</p>}
                  {predict && i === predictedIndex && <p className="check-symbol-for-prediction">✓</p>}
                </div>
              ))
            }
            <div className="data-collection-footer">Total collected: {trainingDataInputs.length}</div>
          </div>
        }
      />


      <Section title="Create the model and train">
        <div>
          <Button type="primary" id="createModel" onClick={createModel} disabled={Boolean(model)}>Create the model</Button>
          <Button type="primary" onClick={trainAndPredict} id="train" disabled={!Boolean(model) || modelTrained}>Train &amp; Predict!</Button>
          <Button type="primary" onClick={reset} id="reset">Reset</Button>
        </div>
        <div>
          <Button type="primary" id="createModel" onClick={launchModelViz} disabled={!Boolean(model)}>Launch model visualization</Button>
        </div>
      </Section>

      <Section title="Download the model">
        <div>
          <Button type="primary" onClick={downloadModel} disabled={!Boolean(model) || !modelTrained} id="download-model">Download</Button>
        </div>
      </Section>

    </Page>
  );
}
