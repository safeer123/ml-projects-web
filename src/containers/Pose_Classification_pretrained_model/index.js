import { useState, useEffect, useRef } from "react";
import { Button, Input } from "antd";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

import useBodyPoseDetection from "./useBodyPoseDetection";
import useBodyPoseGraphicLayer from "./useBodyPoseGraphicLayer";
import useWebCam from "./useWebCam";
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

const DEFAULT_POSE_LABEL_01 = "Mountain"
const DEFAULT_POSE_LABEL_02 = "Downward-facing dog"
const DEFAULT_POSE_LABEL_03 = "Plank"
const DEFAULT_POSE_LABEL_04 = "Cobra"

const MODEL_PATH = "models/pose_classification_01/pose-classification-model.json"

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

    videoRef.current.addEventListener('loadeddata', function () {
      // console.log("Video dim: ", video?.videoWidth, video?.videoHeight)
      if (canvasRef?.current) {
        canvasRef.current.width = videoRef.current?.videoWidth;
        canvasRef.current.height = videoRef.current?.videoHeight;
        console.log(videoRef.current?.videoWidth, videoRef.current?.videoHeight)
        const newH = videoRef.current?.clientWidth * videoRef.current?.videoHeight / videoRef.current?.videoWidth;
        videoRef.current.style.height = `${newH}px`
        canvasRef.current.style.height = `${newH}px`
        console.log(videoRef.current?.clientWidth, videoRef.current?.clientHeight)
      }
    });
  }, [canvasRef])

  useEffect(() => {
    handle = window.requestAnimationFrame(mainLoop)
    return () => window.cancelAnimationFrame(handle)
  }, [detector, ctx2d, drawAnnotations])

  function mainLoop() {
    if (!videoRef?.current) return;
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

  async function loadModel() {
    const model = await tf.loadLayersModel(MODEL_PATH);
    setModel(model);
    tfvis.show.modelSummary({ name: "Model Summary", tab: "Model" }, model);
    setModelTrained(true);
    setPredict(true)
  }

  function onChangeClassName(i, e) {
    // console.log(e.target.value)
    const { value } = e.target;
    const updatedClassList = [...classList];
    updatedClassList[i].name = value;
    setClassList(updatedClassList)
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
            <video ref={videoRef} width="640" height="480" controls autoPlay>
              <source src="YogaPoses.mp4" type="video/mp4" />
            </video>
            <canvas ref={canvasRef} id="overlay-canvas" />
          </div>
        }
        B={
          <div className={`data-collection-window-root ${dataCollectionControlClass}`}>
            {
              classList.map((c, i) => (
                <div key={`${c.id}`} className="data-class-edit-item">
                  <p>{i + 1}. {CLASS_NAME_INPUT_LABEL}: </p>
                  <Input placeholder="Class Name" value={c.name} style={{ width: 200 }} onChange={(e) => onChangeClassName(i, e)} />
                  {predict && i === predictedIndex && <p className="check-symbol-for-prediction">✓</p>}
                </div>
              ))
            }
          </div>
        }
      />


      <Section title="Create the model and train">
        <div>
          <Button type="primary" id="loadModel" onClick={loadModel} disabled={Boolean(model)}>Load the model</Button>
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
