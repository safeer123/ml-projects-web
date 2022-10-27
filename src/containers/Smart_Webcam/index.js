import { useState, useEffect, useRef } from "react";
import { Button } from "antd";

import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

import Page from "../../components/Page";
import Section from "../../components/Section";

import "./styles.css"

const PAGE_TITLE = "Multiple object detection using pre trained model - Coco SSD"

// Check if webcam access is supported.
function getUserMediaSupported() {
  return !!(navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia);
}

const webcamSupported = getUserMediaSupported();

let children = [];

export default function () {

  const [model, setModel] = useState(null);
  const [videoPlaying, setVideoPlaying] = useState(false);

  const videoRef = useRef();
  const liveViewRef = useRef();

  async function loadModel() {
    const model = await cocoSsd.load();
    setModel(model);
  }

  // Placeholder function for next step.
  function predictWebcam() {
    if (!model) return;

    const video = videoRef.current;
    if (!video) return;

    const liveView = liveViewRef.current;

    // Now let's start classifying a frame in the stream.
    model.detect(video).then(function (predictions) {
      // Remove any highlighting we did previous frame.
      for (let i = 0; i < children.length; i++) {
        liveView.removeChild(children[i]);
      }
      children.splice(0);

      // Now lets loop through predictions and draw them to the live view if
      // they have a high confidence score.
      for (let n = 0; n < predictions.length; n++) {
        // If we are over 66% sure we are sure we classified it right, draw it!
        if (predictions[n].score > 0.66) {
          // console.log(predictions[n]);
          const p = document.createElement('p');
          p.innerText = predictions[n].class + ' - with '
            + Math.round(parseFloat(predictions[n].score) * 100)
            + '% confidence.';
          p.style = 'margin-left: ' + predictions[n].bbox[0] + 'px; margin-top: '
            + (predictions[n].bbox[1] - 10) + 'px; width: '
            + (predictions[n].bbox[2] - 10) + 'px; top: 0; left: 0;';

          const highlighter = document.createElement('div');
          highlighter.setAttribute('class', 'highlighter');
          highlighter.style = 'left: ' + predictions[n].bbox[0] + 'px; top: '
            + predictions[n].bbox[1] + 'px; width: '
            + predictions[n].bbox[2] + 'px; height: '
            + predictions[n].bbox[3] + 'px;';

          liveView.appendChild(highlighter);
          liveView.appendChild(p);
          children.push(highlighter);
          children.push(p);
        }
      }

      // Call this function again to keep predicting when the browser is ready.
      window.requestAnimationFrame(predictWebcam);
    });
  }

  // demosSection.classList.remove('invisible');

  function enableWebcam(event) {
    if (!webcamSupported) return;

    const video = videoRef.current;
    if (video) {
      // getUsermedia parameters to force video but not audio.
      const constraints = {
        video: true
      };

      // Activate the webcam stream.
      navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        video.srcObject = stream;
        video.addEventListener('loadeddata', predictWebcam);
        setVideoPlaying(true);
      });
    }
  }


  return (
    <Page className="ml-smart-webcam-root">
      <h1>{PAGE_TITLE}</h1>

      <Section title="Objective">
        We can make our webcam smart, using some of the pre-trained models, enabling
        multiple object detection on the web.
        <ul>
          <li>
            Step 1. Click the button for loading Coco SSD Model. This might take a few seconds for setup.
          </li>
          <li>
            Step 2. Once the model is ready enable the webcam. Provide access on prompt.
          </li>
        </ul>
        <p>Hold some objects up close to your webcam to get a real-time classification!</p>
      </Section>

      <div ref={liveViewRef} id="liveView" className="camView">
        <video ref={videoRef} id="webcam" autoPlay muted width="640" height="480"></video>
        {
          !(Boolean(model) && videoPlaying) &&
          <div className="control-buttons">
            <Button type="primary" id="loadModelButton" disabled={Boolean(model)} onClick={loadModel}>Load Coco SSD Model</Button>
            <Button type="primary" id="webcamButton" disabled={!webcamSupported || !Boolean(model)} onClick={enableWebcam}>Enable Webcam</Button>
          </div>
        }
      </div>
    </Page>
  );
}
