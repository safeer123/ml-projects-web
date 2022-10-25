import { useState, useEffect, useRef } from "react";
import { Button } from "antd";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

import "./styles.css"

// Check if webcam access is supported.
function getUserMediaSupported() {
  return !!(navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia);
}

const webcamSupported = getUserMediaSupported();

let children = [];

export default function () {

  const [model, setModel] = useState(null);

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

    // Only continue if the COCO-SSD has finished loading.
    if (!model) {
      return;
    }

    const video = videoRef.current;
    if (video) {
      // Hide the button once clicked.
      event.target.classList.add('removed');

      // getUsermedia parameters to force video but not audio.
      const constraints = {
        video: true
      };

      // Activate the webcam stream.
      navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        video.srcObject = stream;
        video.addEventListener('loadeddata', predictWebcam);
      });
    }
  }


  return (
    <div className="ml-smart-webcam-root">

      <h1>Multiple object detection using pre trained model in TensorFlow.js</h1>
      <p>Wait for the model to load before clicking the button to enable the webcam - at which point it will become visible to use.</p>

      <section id="demos" className={true ? "" : "invisible"}>

        <p>Hold some objects up close to your webcam to get a real-time classification! When ready click "enable webcam" below and accept access to the webcam when the browser asks (check the top left of your window)</p>

        <div ref={liveViewRef} id="liveView" className="camView">
          <video ref={videoRef} id="webcam" autoPlay muted width="640" height="480"></video>
        </div>

        <Button type="primary" id="loadModelButton" disabled={Boolean(model)} onClick={loadModel}>Load Coco SSD Model</Button>
        <Button type="primary" id="webcamButton" disabled={!webcamSupported} onClick={enableWebcam}>Enable Webcam</Button>
      </section>
    </div>
  );
}
