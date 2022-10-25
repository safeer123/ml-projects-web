import { useState, useEffect } from "react";
import { Button } from "antd";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";


import MnistData from "./minstData";
import NumsModel from "./NumsModel";

// instantiate the wrapper
const numsModel = new NumsModel();

export default function () {

  const [dataReady, setDataReady] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [modelTrained, setModelTrained] = useState(false);

  async function loadData() {
    await numsModel.loadData();
    setDataReady(true);
  }

  function createModel() {
    numsModel.createModel();
    setModelReady(true);
    numsModel.visualize_model();
  }

  async function trainModel() {
    await numsModel.trainModel();
    setModelTrained(true);
  }

  async function testModel() {
    await numsModel.runTest();
  }

  function resetModel() {
    numsModel.resetModel();
    setModelReady(false);
    setModelTrained(false);
  }

  return (
    <div className="ml-training-app-root">
      <h3>CNN - Hand written numbers classification</h3>
      <div className="ml-training-app-section-root">
        <p>Load the minst data set</p>
        <Button type="primary" onClick={loadData} disabled={dataReady}>Load Data</Button>
      </div>

      <div className="ml-training-app-section-root">
        <p>Create a model with few convolutional layers and dense layers</p>
        <Button type="primary" onClick={createModel} disabled={!dataReady || modelReady}>Create Model</Button>
      </div>

      <div className="ml-training-app-section-root">
        <p>Train the model with the data</p>
        <Button type="primary" onClick={trainModel} disabled={!modelReady || modelTrained}>Train the model</Button>
      </div>

      <div className="ml-training-app-section-root">
        <p>Test the model</p>
        <Button type="primary" onClick={testModel} disabled={!modelTrained}>Test the model</Button>
      </div>

      <div className="ml-training-app-section-root">
        <p>Reset modeling and training</p>
        <Button type="primary" onClick={resetModel}>Reset</Button>
      </div>
    </div>
  );
}
