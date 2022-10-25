import { useState } from "react";
import { Button } from "antd";

import CarsModel from "./CarsModel";

// instantiate the wrapper
const carsModel = new CarsModel();

export default function () {

  async function loadData() {
    await carsModel.getData();
    carsModel.visualize_data();
  }

  function createModel() {
    carsModel.createModel();
    carsModel.visualize_model();
  }

  async function trainModel() {
    const {X_train, Y_train} = carsModel.convertToTensor();
    await carsModel.trainModel(X_train, Y_train);
  }

  function testModel() {
    carsModel.testModel();
  }

  return (
    <div className="ml-training-app-root">
      <h3>Cars Miles Per Gallon</h3>
      <div className="ml-training-app-section-root">
        <p>Load the data for cars</p>
        <Button type="primary" onClick={loadData}>Load Data</Button>
      </div>

      <div className="ml-training-app-section-root">
        <p>Create a model with few Dense layers</p>
        <Button type="primary" onClick={createModel}>Create Model</Button>
      </div>

      <div className="ml-training-app-section-root">
        <p>Train the model with normalized data</p>
        <Button type="primary" onClick={trainModel}>Train the model</Button>
      </div>

      <div className="ml-training-app-section-root">
        <p>Test the model</p>
        <Button type="primary" onClick={testModel}>Test the model</Button>
      </div>
    </div>
  );
}
