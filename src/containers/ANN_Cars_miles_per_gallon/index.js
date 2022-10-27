import { useState } from "react";
import { Button } from "antd";
import Page from "../../components/Page";
import Section from "../../components/Section";

import CarsModel from "./CarsModel";

const PAGE_TITLE = "Predict Miles Per Gallon for new cars"
const DOWNLOAD_FILE_NAME = "cars-MPG-prediction-model"

const objectiveText = <div>
  Objective: Given "Horsepower" for a car, the model will learn to predict "Miles per Gallon" (MPG).
  This is part of the tutorial on the tensorflow.js official page. The data is obtained using the endpoint
  <a href="https://storage.googleapis.com/tfjs-tutorials/carsData.json">cars data</a>
</div>

// instantiate the wrapper
const carsModel = new CarsModel();

export default function () {

  const [dataLoaded, setDataLoaded] = useState(false);
  const [modelCreated, setModelCreated] = useState(false);
  const [modelTrained, setModelTrained] = useState(false);

  async function loadData() {
    await carsModel.getData();
    carsModel.visualize_data();
    setDataLoaded(true)
  }

  function createModel() {
    carsModel.createModel();
    carsModel.visualize_model();
    setModelCreated(true)
  }

  async function trainModel() {
    const {X_train, Y_train} = carsModel.convertToTensor();
    await carsModel.trainModel(X_train, Y_train);
    setModelTrained(true)
  }

  function testModel() {
    carsModel.testModel();
  }

  function downloadModel() {
    carsModel.download(DOWNLOAD_FILE_NAME)
  }

  const disableDataLoad = dataLoaded
  const disableCreateModel = modelCreated || !dataLoaded
  const disableTraining = !modelCreated || modelTrained
  const disableTest = !modelTrained
  const disableDownload = !modelTrained


  return (
    <Page className="ml-cars-mpg-root">
      <h1>{PAGE_TITLE}</h1>

      <Section title="Objective" description={objectiveText}>
        Refer: <a href="https://codelabs.developers.google.com/codelabs/tfjs-training-regression?hl=en#0">Link to the codelab</a>
      </Section>

      <Section title="Load the data for cars" description="We can download the data set">
        <div>
          <Button type="primary" onClick={loadData} disabled={disableDataLoad}>Load Data</Button>
        </div>
      </Section>

      <Section title="Create a model with 3 Dense layers">
        <Button type="primary" onClick={createModel} disabled={disableCreateModel}>Create Model</Button>
      </Section>

      <Section title="Train the model with normalized data">
        <Button type="primary" onClick={trainModel} disabled={disableTraining}>Train the model</Button>
      </Section>

      <Section title="Test the model">
        <Button type="primary" onClick={testModel} disabled={disableTest}>Test the model</Button>
      </Section>

      <Section title="Download the model">
        <div>
          <Button type="primary" onClick={downloadModel} disabled={disableDownload} id="download-model">Download</Button>
        </div>
      </Section>
    </Page>
  );
}
