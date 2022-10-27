import { useState, useEffect } from "react";
import { Button } from "antd";

import NumsModel from "./NumsModel";
import Page from "../../components/Page";
import Section from "../../components/Section";

const PAGE_TITLE = "CNN - Hand written numbers classification"
const DOWNLOAD_FILE_NAME = "hand-written-numbers-CNN-model"

const objectiveText = <>
  Objective: Predict hand written numbers from 0 to 9 using CNN. This is often considered as the "hello world" program in DL.
  This is part of the tutorial on the tensorflow.js official page. Digits are catagorized using the index as label from 0 to 9.
  For training we will use 28x28px greyscale images from a dataset called 
  <a href="https://yann.lecun.com/exdb/mnist/">MINST</a>
</>

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

  function downloadModel() {
    numsModel.download(DOWNLOAD_FILE_NAME)
  }

  const disableDownload = false

  return (
    <Page className="ml-handwritten-numbers-root">
      <h1>{PAGE_TITLE}</h1>

      <Section title="Objective" description={objectiveText}>
        Refer: <a href="https://codelabs.developers.google.com/codelabs/tfjs-training-classfication#0">Link to the codelab</a>
      </Section>

      <Section title="Load the minst data set">
        <Button type="primary" onClick={loadData} disabled={dataReady}>Load Data</Button>
      </Section>

      <Section title="Create a model with few convolutional layers and dense layers">
        <Button type="primary" onClick={createModel} disabled={!dataReady || modelReady}>Create Model</Button>
      </Section>

      <Section title="Train the model with the data">
        <Button type="primary" onClick={trainModel} disabled={!modelReady || modelTrained}>Train the model</Button>
      </Section>

      <Section title="Test the model">
        <Button type="primary" onClick={testModel} disabled={!modelTrained}>Test the model</Button>
      </Section>

      <Section title="Reset modeling and training">
        <Button type="primary" onClick={resetModel}>Reset</Button>
      </Section>

      <Section title="Download the model">
        <div>
          <Button type="primary" onClick={downloadModel} disabled={disableDownload} id="download-model">Download</Button>
        </div>
      </Section>
    </Page>
  );
}
