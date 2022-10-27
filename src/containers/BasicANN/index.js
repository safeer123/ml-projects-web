import { useState, useEffect } from "react";
import { Button, Input } from "antd";

import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

import Page from "../../components/Page";
import Section from "../../components/Section";

import "./styles.css"

const PAGE_TITLE = "Create a simple ANN models for learning"
const MODEL_PATH = "models/model_test_01/model.json"
const DOWNLOAD_FILE_NAME = "basic-ANN-model"

const objectiveText = `The goal is to build a simple neural network using Tensorflow.js and get hands on with Deep Learning basics.
We will programatically generate input data, then train and test an ANN model, 
just to get the idea how it works. Here we consider 4 catagories of 2D points (x, y), x and y are positive integers. 
Define catagory 1 as (x, y) such that x and y are under 50.
Catagory 2, only y under 50.
Catagory 3, only x under 50.
Catagory 4, both x and y are above 50. We can programatically generate training data`

const { TextArea } = Input;

function generateRandomNums(low, high, howMany) {
  const res = []
  for (let i = 0; i < howMany; i += 1) {
    res.push(Math.round(low + (high - low) * Math.random()));
  }
  return res;
}

function generateRandomPoints(xlow, xhigh, ylow, yhigh, howMany) {
  const res = []
  const xList = generateRandomNums(xlow, xhigh, howMany)
  const yList = generateRandomNums(ylow, yhigh, howMany)
  for (let i = 0; i < howMany; i += 1) {
    res.push([xList[i], yList[i]])
  }
  return res;
}

const DEFAULT_CLASS_LIST = [{
  label: "0-x-50 0-y-50",
  data: [],
  generate: () => {
    return generateRandomPoints(0, 50, 0, 50, 100)
  }
}, {
  label: "50-x-100 0-y-50",
  data: [],
  generate: () => {
    return generateRandomPoints(50, 100, 0, 50, 100)
  }
}, {
  label: "0-x-50 50-y-100",
  data: [],
  generate: () => {
    return generateRandomPoints(0, 50, 50, 100, 100)
  }
}, {
  label: "50-x-100 50-y-100",
  data: [],
  generate: () => {
    return generateRandomPoints(50, 100, 50, 100, 100)
  }
}]

export default function () {

  const [dataGenerated, setDataGenerated] = useState(false);
  const [model, setModel] = useState(null);
  const [modelTrained, setModelTrained] = useState(false);
  const [classList, setClassList] = useState(DEFAULT_CLASS_LIST)
  const [testResults, setTestResults] = useState(null)

  async function loadModel() {
    const model = await tf.loadLayersModel(MODEL_PATH);

    const predicted_out = model.predict(
      tf.tensor2d([
        [8.63700664, 19.76268724],
        [18.63700664, 1.76268724],
        [13, 9.4]
      ])
    );
    let out_arr = predicted_out.arraySync();
    out_arr = out_arr.map((v) => (v > 0.5 ? 1 : 0));
    console.log(out_arr);
  }

  function createModel() {
    let model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [2], units: 100, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: classList.length, activation: 'softmax' }));

    model.summary();

    // Compile the model with the defined optimizer and specify a loss function to use.
    model.compile({
      // Adam changes the learning rate over time which is useful.
      optimizer: 'adam',
      // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
      // Else categoricalCrossentropy is used if more than 2 classes.
      loss: (classList.length === 2) ? 'binaryCrossentropy' : 'categoricalCrossentropy',
      // As this is a classification problem you can record accuracy in the logs too!
      metrics: ['accuracy']
    });
    setModel(model);

    tfvis.show.modelSummary({ name: "Model Summary" }, model);
  }

  async function trainModel() {
    console.log("Model training starts..")
    let traingDataX = []
    let traingDataY = []
    for (let i = 0; i < classList.length; i += 1) {
      traingDataX = traingDataX.concat(classList[i].data)
      traingDataY = traingDataY.concat(new Array(classList[i].data.length).fill(i))
    }
    console.log("input length: ", traingDataX.length, traingDataY.length);
    tf.util.shuffleCombo(traingDataX, traingDataY);
    let outputsAsTensor = tf.tensor1d(traingDataY, 'int32');
    let oneHotOutputs = tf.oneHot(outputsAsTensor, classList.length);
    let inputsAsTensor = tf.stack(traingDataX);
    inputsAsTensor = tf.reshape(inputsAsTensor, [-1, 2])
    console.log("train X shape: ", inputsAsTensor.shape)
    console.log("train Y shape: ", oneHotOutputs.shape)

    const metrics = ['acc'];
    const container = {
      name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    let results = await model.fit(inputsAsTensor, oneHotOutputs, {
      shuffle: true, batchSize: 20, epochs: 10,
      callbacks: fitCallbacks
    });

    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();
    setModelTrained(true);
    console.log("Model training complete..")
  }

  function logProgress(epoch, logs) {
    console.log('epoch ==> ' + epoch, logs);
  }

  function runTest() {
    const result = makePredictions(20)
    setTestResults(result)
    const result2 = makePredictions(1000)
    const preds = tf.tensor(result2.ypredicted)
    const labels = tf.tensor(result2.ydata)
    showAccuracy(preds, labels)
    showConfusion(preds, labels)
    preds.dispose()
    labels.dispose()
  }

  async function showAccuracy(preds, labels) {
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = { name: 'Accuracy', tab: 'Evaluation' };
    tfvis.show.perClassAccuracy(container, classAccuracy, classList.map(c => c.label));

    // labels.dispose();
}

async function showConfusion(preds, labels) {
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
    tfvis.render.confusionMatrix(container, { values: confusionMatrix, tickLabels: classList.map(c => c.label) });

    // labels.dispose();
}

  function makePredictions(testSize) {
    const randNumsX = generateRandomNums(0, 100, testSize);
    const randNumsY = generateRandomNums(0, 100, testSize);
    const X = []
    const Y = []
    for(let i = 0; i < testSize; i+=1) {
      X.push([randNumsX[i], randNumsY[i]])
      const x_under50 = randNumsX[i] < 50
      const y_under50 = randNumsY[i] < 50
      let yVal = 0
      if(x_under50 && y_under50) yVal = 0
      else if(!x_under50 && y_under50) yVal = 1
      else if(x_under50 && !y_under50) yVal = 2
      else if(!x_under50 && !y_under50) yVal = 3
      Y.push(yVal)
    }

    const Xtest = tf.tidy(() => {
      let inputsAsTensor = tf.tensor(X);
      inputsAsTensor = tf.reshape(inputsAsTensor, [-1, 2])
      return inputsAsTensor
    })

    const yPred = model.predict(Xtest).argMax(1)
    const yPredArray = Array.from(yPred.dataSync())
    return {
      xdata: X,
      ydata: Y,
      ypredicted: yPredArray
    }
  }

  useEffect(() => {
    // loadModel();
  }, []);

  function onChangeClassName(i, e) {
    // console.log(e.target.value)
    const { value } = e.target;
    const updatedClassList = [...classList];
    updatedClassList[i].label = value;
    setClassList(updatedClassList)
  }

  function generateData() {
    const classListUpdated = [...classList];
    for (let c of classListUpdated) {
      c.data = c.generate()
    }
    setClassList(classListUpdated)
    setDataGenerated(true)
  }

  async function downloadModel() {
    if (!model) return;
    await model.save(`downloads://${DOWNLOAD_FILE_NAME}`);
  }

  const disableDataGeneration = dataGenerated
  const disableCreateModel = Boolean(model) || !dataGenerated
  const disableTraining = !Boolean(model) || modelTrained
  const disableTest = !modelTrained
  const disableDownload = !modelTrained

  return (
    <Page className="ml-basic-ann-root">
      <h1>{PAGE_TITLE}</h1>

      <Section title="Objective" description={objectiveText}>
        <div className={`data-collection-window-root`}>
          {
            classList.map((c, i) => (
              <div key={`${c.label}`} className="data-class-edit-item">
                <span>{i + 1}. {"Class Label"}: </span>
                <Input placeholder="Class Name" value={c.label} style={{ width: 200 }} onChange={(e) => onChangeClassName(i, e)} />
                <span>{"Data: "}: </span>
                <TextArea rows={3} value={c.data.map(([x, y]) => `(${x}, ${y})`).join(",")} style={{ width: 400 }} />
                <span>{"data count"}: {c.data.length}</span>
                {/* {predict && i === predictedIndex && <p className="check-symbol-for-prediction">✓</p>} */}
              </div>
            ))
          }
        </div>
        <div>
          <Button type="primary" onClick={generateData} disabled={disableDataGeneration}>Generate Data</Button>
        </div>
      </Section>

      <Section disabled={!dataGenerated} title="Create and train the model" description="Create a simple ANN model to predict a given point (x, y). Data shape is [2]">
        <div>
          <Button type="primary" onClick={createModel} disabled={disableCreateModel}>Create Model</Button>
          <Button type="primary" onClick={trainModel} disabled={disableTraining} >Train Model</Button>
        </div>
      </Section>

      <Section title="Test the model" description="Now we have trained the model. It's time to test with random inputs">
        <div>
          <Button type="primary" onClick={runTest} disabled={disableTest}>Test Model</Button>
        </div>
        <div className="test-data">
          Input X: 
          {
            testResults &&
            testResults.xdata.map(([x, y]) => `(${x}, ${y})`).join(",")
          }
        </div>
        <div className="test-data">
          Expected Y: 
          {
            testResults &&
            testResults.ydata.map((y) => `${y}`).join(",")
          }
        </div>
        <div className="test-data">
          Predicted Y: 
          {
            testResults &&
            testResults.ypredicted.map((y) => `${y}`).join(",")
          }
        </div>
      </Section>

      <Section title="Download the model">
        <div>
          <Button type="primary" onClick={downloadModel} disabled={disableDownload} id="download-model">Download</Button>
        </div>
      </Section>

    </Page>);
}
