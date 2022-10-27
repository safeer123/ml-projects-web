import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

export default class CarsModel {
    constructor() {
        this.data = null;
        this.readyData = null;
        this.model = null;
    }

    /**
    * Get the car data reduced to just the variables we are interested
    * and cleaned of missing data.
    */
    async getData() {
        const carsDataResponse = await fetch(
            "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
        );
        const carsData = await carsDataResponse.json();
        const cleaned = carsData
            .map((car) => ({
                mpg: car.Miles_per_Gallon,
                horsepower: car.Horsepower
            }))
            .filter((car) => car.mpg != null && car.horsepower != null);
        this.data = cleaned;
    }

    createModel() {
        // Create a sequential model
        this.model = tf.sequential();

        // Add a single input layer
        this.model.add(
            tf.layers.dense({
                inputShape: [1],
                units: 100,
                useBias: true,
                kernelInitializer: "zeros",
                biasInitializer: "ones",
                activation: 'relu'
            })
        );

        // pass custom weights
        // const kernals = tf.ones([100, 30]);
        // const biases = tf.ones([30]);

        this.model.add(tf.layers.dense({ 
            units: 30, 
            activation: "relu", 
            // weights: [kernals, biases] 
        }));

        // this.model.add(tf.layers.dense({ units: 5, activation: "tanh" }));

        // Add an output layer
        this.model.add(
            tf.layers.dense({ units: 1, useBias: true, activation: 'sigmoid'})
        );
    }

    /**
   * Convert the input data to tensors that we can use for machine
   * learning. We will also do the important best practices of _shuffling_
   * the data and _normalizing_ the data
   * MPG on the y-axis.
   */
    convertToTensor() {
        const { data } = this;
        // Wrapping these calculations in a tidy will dispose any
        // intermediate tensors.

        return tf.tidy(() => {
            // Step 1. Shuffle the data
            tf.util.shuffle(data);

            // Step 2. Convert data to Tensor
            const inputs = data.map((d) => d.horsepower);
            const labels = data.map((d) => d.mpg);

            const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
            const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

            //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
            const inputMax = inputTensor.max();
            const inputMin = inputTensor.min();
            const labelMax = labelTensor.max();
            const labelMin = labelTensor.min();

            const normalizedInputs = inputTensor
                .sub(inputMin)
                .div(inputMax.sub(inputMin));
            const normalizedLabels = labelTensor
                .sub(labelMin)
                .div(labelMax.sub(labelMin));

            // console.log("inputMax: ", Array.from(inputMax.dataSync()))

            this.readyData = {
                // Return the min/max bounds so we can use them later.
                inputMax: Array.from(inputMax.dataSync()),
                inputMin: Array.from(inputMin.dataSync()),
                labelMax: Array.from(labelMax.dataSync()),
                labelMin: Array.from(labelMin.dataSync())
            };
            return {
                X_train: normalizedInputs,
                Y_train: normalizedLabels,
            }
        });
    }

    async trainModel(X_train, Y_train) {
        const { model } = this;
        console.log(model)
        // console.log("X: ", X_train)
        // console.log("Y: ", Y_train)
        // Prepare the model for training.
        model.compile({
            optimizer: tf.train.adam(),
            loss: tf.losses.meanSquaredError,
            metrics: ["mse"]
        });

        const batchSize = 34;
        const epochs = 50;

        console.log("training starts...")

        await model.fit(X_train, Y_train, {
            // batchSize,
            epochs,
            shuffle: true,
            callbacks: tfvis.show.fitCallbacks(
                { name: "Training Performance", tab: "Training" },
                ["loss", "mse"],
                { height: 200, callbacks: ["onEpochEnd"] }
            )
        });
    }

    testModel() {
        const { model, data } = this;
        let { inputMax, inputMin, labelMin, labelMax } = this.readyData;
        inputMax = tf.tensor(inputMax);
        inputMin = tf.tensor(inputMin);
        labelMin = tf.tensor(labelMin);
        labelMax = tf.tensor(labelMax);

        // Generate predictions for a uniform range of numbers between 0 and 1;
        // We un-normalize the data by doing the inverse of the min-max scaling
        // that we did earlier.
        const testSize = 100;
        const [xs, preds] = tf.tidy(() => {
            const xs = tf.linspace(0, 1, testSize);
            const preds = model.predict(xs.reshape([testSize, 1]));
            // console.log(Array.from(xs.dataSync()), Array.from(preds.dataSync()))

            const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

            const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

            // Un-normalize the data
            return [unNormXs.dataSync(), unNormPreds.dataSync()];
        });

        // console.log(Array.from(xs), Array.from(preds))

        const predictedPoints = Array.from(xs).map((val, i) => {
            return { x: val, y: preds[i] };
        });

        const originalPoints = data.map((d) => ({
            x: d.horsepower,
            y: d.mpg
        }));

        tfvis.render.scatterplot(
            { name: "Model Predictions vs Original Data", tab: "Test" },
            {
                values: [originalPoints, predictedPoints],
                series: ["original", "predicted"]
            },
            {
                xLabel: "Horsepower",
                yLabel: "MPG",
                height: 300
            }
        );
    }

    visualize_data() {
        const { data } = this;
        const values = data.map((d) => ({
            x: d.horsepower,
            y: d.mpg
        }));

        tfvis.render.scatterplot(
            { name: "Horsepower v MPG", tab: "Data" },
            { values },
            {
                xLabel: "Horsepower",
                yLabel: "MPG",
                height: 300
            }
        );
    }

    visualize_model() {
        const { model } = this;

        tfvis.show.modelSummary({ name: "Model Summary", tab: "Model" }, model);
    }

    async download(filename) {
        const { model } = this;
        if (!model) return;
        await model.save(`downloads://${filename}`);
    }
}