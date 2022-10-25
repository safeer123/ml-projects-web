import { useEffect } from "react";

import * as tf from "@tensorflow/tfjs";

export default function () {
  async function loadModel() {
    const model = await tf.loadLayersModel("models/model_test_01/model.json");

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
  useEffect(() => {
    loadModel();
  }, []);

  return <div className="App">Basic ANN</div>;
}
