
export const objective_content = (
    <>
        Predict human body pose for each video frame and provide a label corresponding to
        a list of pose classes. For collecting data we use
        <a href="https://www.tensorflow.org/hub/tutorials/movenet" target={"_blank"}>MoveNet's Lightening</a>
        pose estimation model and obtain 17 keypoints. Then adjacent keypoints are subtracted to obtain
        16 vectors. By normalizing these vectors, we get 16 unit vectors, which forms the input data
        for our deep learning model. Note that, here we assume, these 16 unit vectors are independent of the video frame's
        size or resolution. We can collect the data from the web cam against each class, and train an ANN model.
        You may change the pose labels by simply adding a custom text input. Try out the following.
        <ul>
            <li>Step1: Enable the web camera (Provide access on prompt)</li>
            <li>Step2: Collect the data for each class (Change the labels if you wish)</li>
            <li>Step3: Create a model with input vector shape (32,). We have 16 2D vectors.</li>
            <li>Step4: Train the model and predict for new poses using the webcam input</li>
        </ul>
        <p>Data Flow:<br/> VideoFrames &#8594; [MoveNet Model] &#8594; [Our Model] &#8594; output</p>
    </>)