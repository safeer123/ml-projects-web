import BasicANN from "../containers/BasicANN";
import ANN_Cars_miles_per_gallon from "../containers/ANN_Cars_miles_per_gallon";
import CNN_MINST_handwritten_num_classification from "../containers/CNN_MINST_handwritten_num_classification";
import Smart_Webcam from "../containers/Smart_Webcam";
import Teachable_machine from "../containers/Teachable_machine";
import Pose_Detection_Classification from "../containers/Pose_Detection_Classification";
import Pose_Classification_pretrained_model from "../containers/Pose_Classification_pretrained_model";

export default [
  {
    route: "/ann_example_basic",
    title: "Basic ANN classification example",
    description: "",
    component: BasicANN,
    publish: true,
  },
  {
    route: "/ann_cars_mpg",
    title: "ANN regression - Cars mpg",
    description: "",
    component: ANN_Cars_miles_per_gallon,
    publish: true,
  },
  {
    route: "/ann_minst_num_classification",
    title: "CNN minst number classification",
    description: "",
    component: CNN_MINST_handwritten_num_classification,
    publish: true,
  },
  {
    route: "/smart_webcam",
    title: "Smart Webcam",
    description: "",
    component: Smart_Webcam,
    publish: true,
  },
  {
    route: "/teachable_machine",
    title: "Teachable Machine",
    description: "",
    component: Teachable_machine,
    publish: true,
  },
  {
    route: "/pose_classification",
    title: "Pose Classification",
    description: "",
    component: Pose_Detection_Classification,
    publish: true,
  },
  {
    route: "/pose_classification_pre_trained",
    title: "Pose Classification - Pre-trained Model",
    description: "",
    component: Pose_Classification_pretrained_model,
    publish: false,
  }
];
