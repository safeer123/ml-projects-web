import * as poseDetection from "@tensorflow-models/pose-detection";

export function keypointsToVecs(keypoints) {
    return poseDetection.util
      .getAdjacentPairs(poseDetection.SupportedModels.MoveNet)
      .map(([i, j]) => {
        const kp1 = keypoints[i];
        const kp2 = keypoints[j];

        return [kp1.x - kp2.x, kp1.y - kp2.y];
      });
  }