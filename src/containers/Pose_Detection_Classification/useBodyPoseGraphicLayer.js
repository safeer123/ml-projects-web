import * as poseDetection from "@tensorflow-models/pose-detection";

const KP_SCORE_THRESHOLD = 0.3
const KEY_POINT_RADIUS = 4
const SKELETON_LINE_WIDTH = 2
const SKELETON_LINE_COLOR = "#FFF"

export default function ({
    ctx2d,
}) {

  function drawSkeleton(keypoints) {
    // Each poseId is mapped to a color in the color palette.
    ctx2d.fillStyle = SKELETON_LINE_COLOR;
    ctx2d.strokeStyle = SKELETON_LINE_COLOR;
    ctx2d.lineWidth = SKELETON_LINE_WIDTH;

    poseDetection.util
      .getAdjacentPairs(poseDetection.SupportedModels.MoveNet)
      .forEach(([i, j]) => {
        const kp1 = keypoints[i];
        const kp2 = keypoints[j];

        drawSegment(kp1, kp2);
      });
  }

  function drawSegment(kp1, kp2) {
    // console.log(kp1, kp2)
    // If score is null, just show the keypoint.
    const score1 = kp1.score != null ? kp1.score : 1;
    const score2 = kp2.score != null ? kp2.score : 1;
    ctx2d.setLineDash([20, 5]);

    if (score1 >= KP_SCORE_THRESHOLD && score2 >= KP_SCORE_THRESHOLD) {
      ctx2d.beginPath();
      ctx2d.moveTo(kp1.x, kp1.y);
      ctx2d.lineTo(kp2.x, kp2.y);
      ctx2d.stroke();
    }
    ctx2d.setLineDash([]);
  }

  function drawKeypoints(keypoints) {
    const keypointInd = poseDetection.util.getKeypointIndexBySide(
      poseDetection.SupportedModels.MoveNet
    );
    ctx2d.fillStyle = "red";
    ctx2d.strokeStyle = "white";
    ctx2d.lineWidth = 2;

    for (const i of keypointInd.middle) {
      drawKeypoint(keypoints[i]);
    }

    ctx2d.fillStyle = "green";
    for (const i of keypointInd.left) {
      drawKeypoint(keypoints[i]);
    }

    ctx2d.fillStyle = "orange";
    for (const i of keypointInd.right) {
      drawKeypoint(keypoints[i]);
    }
  };

  function drawKeypoint(keypoint) {
    // If score is null, just show the keypoint.
    const score = keypoint.score != null ? keypoint.score : 1;

    if (score >= KP_SCORE_THRESHOLD) {
      const circle = new Path2D();
      circle.arc(keypoint.x, keypoint.y, KEY_POINT_RADIUS, 0, 2 * Math.PI);
      ctx2d.fill(circle);
      ctx2d.stroke(circle);
    }
  }

  function drawAnnotations({
    keypoints,
    hideSkeleton=false,
    hideKeypoints=false,
  } = {}) {
    if(!hideSkeleton) {
      // console.log(keypoints)
      drawSkeleton(keypoints);
    }
    if(!hideKeypoints) {
      drawKeypoints(keypoints);
    }
  }

  return {
    drawAnnotations,
  };
}
