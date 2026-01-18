import cv2
import math
import time
import mediapipe as mp
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing


class poseDetector():
    def __init__(self, static_image_mode=False, model_complexity=1,
                 smooth_landmarks=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpPose = mp_pose
        self.mpDraw = mp_drawing

        self.pose = self.mpPose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.results = None

    def findPose(self, img, draw=True):
        """Processes the image and finds the pose."""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=False):
        """
        Extracts landmarks as normalized coordinates.
        Crucial for AI training: We use raw floats (0.0 - 1.0) instead of pixels.
        Returns: [id, x, y, z, visibility]
        """
        lmList = []
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # We store raw normalized values (lm.x, lm.y)
                # instead of multiplying by image width/height.
                lmList.append([id, lm.x, lm.y, lm.z, lm.visibility])

                if draw:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        """Calculates angles based on pixel coordinates for visualization."""
        # Note: This is still useful for real-time feedback,
        # but the LSTM will learn these relationships automatically from raw data.
        lmList = self.findPosition(img, draw=False)
        if len(lmList) == 0:
            return 0

        h, w, c = img.shape
        # Convert normalized back to pixels for math/drawing
        x1, y1 = int(lmList[p1][1] * w), int(lmList[p1][2] * h)
        x2, y2 = int(lmList[p2][1] * w), int(lmList[p2][2] * h)
        x3, y3 = int(lmList[p3][1] * w), int(lmList[p3][2] * h)

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

# Example usage stays the same
