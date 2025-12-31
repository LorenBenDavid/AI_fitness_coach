import cv2
import math
import mediapipe as mp

# Robust import for MediaPipe solutions to handle Colab 3.12+ environments
try:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing
except (ImportError, AttributeError):
    # Fallback for specific Python/Mediapipe distributions
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing


class poseDetector():
    """
    A class to detect human poses using MediaPipe.
    Optimized for LSTM training by extracting normalized coordinates.
    """

    def __init__(self, static_image_mode=False, model_complexity=1,
                 smooth_landmarks=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initializes the pose detector with specific confidence and complexity.
        :param model_complexity: 0 for Lite, 1 for Full, 2 for Heavy.
        """
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
        self.lmList = []

    def findPose(self, img, draw=True):
        """
        Processes an image and draws landmarks if requested.
        :param img: BGR image from OpenCV.
        :param draw: Boolean, whether to draw landmarks on the image.
        :return: Image with or without landmarks.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results and self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=False):
        """
        Extracts landmark positions as normalized coordinates.
        :return: List of [id, x, y, z, visibility]
        """
        self.lmList = []
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # Using raw normalized values (0.0 to 1.0) for AI model consistency
                self.lmList.append([id, lm.x, lm.y, lm.z, lm.visibility])

                if draw:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        """
        Calculates the angle between three joints.
        :param p1, p2, p3: Indices of the joints.
        :return: Calculated angle in degrees.
        """
        if not self.lmList:
            self.findPosition(img)

        if len(self.lmList) == 0:
            return 0

        h, w, c = img.shape
        # Convert normalized values back to pixel coordinates for math/drawing
        x1, y1 = int(self.lmList[p1][1] * w), int(self.lmList[p1][2] * h)
        x2, y2 = int(self.lmList[p2][1] * w), int(self.lmList[p2][2] * h)
        x3, y3 = int(self.lmList[p3][1] * w), int(self.lmList[p3][2] * h)

        # Calculate the angle using atan2
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
