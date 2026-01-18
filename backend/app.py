import os
import cv2
import numpy as np
import keras
import joblib
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import Counter

app = Flask(__name__)
CORS(app)  # allows React to talk to Flask

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "powerlifting_unified_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "master_scaler.pkl")
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# AI Setup
WINDOW_SIZE = 30
SELECTED_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27]
EXERCISE_MAP = {0: "Squat", 1: "Bench Press",
                2: "Sumo Deadlift", 3: "Conventional Deadlift"}

# Load Model & MediaPipe
model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence_buffer, exercise_results, technique_results = [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            data = []
            for idx in SELECTED_LANDMARKS:
                lm = results.pose_landmarks.landmark[idx]
                data.extend([lm.x, lm.y, lm.visibility])

            sequence_buffer.append(data)
            if len(sequence_buffer) == WINDOW_SIZE:
                input_scaled = scaler.transform(np.array(sequence_buffer))
                preds = model.predict(
                    input_scaled.reshape(1, 30, 33), verbose=0)

                technique_results.append(float(preds[0][0]) > 0.5)
                exercise_results.append(EXERCISE_MAP.get(
                    np.argmax(preds[1][0]), "Unknown"))
                sequence_buffer.pop(0)

    cap.release()
    if not exercise_results:
        return {"error": "No exercise detected"}

    final_ex = Counter(exercise_results).most_common(1)[0][0]
    score = (sum(technique_results) / len(technique_results)) * 100
    return {"exercise": final_ex, "score": round(score, 1), "verdict": "GOOD" if score > 70 else "BAD"}


@app.route('/analyze', methods=['POST'])
def upload():
    file = request.files['video']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    result = analyze_video(path)
    os.remove(path)  # cleanup
    return jsonify(result)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
