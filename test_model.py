import cv2
import numpy as np
import keras
import joblib
import os
import mediapipe as mp
from collections import Counter

# --- CONFIGURATION ---
MODEL_PATH = "powerlifting_unified_model.keras"
SCALER_PATH = "master_scaler.pkl"
VIDEO_PATH = "MyVideos/Tests/bad-sd.mp4"
WINDOW_SIZE = 30

EXERCISE_MAP = {
    0: "Squat",
    1: "Bench Press",
    2: "Sumo Deadlift",
    3: "Conventional Deadlift"
}

SELECTED_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27]

print("🔄 Loading AI Model and Scaler...")
try:
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Load Successful.")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit()

# Lists to store results for final feedback
exercise_results = []
technique_results = []

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(VIDEO_PATH)
sequence_buffer = []

print(f"🎬 Starting analysis on: {VIDEO_PATH}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        current_frame_data = []
        for idx in SELECTED_LANDMARKS:
            landmark = results.pose_landmarks.landmark[idx]
            current_frame_data.extend(
                [landmark.x, landmark.y, landmark.visibility])

        sequence_buffer.append(current_frame_data)

        if len(sequence_buffer) == WINDOW_SIZE:
            input_array = np.array(sequence_buffer)
            input_scaled = scaler.transform(input_array)
            input_reshaped = input_scaled.reshape(1, WINDOW_SIZE, 33)

            # Prediction
            predictions = model.predict(input_reshaped, verbose=0)

            # Technique logic
            tech_score = predictions[0][0][0]
            is_good = tech_score > 0.5
            technique_results.append(is_good)

            tech_label = "GOOD" if is_good else "BAD"
            tech_color = (0, 255, 0) if is_good else (0, 0, 255)

            # Exercise logic
            ex_probs = predictions[1][0]
            ex_idx = np.argmax(ex_probs)
            ex_name = EXERCISE_MAP.get(ex_idx, "Unknown")
            exercise_results.append(ex_name)

            # Live UI
            cv2.rectangle(frame, (10, 10), (450, 110), (0, 0, 0), -1)
            cv2.putText(frame, f"EXERCISE: {ex_name.upper()}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"TECHNIQUE: {tech_label} ({tech_score:.2f})", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, tech_color, 3)

            sequence_buffer.pop(0)

    cv2.imshow('Powerlifting AI - Live Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- FINAL FEEDBACK SECTION ---
print("\n" + "="*30)
print("📊 FINAL ANALYSIS REPORT")
print("="*30)

if exercise_results and technique_results:
    # 1. Final Exercise Identification (Most common prediction)
    final_exercise = Counter(exercise_results).most_common(1)[0][0]

    # 2. Final Technique Assessment
    good_count = sum(technique_results)
    total_count = len(technique_results)
    good_percentage = (good_count / total_count) * 100

    final_tech_label = "GOOD FORM ✅" if good_percentage > 70 else "BAD FORM ❌"

    print(f"🔹 Detected Exercise: {final_exercise}")
    print(f"🔹 Form Consistency: {good_percentage:.1f}% Good Frames")
    print(f"🔹 Final Verdict: {final_tech_label}")
else:
    print("❌ Not enough data was collected for a final report.")

print("="*30)
