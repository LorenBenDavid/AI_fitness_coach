import cv2
import numpy as np
import keras
import joblib
import mediapipe as mp
from collections import Counter
import os

# --- DYNAMIC PATH CONFIGURATION ---
# Get the absolute path of the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the project root (backend folder) which is one level up from scripts
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Define paths to the trained AI assets located in the 'models' folder
MODEL_PATH = os.path.join(BASE_DIR, "models", "powerlifting_unified_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "master_scaler.pkl")
# Path to the test video
VIDEO_PATH = os.path.join(BASE_DIR, "MyVideos", "Tests", "g-b.mp4")

# Window size matches the training configuration
WINDOW_SIZE = 30

# Label mapping (Ensure this matches the Exercise_Label order used in training)
EXERCISE_MAP = {
    0: "Squat",
    1: "Bench Press",
    2: "Sumo Deadlift",
    3: "Conventional Deadlift"
}

# The 11 landmarks used: Should result in exactly 33 features (x, y, vis)
SELECTED_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27]

print("🔄 Loading AI Model and Scaler...")
try:
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Load Successful (Synced for 33 features).")
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")
    exit()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(VIDEO_PATH)
sequence_buffer = []  # Sliding window storage
exercise_results = []  # History for final report
technique_results = []

# Display defaults
current_ex = "Waiting..."
current_tech = "Analyzing..."
tech_color = (255, 255, 255)

print(f"🎬 Analysis started on: {VIDEO_PATH}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Standardize frame for processing
    frame = cv2.resize(frame, (800, 600))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Draw skeleton on the visual output
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Feature Extraction: Collecting 33 values (11 points * 3 data points)
        current_frame_data = []
        for idx in SELECTED_LANDMARKS:
            lm = results.pose_landmarks.landmark[idx]
            current_frame_data.extend([lm.x, lm.y, lm.visibility])

        sequence_buffer.append(current_frame_data)

        # Start prediction once the buffer matches the model's required window size
        if len(sequence_buffer) == WINDOW_SIZE:
            # Prepare data: (30, 33)
            input_array = np.array(sequence_buffer).astype(np.float32)
            input_array = np.nan_to_num(input_array)  # Replace any NaNs with 0

            try:
                # --- SYNCED SCALING ---
                # Scaling directly with 33 features (No padding needed now)
                input_scaled = scaler.transform(input_array)

                # Reshape for LSTM: (Batch, Timesteps, Features)
                input_reshaped = input_scaled.reshape(1, WINDOW_SIZE, 33)

                # Inference
                preds = model.predict(input_reshaped, verbose=0)

                # Extract outputs from the multi-output model structure
                # Output 0 = Technique (Sigmoid), Output 1 = Exercise (Softmax)
                if isinstance(preds, list):
                    tech_score = float(preds[0][0])
                    ex_idx = np.argmax(preds[1][0])
                else:
                    tech_score = float(preds[0][0])
                    ex_idx = np.argmax(preds[0][1:])

                # Update Display Text
                current_ex = EXERCISE_MAP.get(ex_idx, "Unknown")
                is_good = tech_score > 0.5
                current_tech = "GOOD" if is_good else "BAD"
                tech_color = (0, 255, 0) if is_good else (0, 0, 255)

                # Collect results for summary
                technique_results.append(is_good)
                exercise_results.append(current_ex)

            except Exception as e:
                print(f"⚠️ Prediction Error: {e}")

            # Slide window: Remove the oldest frame
            sequence_buffer.pop(0)

    # --- UI RENDERING ---
    cv2.rectangle(frame, (0, 0), (450, 110), (0, 0, 0), -1)
    cv2.putText(frame, f"EXERCISE: {current_ex.upper()}", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"TECHNIQUE: {current_tech}", (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1, tech_color, 3)

    cv2.imshow('Powerlifting AI - Real-time Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- SUMMARY REPORT ---
print("\n" + "="*40)
print("📊 FINAL ANALYSIS REPORT")
print("="*40)
if exercise_results:
    # Get the most common exercise detected
    final_exercise = Counter(exercise_results).most_common(1)[0][0]
    # Calculate percentage of frames with good form
    good_pct = (sum(technique_results) / len(technique_results)) * 100

    print(f"🔹 Detected Exercise: {final_exercise}")
    print(f"🔹 Quality Consistency: {good_pct:.1f}%")
    print(f"🔹 Verdict: {'GOOD FORM ✅' if good_pct > 70 else 'BAD FORM ❌'}")
print("="*40)
