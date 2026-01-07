import cv2
import numpy as np
import keras
import joblib
import os
import mediapipe as mp

# 1. Setup MediaPipe with direct access to solutions
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 2. Load the trained model and helper files
# Make sure these files are in the same folder as this script
print("🔄 Loading Model, Scaler, and Encoder...")
try:
    model = keras.models.load_model("powerlifting_sequence_model.keras")
    scaler = joblib.load("master_scaler.pkl")
    exercise_encoder = joblib.load("exercise_encoder.pkl")
    print("✅ All model files loaded successfully!")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit()

# 3. Video Configuration
video_path = "PoseVideos/2.mov"
if not os.path.exists(video_path):
    print(f"❌ Error: Video file not found at {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)

# Buffer to store sequences (Must be same as WINDOW_SIZE used in training)
WINDOW_SIZE = 30
sequence_buffer = []

print(f"🎬 Processing video: {video_path} (Press 'q' to stop)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("🏁 End of video or cannot read file.")
        break

    # Resize frame for better performance on Mac
    frame = cv2.resize(frame, (640, 480))

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Draw skeleton on the frame
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract Landmark coordinates (x, y, z, visibility)
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

        # Add current frame's landmarks to buffer
        sequence_buffer.append(landmarks)

        # Only predict if we have enough frames for a full sequence (30 frames)
        if len(sequence_buffer) == WINDOW_SIZE:
            # Prepare data: Convert to numpy array and scale
            input_data = np.array(sequence_buffer)
            input_data_scaled = scaler.transform(input_data)

            # Reshape to (1, 30, Features) to match model input
            input_data_reshaped = input_data_scaled.reshape(1, WINDOW_SIZE, -1)

            # Perform Prediction
            # prediction[0] is Technique, prediction[1] is Exercise Type
            prediction = model.predict(input_data_reshaped, verbose=0)

            tech_prob = prediction[0][0][0]  # Probability of "Good" technique
            # Probabilities for each exercise class
            ex_prob = prediction[1][0]

            # Identify the exercise name
            ex_index = np.argmax(ex_prob)
            ex_name = exercise_encoder.inverse_transform([ex_index])[0]

            # Define Feedback based on 0.5 threshold
            feedback = "GOOD" if tech_prob > 0.5 else "BAD"
            color = (0, 255, 0) if feedback == "GOOD" else (
                0, 0, 255)  # Green for Good, Red for Bad

            # UI: Display Exercise Name and Feedback on frame
            # Black background for text
            cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
            cv2.putText(frame, f"Exercise: {ex_name}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Technique: {feedback}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Confidence: {tech_prob:.2f}", (240, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Slide the window: remove the oldest frame
            sequence_buffer.pop(0)

    # Show the final processed frame
    cv2.imshow('Powerlifting AI Feedback', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("👋 Process finished.")
