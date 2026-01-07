import cv2
import mediapipe as mp
import pandas as pd
import os

# --- CONFIGURATION ---
# Change this for each exercise you are collecting (Squat, Bench_Press, etc.)
EXERCISE_NAME = "Sumo_Deadlift"
# Ensure your Mac folder structure is: MyVideos/Bench_Press/Good and MyVideos/Bench_Press/Bad
BASE_FOLDER = f"MyVideos/{EXERCISE_NAME}"
OUTPUT_CSV = f"Dataset_{EXERCISE_NAME}.csv"

# The 11 landmarks we agreed upon (33 features total: X, Y, Visibility for each)
SELECTED_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def collect_with_technique():
    all_rows = []

    # Iterate through both technique sub-folders
    for status in ["good", "bad"]:
        folder_path = os.path.join(BASE_FOLDER, status)

        # Check if the sub-folder exists to avoid errors
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Folder {folder_path} not found, skipping...")
            continue

        # Support common video formats on Mac (.mp4, .mov, .MOV)
        video_files = [f for f in os.listdir(
            folder_path) if f.endswith(('.mp4', '.mov', '.MOV'))]
        print(
            f"🚀 Processing {len(video_files)} {status} videos for {EXERCISE_NAME}...")

        for video_name in video_files:
            video_path = os.path.join(folder_path, video_name)
            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # Convert BGR (OpenCV) to RGB (MediaPipe)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    row = []
                    # Extract features for the selected 11 landmarks
                    for idx in SELECTED_LANDMARKS:
                        lm = results.pose_landmarks.landmark[idx]
                        row.extend([lm.x, lm.y, lm.visibility])

                    # Append Metadata: Exercise Label and Technique Label
                    row.append(EXERCISE_NAME)  # e.g., "Bench_Press"
                    # 1 for Good, 0 for Bad
                    row.append(1 if status == "good" else 0)

                    all_rows.append(row)

            cap.release()
            print(f"  ✅ Finished video: {video_name}")

    if not all_rows:
        print("❌ No data was collected. Please check your video files.")
        return

    # Create a DataFrame from the collected rows
    df = pd.DataFrame(all_rows)

    # If the CSV already exists, append new data without writing the header again
    if os.path.exists(OUTPUT_CSV):
        df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
        print(f"📂 Appended new data to {OUTPUT_CSV}")
    else:
        # If it's a new file, create headers for 33 features + Label + Technique
        headers = []
        for i in range(len(SELECTED_LANDMARKS)):
            headers.extend([f'x{i}', f'y{i}', f'v{i}'])
        headers.extend(['Label', 'Technique'])

        df.to_csv(OUTPUT_CSV, header=headers, index=False)
        print(f"🆕 Created new file and saved data: {OUTPUT_CSV}")

    print(f"\n✨ FINAL STATUS: Added {len(all_rows)} rows of data.")


if __name__ == "__main__":
    collect_with_technique()
