import os
import sys

# 1. MAC FIX: Force terminal to show output immediately
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("--- DEBUG: SCRIPT STARTED ---", flush=True)

# 2. LOAD LIBRARIES ONE BY ONE
try:
    print("⏳ Loading Pandas...", end=" ", flush=True)
    import pandas as pd
    print("OK")

    print("⏳ Loading OpenCV...", end=" ", flush=True)
    import cv2
    print("OK")

    print("⏳ Loading YouTube Tools...", end=" ", flush=True)
    import yt_dlp
    from youtubesearchpython import VideosSearch
    print("OK")

    print("⏳ Loading MediaPipe & Custom Logic...", end=" ", flush=True)
    from MyPoseLogic import poseDetector
    from ExerciseConfig import EXERCISE_CONFIG
    print("OK")
except Exception as e:
    print(f"\n❌ Error during import: {e}")
    sys.exit()

# --- CONFIGURATION ---
EXERCISE_NAME = "Squat"
BASE_FOLDER = f"MyVideos/{EXERCISE_NAME}"
OUTPUT_FOLDER = "Data_Sets/My"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, f"My_{EXERCISE_NAME}.csv")
SELECTED_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27]

# Initialize detector
detector = poseDetector()


def collect_with_technique():
    print(f"\n🚀 Processing: {EXERCISE_NAME}", flush=True)
    all_rows = []

    for status in ["good", "bad"]:
        folder_path = os.path.join(BASE_FOLDER, status)
        if not os.path.exists(folder_path):
            print(f"⚠️ Folder missing: {folder_path}")
            continue

        video_files = [f for f in os.listdir(
            folder_path) if f.endswith(('.mp4', '.mov', '.MOV'))]
        print(f"📂 Found {len(video_files)} {status} videos.")

        for video_name in video_files:
            video_path = os.path.join(folder_path, video_name)
            cap = cv2.VideoCapture(video_path)
            print(f"   📹 Analyzing: {video_name}...", end=" ", flush=True)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                frame = detector.findPose(frame, draw=False)
                lmList = detector.findPosition(frame, draw=False)

                if lmList:
                    row = []
                    for idx in SELECTED_LANDMARKS:
                        lm = lmList[idx]
                        row.extend([lm[1], lm[2], 1.0])
                    row.append(EXERCISE_NAME)
                    row.append(1 if status == "good" else 0)
                    all_rows.append(row)

            cap.release()
            print("Done ✅")

    if all_rows:
        df = pd.DataFrame(all_rows)
        headers = []
        for i in range(len(SELECTED_LANDMARKS)):
            headers.extend([f'x{i}', f'y{i}', f'v{i}'])
        headers.extend(['Label', 'Technique'])
        df.to_csv(OUTPUT_CSV, header=headers, index=False)
        print(f"\n✨ SUCCESS! Saved to: {OUTPUT_CSV}")
    else:
        print("\n❌ No data collected. Check video paths.")


if __name__ == "__main__":
    collect_with_technique()
