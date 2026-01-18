import cv2
import pandas as pd
import yt_dlp
import os
import time
from youtubesearchpython import VideosSearch
from MyPoseLogic import poseDetector
from ExerciseConfig import EXERCISE_CONFIG

# --- 1. COLUMN NAMES GENERATOR ---


def get_column_names(exercise_name, view_type):
    """
    Creates human-readable headers for the CSV file.
    Each joint will have 3 columns: X coordinate, Y coordinate, and Visibility.
    """
    relevant_joints = EXERCISE_CONFIG[exercise_name][view_type]["joints"]
    headers = []

    # Generate headers for each joint defined in your config
    for joint_id in relevant_joints:
        headers.extend(
            [f"J{joint_id}_X", f"J{joint_id}_Y", f"J{joint_id}_Vis"])

    # Metadata columns to help the model distinguish context
    headers.extend(["Label", "Exercise", "View", "Source_URL"])
    return headers

# --- 2. DATA NORMALIZATION ---


def normalize_landmarks(lmList, relevant_indices):
    """
    Normalizes coordinates relative to the Mid-Hip and scales by torso length.
    This ensures the model focuses on movement patterns rather than person size or camera distance.
    """
    try:
        # Step 1: Find the Center (Mid-Hip) using landmarks 23 and 24
        hip_l = next(l for l in lmList if l[0] == 23)
        hip_r = next(l for l in lmList if l[0] == 24)
        mid_hip_x = (hip_l[1] + hip_r[1]) / 2
        mid_hip_y = (hip_l[2] + hip_r[2]) / 2

        # Step 2: Calculate scale based on torso size (Shoulders to Hips)
        shldr_l = next(l for l in lmList if l[0] == 11)
        shldr_r = next(l for l in lmList if l[0] == 12)
        mid_shldr_y = (shldr_l[2] + shldr_r[2]) / 2

        # Add a tiny value (1e-6) to prevent division by zero
        scale = abs(mid_hip_y - mid_shldr_y) + 1e-6

        normalized = []
        for idx in relevant_indices:
            joint = next((l for l in lmList if l[0] == idx), None)
            if joint:
                # Normalizing: (Coordinate - Center) / Scale
                norm_x = (joint[1] - mid_hip_x) / scale
                norm_y = (joint[2] - mid_hip_y) / scale
                normalized.extend([norm_x, norm_y, joint[4]]
                                  )  # x, y, visibility
            else:
                # Padding for missing joints
                normalized.extend([0, 0, 0])
        return normalized
    except Exception:
        return None

# --- 3. VIDEO PROCESSING STREAM ---


def process_video_stream(url, exercise_name, view_type, detector, label):
    """
    Streams YouTube video, filters out static/standing frames, 
    and extracts normalized landmark data.
    """
    frame_data = []
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
        'proxy': "",

    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            stream_url = info['url']

        cap = cv2.VideoCapture(stream_url)
        relevant_joints = EXERCISE_CONFIG[exercise_name][view_type]["joints"]

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            # Detection using your PoseModule
            img = detector.findPose(img, draw=False)
            lmList = detector.findPosition(img, draw=False)

            if len(lmList) > 24:
                # SMART FILTERING: Calculate knee angle to detect active movement
                # If the person is just standing (Angle > 170), we skip the frame
                knee_angle = detector.findAngle(img, 24, 26, 28, draw=False)

                is_active = True
                if "Squat" in exercise_name or "Deadlift" in exercise_name:
                    if knee_angle > 170:
                        is_active = False  # Standing still

                if is_active:
                    features = normalize_landmarks(lmList, relevant_joints)
                    if features:
                        # Create a complete row: Features + Metadata
                        row = features + [label, exercise_name, view_type, url]
                        frame_data.append(row)

        cap.release()
        return frame_data
    except Exception as e:
        print(f"   ⚠️ Skipping video: {e}")
        return []


# --- 4. MAIN EXECUTION PIPELINE ---
if __name__ == "__main__":
    # Initialize the detector (optimized for Mac M1/M2)
    detector = poseDetector(model_complexity=1)

    # Goal: 100 videos for each type of exercise/form/view
    VIDEOS_PER_SEARCH = 100

    # Binary Labeling: 1 for Correct Form, 0 for Common Mistakes
    search_modes = [
        {"suffix": "perfect proper form technique tutorial", "label": 1},
        {"suffix": "common mistakes errors bad form", "label": 0}
    ]

    print("🚀 Starting Massive Exercise Data Extraction...")

    # OUTER LOOP: Iterates through Squat, Bench Press, Conv Deadlift, Sumo Deadlift
    for exercise_key in EXERCISE_CONFIG.keys():

        # Define a unique CSV file for each specific exercise
        clean_name = exercise_key.replace(" ", "_")
        CSV_FILE = f"Dataset_{clean_name}.csv"

        print(f"\n📂 DATASET TARGET: {CSV_FILE}")

        # INNER LOOP 1: Different views (Side, Front, etc.)
        for view_name in EXERCISE_CONFIG[exercise_key].keys():

            # INNER LOOP 2: Good Form vs. Bad Form (Label 1 vs. 0)
            for mode in search_modes:
                query = f"{exercise_key} {view_name} {mode['suffix']}"
                print(f"   🔍 YouTube Search: {query}")

                search = VideosSearch(query, limit=VIDEOS_PER_SEARCH)
                results = search.result()

                for video in results['result']:
                    link = video['link']
                    print(
                        f"      🎬 Processing: {video['title'][:30]}... (Label: {mode['label']})")

                    # Run extraction for the current video
                    video_results = process_video_stream(
                        link, exercise_key, view_name, detector, mode['label'])

                    if video_results:
                        df = pd.DataFrame(video_results)

                        # Apply human-readable headers
                        headers = get_column_names(exercise_key, view_name)
                        if len(df.columns) == len(headers):
                            df.columns = headers

                        # Save progress immediately using 'append' mode
                        # header is only written the first time the file is created
                        df.to_csv(CSV_FILE, mode='a', index=False,
                                  header=not os.path.isfile(CSV_FILE))
                        print(f"      ✅ Saved {len(video_results)} frames.")

                    # Short sleep to prevent YouTube IP limiting on Mac
                    time.sleep(1.2)

    print(f"\n🏁 ALL DONE! 4 Labeled Datasets are ready in your folder.")
