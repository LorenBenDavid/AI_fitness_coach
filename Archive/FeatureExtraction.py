import cv2
import os
import pandas as pd
import math
from PoseModule import poseDetector
from ExerciseConfig import EXERCISE_CONFIG


def calculate_angle(p1, p2, p3):
    """Calculates the angle between three points (x, y coordinates)."""
    # p1, p2, p3 are [id, x, y]
    x1, y1 = p1[1], p1[2]
    x2, y2 = p2[1], p2[2]
    x3, y3 = p3[1], p3[2]

    # Calculate the angle using arctan2
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                         math.atan2(y1 - y2, x1 - x2))
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle


def extract_features():
    # Load the routing map we created in DataPrep
    if not os.path.exists("video_routing_map.csv"):
        print("Error: video_routing_map.csv not found! Run DataPrep.py first.")
        return

    routing_df = pd.read_csv("video_routing_map.csv")
    detector = poseDetector(model_complexity=1)  # Stable for Mac
    all_data = []

    print(f"--- Starting Extraction for {len(routing_df)} videos ---")

    for index, row in routing_df.iterrows():
        video_link = row['link']
        exercise_name = row['exercise']
        view_angle = row['view']

        # In a real scenario, we'd use the local filename.
        # For now, let's assume the file is named based on a simple ID or link hash.
        video_path = f"NormVideos/video_{index}.mp4"

        if not os.path.exists(video_path):
            print(f"Skipping {video_path} (File not found)")
            continue

        cap = cv2.VideoCapture(video_path)
        print(f"Processing: {exercise_name} ({view_angle})")

        frame_count = 0
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            frame_count += 1
            # Process frame every 5 frames to save time/space
            if frame_count % 5 != 0:
                continue

            img = detector.findPose(img, draw=False)
            lmList = detector.findPosition(img, draw=False)

            if len(lmList) != 0:
                # Get specific joints from our ExerciseConfig
                config = EXERCISE_CONFIG[exercise_name][view_angle]
                joint_ids = config["joints"]

                # Extract the 3 points needed for the angle
                p1 = lmList[joint_ids[0]]
                p2 = lmList[joint_ids[1]]
                p3 = lmList[joint_ids[2]]

                angle = calculate_angle(p1, p2, p3)

                # Store data
                all_data.append({
                    "Exercise": exercise_name,
                    "View": view_angle,
                    "Angle": angle,
                    "Joint_A": joint_ids[0],
                    "Joint_B": joint_ids[1],
                    "Joint_C": joint_ids[2],
                    "Frame": frame_count
                })

        cap.release()

    # Save final dataset
    final_df = pd.DataFrame(all_data)
    final_df.to_csv("Final_Exercise_Dataset.csv", index=False)
    print("✅ Extraction Complete! Saved to Final_Exercise_Dataset.csv")


if __name__ == "__main__":
    extract_features()
