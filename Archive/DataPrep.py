import os
import cv2
import pandas as pd
from youtubesearchpython import VideosSearch
from ExerciseConfig import EXERCISE_CONFIG

# --- SETTINGS ---
NORM_VIDEOS_DIR = "NormVideos"
os.makedirs(NORM_VIDEOS_DIR, exist_ok=True)


def search_and_get_links(query, limit=10):
    """Searches YouTube and returns a list of video links."""
    print(f"🔍 Searching YouTube for: '{query}'")
    videos_search = VideosSearch(query, limit=limit)
    results = videos_search.result()
    return [video['link'] for video in results['result']]


def download_youtube_videos(links, output_dir):
    """Placeholder for your existing download logic (pytube/yt-dlp)."""
    # Here you would call your pytube/yt-dlp download function
    print(f"📥 Downloading {len(links)} videos to {output_dir}...")
    pass


if __name__ == "__main__":
    # This list will store all our routing data to eventually save to a CSV/JSON
    # Mapping: Video Link -> {Exercise Name, View Angle}
    video_routing_data = []

    print("--- STEP 1: SMART HUNTING (FRONT & SIDE) ---")

    # We loop through each base exercise defined in your ExerciseConfig
    for exercise_name in EXERCISE_CONFIG.keys():
        # For each exercise, we perform two searches:
        for view_angle in ["Side", "Front"]:
            search_query = f"{exercise_name} {view_angle} View"
            links = search_and_get_links(
                search_query, limit=5)  # Start small for testing

            for link in links:
                video_routing_data.append({
                    "link": link,
                    "exercise": exercise_name,
                    "view": view_angle
                })

    # Convert to DataFrame for easy management
    routing_df = pd.DataFrame(video_routing_data)
    unique_links = routing_df['link'].unique().tolist()

    print(f"\n✅ Total metadata entries created: {len(routing_df)}")
    print(f"✅ Total unique videos to download: {len(unique_links)}")

    # --- STEP 2: DOWNLOAD ---
    # download_youtube_videos(unique_links, NORM_VIDEOS_DIR)

    # --- STEP 3: PREPARE FOR EXTRACTION ---
    # Save the routing map so the Extraction script knows what to do with each file
    routing_df.to_csv("video_routing_map.csv", index=False)
    print("📂 Routing map saved to 'video_routing_map.csv'")
