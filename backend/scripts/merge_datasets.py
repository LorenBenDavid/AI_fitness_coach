import pandas as pd
import os


def merge_datasets_standardized():
    exercise_mapping = {
        "Squat": 0,
        "Bench_Press": 1,
        "Sumo_Deadlift": 2,
        "Conventional_Deadlift": 3
    }

    YOUTUBE_DIR = "Data_Sets/Youtube"
    MY_DIR = "Data_Sets/My"
    FINAL_DIR = "Data_Sets/Final"

    if not os.path.exists(FINAL_DIR):
        os.makedirs(FINAL_DIR)

    for ex_name, ex_id in exercise_mapping.items():
        youtube_file = os.path.join(YOUTUBE_DIR, f"Youtube_{ex_name}.csv")
        my_file = os.path.join(MY_DIR, f"My_{ex_name}.csv")
        output_file = os.path.join(FINAL_DIR, f"Dataset_{ex_name}.csv")

        all_dfs = []

        print(f"\n🔍 Processing {ex_name}...", flush=True)

        # Process YouTube File
        if os.path.exists(youtube_file):
            print(f"✅ Reading YouTube: {youtube_file}")
            # FIX: Added on_bad_lines='skip' and low_memory=False to handle the error
            df_yt = pd.read_csv(
                youtube_file, on_bad_lines='skip', low_memory=False)

            # Ensure consistent columns
            df_yt['Exercise_Label'] = ex_id
            df_yt['Technique'] = 1.0
            all_dfs.append(df_yt)

        # Process My File
        if os.path.exists(my_file):
            print(f"✅ Reading My Data: {my_file}")
            # FIX: Also added here just in case
            df_my = pd.read_csv(my_file, on_bad_lines='skip', low_memory=False)

            df_my['Exercise_Label'] = ex_id
            if 'Technique' not in df_my.columns and 'Label' in df_my.columns:
                df_my['Technique'] = df_my['Label']
            all_dfs.append(df_my)

        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            final_df.to_csv(output_file, index=False)
            print(
                f"✨ SUCCESS: Created {output_file} with {len(final_df)} rows")
        else:
            print(f"⚠️ No files found for {ex_name}")


if __name__ == "__main__":
    merge_datasets_standardized()
