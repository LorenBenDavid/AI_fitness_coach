import pandas as pd
import os


def merge_datasets_standardized():
    # Mapping exercise names to numeric IDs for the trainer
    exercise_mapping = {
        "Squat": 0,
        "Bench_Press": 1,
        "Sumo_Deadlift": 2,
        "Conventional_Deadlift": 3
    }

    for ex_name, ex_id in exercise_mapping.items():
        youtube_file = f"Youtube_{ex_name}.csv"
        my_file = f"My_{ex_name}.csv"
        output_file = f"Dataset_{ex_name}.csv"

        all_dfs = []

        # --- Process Files ---
        for file_path, is_youtube in [(youtube_file, True), (my_file, False)]:
            if os.path.exists(file_path):
                print(f"Reading {file_path}...")
                # Skip bad lines to handle any formatting errors
                df = pd.read_csv(file_path, on_bad_lines='skip')

                # 1. Identify all Landmark columns (J1 to J31 with X, Y, Vis)
                # This ensures we keep all 33 numerical columns from your screenshots
                landmark_cols = [c for c in df.columns if 'J' in c and (
                    '_X' in c or '_Y' in c or '_Vis' in c)]

                # 2. Create clean dataframe with only the landmarks
                df_clean = df[landmark_cols].copy()

                # 3. Add Exercise_Label (The ID of the exercise)
                df_clean['Exercise_Label'] = ex_id

                # 4. Add Technique (1 for YouTube, or use existing from your data)
                if is_youtube:
                    df_clean['Technique'] = 1.0  # Assume YouTube is correct
                else:
                    # Take 'Technique' if exists, otherwise use 'Label' column as quality indicator
                    if 'Technique' in df.columns:
                        df_clean['Technique'] = df['Technique']
                    elif 'Label' in df.columns:
                        df_clean['Technique'] = df['Label']

                all_dfs.append(df_clean)

        if all_dfs:
            # Combine everything into one dataset for this exercise
            final_df = pd.concat(all_dfs, ignore_index=True)

            # Save final file: Includes 33 landmarks + Exercise_Label + Technique = 35 columns
            final_df.to_csv(output_file, index=False)
            print(
                f"✅ Success: {output_file} created with {len(final_df.columns)} columns.\n")


if __name__ == "__main__":
    merge_datasets_standardized()
