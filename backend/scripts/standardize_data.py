import pandas as pd
import os

# Updated paths to look inside your Final folder
FILES = [
    "Data_Sets/Final/Dataset_Squat.csv",
    "Data_Sets/Final/Dataset_Sumo_Deadlift.csv",
    "Data_Sets/Final/Dataset_Conventional_Deadlift.csv",
    "Data_Sets/Final/Dataset_Bench_Press.csv"  # Fixed name to match your folders
]


def prepare_data():
    print("🧹 Starting standardization process in Data_Sets/Final...", flush=True)
    for file in FILES:
        if not os.path.exists(file):
            print(f"⚠️ Skipping {file} - File not found.")
            continue

        print(f"📖 Processing {file}...", end=" ", flush=True)

        # 1. Load data and skip the broken lines
        df = pd.read_csv(file, on_bad_lines='skip')

        # 2. Identify target columns to keep
        # We want to keep: 'Technique' and 'Exercise_Label'
        targets = df[['Technique', 'Exercise_Label']].copy(
        ) if 'Technique' in df.columns else pd.DataFrame()

        # Take only the numeric columns for landmarks
        features = df.select_dtypes(include=['number']).drop(
            columns=['Technique', 'Exercise_Label', 'Label'], errors='ignore')

        # 3. FORCE 33 COLUMNS (11 landmarks * 3 values: x,y,v)
        if len(features.columns) < 33:
            for i in range(len(features.columns), 33):
                features[f'padding_{i}'] = 0.0
        elif len(features.columns) > 33:
            features = features.iloc[:, :33]

        # 4. Reconstruct the clean dataframe
        final_df = features.copy()
        for col in targets.columns:
            final_df[col] = targets[col]

        final_df.to_csv(file, index=False)
        print(f"✅ Standardized to 33 features.")


if __name__ == "__main__":
    prepare_data()
