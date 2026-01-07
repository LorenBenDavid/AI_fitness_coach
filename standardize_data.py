import pandas as pd
import os

# The files you have in your folder
FILES = [
    "Dataset_Squat.csv",
    "Dataset_Sumo_Deadlift.csv",
    "Dataset_Conventional_Deadlift.csv",
    "Dataset_Bench.csv"
]


def prepare_data():
    print("🧹 Starting standardization process...")
    for file in FILES:
        if not os.path.exists(file):
            print(f"⚠️ Skipping {file} - File not found.")
            continue

        # 1. Load data and skip the broken lines (like row 27892)
        df = pd.read_csv(file, on_bad_lines='skip')

        # 2. Identify labels and numeric features
        if 'Label' not in df.columns:
            print(f"❌ Error in {file}: No 'Label' column found.")
            continue

        labels = df['Label']
        # Take only the numeric columns and drop the Label
        features = df.select_dtypes(include=['number']).drop(
            columns=['Label'], errors='ignore')

        # 3. FORCE 33 COLUMNS (11 landmarks * 3 values)
        # If it's Bench (24 columns), it adds 9 columns of zeros
        if len(features.columns) < 33:
            print(
                f"📊 {file}: Adding padding (current features: {len(features.columns)})")
            for i in range(len(features.columns), 33):
                features[f'padding_{i}'] = 0.0
        elif len(features.columns) > 33:
            print(
                f"✂️ {file}: Trimming extra features (current features: {len(features.columns)})")
            features = features.iloc[:, :33]

        # 4. Save the clean, uniform version back to the same file
        final_df = features.copy()
        final_df['Label'] = labels
        final_df.to_csv(file, index=False)
        print(f"✅ {file} is now standardized to 33 features.")


if __name__ == "__main__":
    prepare_data()
