import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
# Updated paths based on your folder structure: Data_Sets/Final/
FILES = [
    "Data_Sets/Final/Dataset_Squat.csv",
    "Data_Sets/Final/Dataset_Sumo_Deadlift.csv",
    "Data_Sets/Final/Dataset_Conventional_Deadlift.csv",
    "Data_Sets/Final/Dataset_Bench_Press.csv"
]

WINDOW_SIZE = 30  # Number of frames per sequence for the LSTM


def create_sequences(data, tech_labels, exercise_labels, window_size):
    """
    Transforms flat rows into 3D sequences (Samples, Time_Steps, Features).
    """
    X, y_tech, y_ex = [], [], []
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size])
        y_tech.append(tech_labels[i + window_size - 1])
        y_ex.append(exercise_labels[i + window_size - 1])
    return np.array(X), np.array(y_tech), np.array(y_ex)


def train_unified_model():
    print("📂 Step 1: Loading datasets from Data_Sets/Final/...")

    all_dfs = []

    # 1. Load each CSV file from the specific subfolder
    for file in FILES:
        if os.path.exists(file):
            df = pd.read_csv(file)
            all_dfs.append(df)
            print(f"✅ Loaded {file} successfully.")
        else:
            print(f"❌ Error: {file} NOT found. Check if the path is correct.")

    if not all_dfs:
        print("❌ No data available. Training aborted.")
        return

    # 2. Combine all data
    master_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    master_df = master_df.fillna(0)

    # 3. Separate Features and Targets
    # We drop 'Technique' and 'Exercise_Label' to keep only numeric landmarks in X
    X_raw = master_df.drop(['Technique', 'Exercise_Label'],
                           axis=1, errors='ignore').values
    y_tech_raw = master_df['Technique'].values
    y_ex_raw = master_df['Exercise_Label'].values

    print(f"📊 Total frames loaded: {X_raw.shape[0]}")
    print(f"📊 Features per frame: {X_raw.shape[1]} (33 landmarks expected)")

    # 4. Global Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 5. Create Sequences (Processing each exercise separately to avoid data mixing)
    all_X_seq, all_y_tech_seq, all_y_ex_seq = [], [], []
    start_idx = 0

    for df in all_dfs:
        end_idx = start_idx + len(df)

        X_group = X_scaled[start_idx:end_idx]
        y_t_group = df['Technique'].values
        y_e_group = df['Exercise_Label'].values

        X_s, yt_s, ye_s = create_sequences(
            X_group, y_t_group, y_e_group, WINDOW_SIZE)

        if len(X_s) > 0:
            all_X_seq.append(X_s)
            all_y_tech_seq.append(yt_s)
            all_y_ex_seq.append(ye_s)

        start_idx = end_idx

    X_final = np.concatenate(all_X_seq)
    y_tech_final = np.concatenate(all_y_tech_seq)
    y_ex_final = np.concatenate(all_y_ex_seq)

    # 6. Split into Training and Testing sets
    X_train, X_test, y_tech_train, y_tech_test, y_ex_train, y_ex_test = train_test_split(
        X_final, y_tech_final, y_ex_final, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"🧠 Step 2: Building Multi-Output LSTM Architecture...")

    # --- Model Definition ---
    inputs = Input(shape=(WINDOW_SIZE, X_train.shape[2]))

    # LSTM Layers
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)

    # Branch 1: Technique Quality (Good/Bad)
    tech_out = Dense(1, activation='sigmoid', name='technique_output')(x)

    # Branch 2: Exercise Classification (0, 1, 2, 3)
    ex_out = Dense(4, activation='softmax', name='exercise_output')(x)

    model = Model(inputs=inputs, outputs=[tech_out, ex_out])

    model.compile(
        optimizer='adam',
        loss={
            'technique_output': 'binary_crossentropy',
            'exercise_output': 'sparse_categorical_crossentropy'
        },
        metrics={
            'technique_output': 'accuracy',
            'exercise_output': 'accuracy'
        }
    )

    print("\n🚀 Step 3: Starting Training Process...")

    model.fit(
        X_train,
        {'technique_output': y_tech_train, 'exercise_output': y_ex_train},
        validation_data=(
            X_test, {'technique_output': y_tech_test, 'exercise_output': y_ex_test}),
        epochs=30,
        batch_size=32
    )

    # --- Step 4: Save Model & Scaler ---
    print("\n💾 Step 4: Saving final assets...")
    model.save("powerlifting_unified_model.keras")
    joblib.dump(scaler, "master_scaler.pkl")

    print("\n🏆 DONE! The model and scaler are ready for live detection.")


if __name__ == "__main__":
    train_unified_model()
