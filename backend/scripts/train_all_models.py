import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras import Model

# --- MAC M1/M2 OPTIMIZATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- DYNAMIC PATH CONFIGURATION ---
# Get the absolute path of the 'scripts' folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Points to the 'backend' root folder
BASE_DIR = os.path.dirname(SCRIPT_DIR)
# Ensures 'models' folder is created in the backend root
SAVE_DIR = os.path.join(BASE_DIR, "models")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# List of dataset paths using absolute paths
FILES = [
    os.path.join(BASE_DIR, "Data_Sets/Final/Dataset_Squat.csv"),
    os.path.join(BASE_DIR, "Data_Sets/Final/Dataset_Sumo_Deadlift.csv"),
    os.path.join(
        BASE_DIR, "Data_Sets/Final/Dataset_Conventional_Deadlift.csv"),
    os.path.join(BASE_DIR, "Data_Sets/Final/Dataset_Bench_Press.csv")
]

# LSTM Hyperparameters
WINDOW_SIZE = 30


def create_sequences(data, tech_labels, exercise_labels, window_size):
    """
    Converts raw time-series data into overlapping sequences for LSTM input.
    """
    X, y_tech, y_ex = [], [], []
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size])
        y_tech.append(tech_labels[i + window_size - 1])
        y_ex.append(exercise_labels[i + window_size - 1])
    return np.array(X), np.array(y_tech), np.array(y_ex)


def train_unified_model():
    print("\n📂 Step 1: Loading CSV files and auto-detecting features...", flush=True)
    all_dfs_X = []
    all_dfs_y_tech = []
    all_dfs_y_ex = []

    for file in FILES:
        if os.path.exists(file):
            df = pd.read_csv(file).dropna()

            # Feature Selection: Use first 33 columns (11 landmarks * 3 values)
            X_features = df.iloc[:, :33].values

            if 'Technique' in df.columns and 'Exercise_Label' in df.columns:
                y_tech = df['Technique'].values
                y_ex = df['Exercise_Label'].values

                all_dfs_X.append(X_features)
                all_dfs_y_tech.append(y_tech)
                all_dfs_y_ex.append(y_ex)
                print(
                    f"✅ Loaded {os.path.basename(file)}: {X_features.shape[1]} features.")
            else:
                print(f"❌ Missing labels in {file}.")
        else:
            print(f"⚠️ File not found: {file}")

    if not all_dfs_X:
        print("❌ No valid data found! Training aborted.")
        return

    # Data Processing
    X_raw = np.concatenate(all_dfs_X)
    y_tech_raw = np.concatenate(all_dfs_y_tech)
    y_ex_raw = np.concatenate(all_dfs_y_ex)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Sequence Generation
    all_X_seq, all_y_tech_seq, all_y_ex_seq = [], [], []
    curr_idx = 0
    for i in range(len(all_dfs_X)):
        df_len = len(all_dfs_X[i])
        X_group = X_scaled[curr_idx: curr_idx + df_len]
        Xs, yts, yes = create_sequences(
            X_group, all_dfs_y_tech[i], all_dfs_y_ex[i], WINDOW_SIZE)
        if len(Xs) > 0:
            all_X_seq.append(Xs)
            all_y_tech_seq.append(yts)
            all_y_ex_seq.append(yes)
        curr_idx += df_len

    X_final = np.concatenate(all_X_seq)
    y_tech_final = np.concatenate(all_y_tech_seq)
    y_ex_final = np.concatenate(all_y_ex_seq)

    print(f"\n🧠 Step 2: Building Multi-Output LSTM Architecture...", flush=True)

    inputs = Input(shape=(WINDOW_SIZE, 33))

    # LSTM Layers with slightly higher Dropout to prevent overfitting
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)

    # Technique Output
    tech_out = Dense(1, activation='sigmoid', name='technique_output')(x)

    # Exercise Output (4 categories)
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

    print("\n🚀 Step 3: Starting Training (15 Epochs - Optimized)...", flush=True)

    # Training with shuffle enabled to reduce bias
    model.fit(
        X_final,
        {
            'technique_output': y_tech_final,
            'exercise_output': y_ex_final
        },
        epochs=15,             # Reduced from 30 to 15 to prevent overfitting
        batch_size=32,
        verbose=1,
        validation_split=0.2,
        shuffle=True           # Mixes data to improve differentiation
    )

    # Saving assets
    model_save_path = os.path.join(SAVE_DIR, "powerlifting_unified_model.h5")
    scaler_save_path = os.path.join(SAVE_DIR, "master_scaler.pkl")

    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)

    print(f"\n🏆 SUCCESS! Files saved in: {SAVE_DIR}")


if __name__ == "__main__":
    train_unified_model()
