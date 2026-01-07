import os
import pandas as pd
import numpy as np
import keras
from keras import Model
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Configuration - Ensure these filenames match your folder exactly
FILES = [
    "Dataset_Squat.csv",
    "Dataset_Sumo_Deadlift.csv",
    "Dataset_Conventional_Deadlift.csv",
    "Dataset_Bench_Press.csv"
]
# 30 frames = approx 1 second of video. Essential for movement analysis.
WINDOW_SIZE = 30


def create_sequences(data, tech_labels, exercise_labels, window_size):
    """
    Transforms flat rows into 3D sequences (Samples, Time_Steps, Features).
    This allows the LSTM to 'see' the motion over time.
    """
    X, y_tech, y_ex = [], [], []
    for i in range(len(data) - window_size):
        # Extract a window of frames
        X.append(data[i: i + window_size])
        # We take the label of the last frame in the sequence as the target
        y_tech.append(tech_labels[i + window_size - 1])
        y_ex.append(exercise_labels[i + window_size - 1])
    return np.array(X), np.array(y_tech), np.array(y_ex)


def train_unified_model():
    all_X = []
    all_y_tech = []
    all_y_ex = []

    # Pre-configure encoders
    exercise_encoder = LabelEncoder()
    exercise_names = [f.replace("Dataset_", "").replace(
        ".csv", "") for f in FILES]
    exercise_encoder.fit(exercise_names)
    scaler = StandardScaler()

    print("📂 Step 1: Loading and Sequencing Data...")

    for file in FILES:
        if os.path.exists(file):
            # Load CSV - skip lines with errors
            df = pd.read_csv(file, on_bad_lines='skip')

            # Extract Labels: Technique (0/1) and Exercise Name
            y_tech_raw = df['Label'].values
            exercise_name = file.replace("Dataset_", "").replace(".csv", "")
            y_ex_raw = np.full(
                len(df), exercise_encoder.transform([exercise_name])[0])

            # Select only Numeric Columns (Landmarks) and drop non-numeric labels
            X_numeric = df.select_dtypes(include=[np.number]).drop(
                ['Label'], axis=1, errors='ignore').values

            # Normalize data (Scale to 0-1 range roughly)
            X_scaled = scaler.fit_transform(X_numeric)

            # Create sequences per file to prevent mixing different exercises at the boundaries
            X_seq, y_tech_seq, y_ex_seq = create_sequences(
                X_scaled, y_tech_raw, y_ex_raw, WINDOW_SIZE)

            all_X.append(X_seq)
            all_y_tech.append(y_tech_seq)
            all_y_ex.append(y_ex_seq)
            print(f"✅ Processed {file} into {len(X_seq)} sequences.")
        else:
            print(f"⚠️ Warning: {file} not found. Skipping.")

    # Combine all sequences into one master dataset
    X_final = np.concatenate(all_X)
    y_tech_final = np.concatenate(all_y_tech)
    y_ex_final = np.concatenate(all_y_ex)

    # Split into 80% Training and 20% Testing (with Shuffling)
    X_train, X_test, y_tech_train, y_tech_test, y_ex_train, y_ex_test = train_test_split(
        X_final, y_tech_final, y_ex_final, test_size=0.2, random_state=42
    )

    # Should be (Samples, 30, Landmarks)
    print(f"📊 Training set size: {X_train.shape}")

    print("🧠 Step 2: Building Multi-Output LSTM Model...")
    # Define input layer matching our window size and number of landmarks
    inputs = Input(shape=(WINDOW_SIZE, X_train.shape[2]))

    # LSTM Layers - Processing the movement flow
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)  # Dropout to prevent memorizing data (Overfitting)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)

    # Head 1: Binary Output for Technique (0 = Bad, 1 = Good)
    tech_out = Dense(1, activation='sigmoid', name='technique_output')(x)

    # Head 2: Multi-class Output for Exercise Identification
    ex_out = Dense(len(exercise_encoder.classes_),
                   activation='softmax', name='exercise_output')(x)

    model = Model(inputs=inputs, outputs=[tech_out, ex_out])

    # Compiling with metrics for EACH output to avoid the ValueError
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

    print("\n🚀 Step 3: Training on Sequences (20 Epochs)...")
    # Training the master model
    model.fit(
        X_train,
        {'technique_output': y_tech_train, 'exercise_output': y_ex_train},
        validation_data=(
            X_test, {'technique_output': y_tech_test, 'exercise_output': y_ex_test}),
        epochs=20,
        batch_size=64
    )

    # Step 4: Save model and helper files for the app
    model.save("powerlifting_sequence_model.keras")
    joblib.dump(scaler, "master_scaler.pkl")
    joblib.dump(exercise_encoder, "exercise_encoder.pkl")

    print("\n🏆 Training finished successfully! Files saved.")


if __name__ == "__main__":
    train_unified_model()
