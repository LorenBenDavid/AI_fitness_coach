# ExerciseConfig.py - Full Configuration with Sumo Deadlift

EXERCISE_CONFIG = {
    # --- SQUAT ---
    "Squat": {
        "Left Side": {"joints": [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31], "label": "Squat_Side"},
        "Right Side": {"joints": [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32], "label": "Squat_Side"},
        "Front": {"joints": [11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], "label": "Squat_Front"}
    },

    # --- PRESSING GROUP ---
    "Bench": {
        "Left Side": {"joints": [11, 13, 15, 17, 19, 21, 23, 25], "label": "Bench_Side"},
        "Right Side": {"joints": [12, 14, 16, 18, 20, 22, 24, 26], "label": "Bench_Side"},
        "Front": {"joints": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], "label": "Bench_Front"}
    },


    # --- DEADLIFT ---
    "Conventional Deadlift": {
        "Left Side": {"joints": [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31], "label": "Deadlift_Side"},
        "Right Side": {"joints": [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32], "label": "Deadlift_Side"},
        "Front": {"joints": [11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], "label": "Deadlift_Front"}
    },
    "Sumo Deadlift": {
        "Left Side": {"joints": [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31], "label": "Sumo_Side"},
        "Right Side": {"joints": [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32], "label": "Sumo_Side"},
        "Front": {"joints": [11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], "label": "Sumo_Front"}
    },


}
