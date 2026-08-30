"""Shared constants. Change WINDOW_S/STRIDE_S only before ingesting — every
derived artifact (npz, index, encoder) assumes the same windowing."""

WINDOW_S = 4.0          # seconds of context per searchable moment
STRIDE_S = 2.0          # hop between window centers
IMU_LEN = 200           # samples per window after resampling (=> 50 Hz effective)
IMU_CHANNELS = 6        # ax ay az gx gy gz (gyro zero-filled when absent)
EMB_DIM = 512           # video/text/IMU embedding dimensionality
FRAME_WIDTH = 256       # width of frames fed to the video embedder
ASSUMED_IMU_RATE = 100.0  # fallback when the dataset exposes no IMU timestamps

DERIVED_DIR = "work/derived"
INDEX_DIR = "work/index"
MODELS_DIR = "work/models"
