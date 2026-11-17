
from pathlib import Path
import os
from dotenv import load_dotenv

# ============================================================
# ROOT PROJECT
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

# Membaca API key dari .env untuk penggunaan lokal.
# Pada Hugging Face, environment variable dari Secrets
# akan tetap dapat dibaca melalui os.getenv().
load_dotenv(ROOT_DIR / ".env")


# ============================================================
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42
YOLO_DETECTION_RANDOM_STATE = 42


# ============================================================
# DATASET CLASSIFICATION (LAMA)
# ============================================================

RAW_DATASET_DIR = ROOT_DIR / "dataset" / "Soybean Seeds"

PROCESSED_DATASET_DIR = (
    ROOT_DIR / "dataset" / "processed"
)


# ============================================================
# DATASET OBJECT DETECTION
# ============================================================

ROBOFLOW_DATASET_DIR = (
    ROOT_DIR / "dataset" / "soybean_quantification"
)

ROBOFLOW_ORIGINAL_YAML = (
    ROBOFLOW_DATASET_DIR / "data.yaml"
)

YOLO_DETECTION_DATA_YAML = (
    ROBOFLOW_DATASET_DIR / "data_agrotofu.yaml"
)


# ============================================================
# RANDOM FOREST DATASET
# ============================================================

REGRESSION_DATA_DIR = ROOT_DIR / "data_regresi"

REGRESSION_DATA_PATH = (
    REGRESSION_DATA_DIR / "dataset_regresi.xlsx"
)

PREPARED_REGRESSION_PATH = (
    REGRESSION_DATA_DIR / "dataset_regresi_prepared.xlsx"
)


# ============================================================
# MODEL DIRECTORIES
# ============================================================

YOLO_MODEL_DIR = ROOT_DIR / "models" / "yolo"

YOLO_DETECTION_MODEL_DIR = YOLO_MODEL_DIR

RF_MODEL_DIR = ROOT_DIR / "models" / "random_forest"


# ============================================================
# MODEL FILES
# ============================================================

# Model YOLO Object Detection yang sudah dilatih.
YOLO_DETECTION_MODEL_PATH = (
    YOLO_MODEL_DIR / "soybean_yolo_detection_best.pt"
)

# Model YOLO Classification lama, tidak digunakan oleh API.
YOLO_CLASSIFICATION_MODEL_PATH = (
    YOLO_MODEL_DIR / "soybean_yolo_classification_best.pt"
)

# Model Random Forest Regression.
RF_MODEL_PATH = (
    RF_MODEL_DIR / "soybean_random_forest.joblib"
)

# Metadata Random Forest.
RF_METADATA_PATH = (
    RF_MODEL_DIR / "metadata.json"
)


# ============================================================
# RESULT DIRECTORIES
# ============================================================

YOLO_RESULT_DIR = ROOT_DIR / "results" / "yolo"

YOLO_DETECTION_RESULT_DIR = (
    YOLO_RESULT_DIR / "detection"
)

YOLO_CLASSIFICATION_RESULT_DIR = (
    YOLO_RESULT_DIR / "classification"
)

RF_RESULT_DIR = ROOT_DIR / "results" / "random_forest"


# ============================================================
# YOLO CLASSIFICATION (LAMA)
# ============================================================

YOLO_BASE_MODEL = "yolo26n-cls.pt"

IMAGE_SIZE = 224

YOLO_EPOCHS = 20

YOLO_BATCH_SIZE = 8

YOLO_PATIENCE = 15


# ============================================================
# YOLO OBJECT DETECTION
# ============================================================

YOLO_DETECTION_BASE_MODEL = "yolo11n.pt"

YOLO_DETECTION_IMAGE_SIZE = 640

YOLO_DETECTION_EPOCHS = 20

YOLO_DETECTION_BATCH_SIZE = 4

YOLO_DETECTION_PATIENCE = 25

YOLO_DETECTION_WORKERS = 0

YOLO_DETECTION_CONFIDENCE = 0.25

YOLO_DETECTION_IOU = 0.70

# Confidence minimum saat mengambil kandidat deteksi.
# Hasil akhir tetap disaring menggunakan threshold di bawah.
YOLO_MIN_CANDIDATE_CONFIDENCE = 0.01

# Confidence minimum agar satu biji dihitung.
DETECTION_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# GOOD AND BAD CLASS CONFIGURATION
# ============================================================

GOOD_CLASS_KEYWORDS = (
    "intact",
    "good",
    "normal",
    "healthy",
)

# Jika nama kelas model adalah Good dan Bad,
# keyword "good" akan mengenali kelas Good.
#
# Jika nama kelas model adalah Intact Soybeans,
# keyword "intact" akan mengenali kelas tersebut.
#
# Semua kelas yang tidak dikenali sebagai Good
# akan dihitung sebagai Bad.


# ============================================================
# DETECTION VALIDATION
# ============================================================

MIN_CONFIDENT_COVERAGE = 0.70

MAX_IMAGES = 300

MAX_IMAGE_SIZE_MB = 10

MAX_IMAGE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

MAX_IMAGE_PIXELS = 40_000_000


# ============================================================
# GPU CONFIGURATION
# ============================================================

# 0 = GPU NVIDIA pertama.
# "cpu" = menggunakan CPU.
YOLO_DEVICE = "cpu"

YOLO_AMP = True

YOLO_WORKERS = 0


# ============================================================
# RANDOM FOREST REGRESSION
# ============================================================

FEATURE_COLUMNS = [
    "berat_kedelai_kg",
    "good_persen",
]

TARGET_COLUMNS = [
    "hasil_tahu_aktual_kg",
    "waktu_memasak_aktual_jam",
]


# ============================================================
# RANDOM FOREST TRAINING
# ============================================================

RF_TEST_SIZE = 0.20

RF_N_ESTIMATORS = 300

RF_MAX_DEPTH = 12

RF_MIN_SAMPLES_SPLIT = 2

RF_MIN_SAMPLES_LEAF = 2


# ============================================================
# API CONFIGURATION
# ============================================================

API_TITLE = "AgroTofu AI API"

API_VERSION = "1.0.0"

API_DESCRIPTION = (
    "API untuk mendeteksi kualitas biji kedelai "
    "menggunakan YOLO Object Detection dan memprediksi "
    "hasil tahu serta durasi memasak menggunakan "
    "Random Forest Regression."
)

API_HOST = "127.0.0.1"

API_PORT = 8001


# ============================================================
# API SECURITY
# ============================================================

# Nama environment variable API key.
API_KEY_ENV_NAME = "SOYBEAN_AI_KEY"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

DIRECTORIES = [
    PROCESSED_DATASET_DIR,
    ROBOFLOW_DATASET_DIR,
    REGRESSION_DATA_DIR,
    YOLO_MODEL_DIR,
    RF_MODEL_DIR,
    YOLO_RESULT_DIR,
    YOLO_DETECTION_RESULT_DIR,
    YOLO_CLASSIFICATION_RESULT_DIR,
    RF_RESULT_DIR,
]

for folder in DIRECTORIES:
    folder.mkdir(parents=True, exist_ok=True)
