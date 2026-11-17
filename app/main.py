
import io
import json
import math
import os
import secrets
import threading

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from PIL import (
    Image,
    ImageOps,
    UnidentifiedImageError,
)

from fastapi import (
    Depends,
    FastAPI,
    UploadFile,
    File,
    Form,
    Header,
    HTTPException,
    Request,
)

from ultralytics import YOLO

from config import (
    YOLO_DETECTION_MODEL_PATH,
    RF_MODEL_PATH,
    RF_METADATA_PATH,

    FEATURE_COLUMNS,
    TARGET_COLUMNS,

    GOOD_CLASS_KEYWORDS,

    YOLO_DETECTION_IMAGE_SIZE,
    YOLO_DETECTION_CONFIDENCE,
    YOLO_DETECTION_IOU,
    YOLO_MIN_CANDIDATE_CONFIDENCE,
    DETECTION_CONFIDENCE_THRESHOLD,

    YOLO_DEVICE,

    MAX_IMAGES,
    MAX_IMAGE_SIZE_MB,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,

    MIN_CONFIDENT_COVERAGE,

    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    API_KEY_ENV_NAME,
)


# ============================================================
# KONFIGURASI
# ============================================================

API_KEY = os.getenv(API_KEY_ENV_NAME, "")

INFERENCE_LOCK = threading.Lock()

# Membatasi ukuran gambar yang dapat dibuka Pillow.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


# ============================================================
# KEAMANAN API
# ============================================================

def verify_api_key(
    x_api_key: str = Header(default="")
):
    """
    Memastikan request Laravel memiliki API key yang benar.
    """

    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API key server belum dikonfigurasi.",
        )

    if not secrets.compare_digest(
        x_api_key,
        API_KEY,
    ):
        raise HTTPException(
            status_code=401,
            detail="API key tidak valid.",
        )


# ============================================================
# FUNGSI KELAS YOLO
# ============================================================

def get_model_class_names(model):
    """
    Mengambil nama kelas dari model YOLO.
    """

    names = model.names

    if isinstance(names, dict):

        return {
            int(index): str(name)
            for index, name in names.items()
        }

    return {
        index: str(name)
        for index, name in enumerate(names)
    }


def is_good_class(class_name: str) -> bool:
    """
    Menentukan apakah nama kelas termasuk Good.

    Contoh:
    Good              -> Good
    Intact Soybeans   -> Good
    Broken Soybeans   -> Bad
    Bad               -> Bad
    """

    normalized_name = class_name.casefold()

    for keyword in GOOD_CLASS_KEYWORDS:

        normalized_keyword = keyword.casefold()

        if normalized_keyword in normalized_name:
            return True

    return False


# ============================================================
# VALIDASI MODEL RANDOM FOREST
# ============================================================

def validate_rf_model(model):
    """
    Memastikan fitur model Random Forest sama
    dengan fitur yang dikirim oleh API.
    """

    trained_features = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if trained_features is not None:

        trained_features = list(trained_features)

        if trained_features != FEATURE_COLUMNS:

            raise RuntimeError(
                "Fitur Random Forest tidak cocok.\n"
                f"Fitur training: {trained_features}\n"
                f"Fitur API: {FEATURE_COLUMNS}"
            )

    # Memastikan model mendukung dua target output.
    n_outputs = getattr(
        model,
        "n_outputs_",
        None,
    )

    if n_outputs is not None and n_outputs != 2:

        raise RuntimeError(
            "Random Forest harus menghasilkan "
            "dua target prediksi: hasil tahu dan "
            "waktu memasak."
        )


# ============================================================
# VALIDASI METADATA
# ============================================================

def validate_metadata():

    if not RF_METADATA_PATH.exists():
        print(
            "Peringatan: metadata.json tidak ditemukan. "
            "Validasi metadata dilewati."
        )
        return

    with open(
        RF_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    metadata_features = metadata.get("features", [])

    metadata_targets = metadata.get("targets", [])

    if metadata_features != FEATURE_COLUMNS:

        raise RuntimeError(
            "Fitur pada metadata.json tidak sesuai.\n"
            f"Metadata: {metadata_features}\n"
            f"Config: {FEATURE_COLUMNS}"
        )

    if metadata_targets != TARGET_COLUMNS:

        raise RuntimeError(
            "Target pada metadata.json tidak sesuai.\n"
            f"Metadata: {metadata_targets}\n"
            f"Config: {TARGET_COLUMNS}"
        )

    print(
        "Metadata Random Forest berhasil divalidasi."
    )


# ============================================================
# LOAD MODEL
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    if not API_KEY:

        raise RuntimeError(
            f"Environment variable {API_KEY_ENV_NAME} "
            "belum diisi."
        )

    # --------------------------------------------------------
    # PERIKSA FILE MODEL
    # --------------------------------------------------------

    if not YOLO_DETECTION_MODEL_PATH.exists():

        raise RuntimeError(
            "Model YOLO Detection tidak ditemukan: "
            f"{YOLO_DETECTION_MODEL_PATH}"
        )

    if not RF_MODEL_PATH.exists():

        raise RuntimeError(
            "Model Random Forest tidak ditemukan: "
            f"{RF_MODEL_PATH}"
        )

    if not GOOD_CLASS_KEYWORDS:

        raise RuntimeError(
            "GOOD_CLASS_KEYWORDS belum diatur."
        )

    # --------------------------------------------------------
    # LOAD YOLO DETECTION
    # --------------------------------------------------------

    print("\n======================================")
    print("MEMUAT MODEL AI AGROTOFU")
    print("======================================")

    print("\n[1] Memuat YOLO Object Detection...")

    app.state.yolo_model = YOLO(
        str(YOLO_DETECTION_MODEL_PATH)
    )

    yolo_model = app.state.yolo_model

    # Pastikan model adalah Detection.
    if getattr(yolo_model, "task", None) != "detect":

        raise RuntimeError(
            "Model yang dimuat bukan YOLO Object Detection. "
            f"Task model: {getattr(yolo_model, 'task', None)}"
        )

    class_names = get_model_class_names(
        yolo_model
    )

    print("Kelas YOLO:")

    for index, class_name in class_names.items():

        print(f"  {index}: {class_name}")

    good_classes = [
        name
        for name in class_names.values()
        if is_good_class(name)
    ]

    if not good_classes:

        raise RuntimeError(
            "Tidak ditemukan kelas Good pada model YOLO.\n"
            f"Kelas model: {list(class_names.values())}\n"
            f"Keyword Good: {GOOD_CLASS_KEYWORDS}"
        )

    print("Kelas Good:", good_classes)

    # --------------------------------------------------------
    # LOAD RANDOM FOREST
    # --------------------------------------------------------

    print("\n[2] Memuat Random Forest Regression...")

    validate_metadata()

    app.state.rf_model = joblib.load(
        RF_MODEL_PATH
    )

    validate_rf_model(
        app.state.rf_model
    )

    print("Fitur Random Forest:", FEATURE_COLUMNS)

    print("Target Random Forest:", TARGET_COLUMNS)

    # --------------------------------------------------------
    # SELESAI
    # --------------------------------------------------------

    app.state.model_ready = True

    print("\n======================================")
    print("SEMUA MODEL BERHASIL DIMUAT")
    print("YOLO: OBJECT DETECTION")
    print("RF: MULTI-OUTPUT REGRESSION")
    print("API SIAP DIGUNAKAN")
    print("======================================\n")

    yield

    app.state.model_ready = False

    print("API AgroTofu dihentikan.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)


# ============================================================
# ENDPOINT HOME
# ============================================================

@app.get("/")
def home():

    return {
        "application": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "model": {
            "yolo": "YOLO Object Detection",
            "regression": "Random Forest Regression",
        },
        "documentation": "/docs",
    }


# ============================================================
# ENDPOINT HEALTH CHECK
# ============================================================

@app.get("/health")
def health(request: Request):

    yolo_loaded = hasattr(
        request.app.state,
        "yolo_model",
    )

    rf_loaded = hasattr(
        request.app.state,
        "rf_model",
    )

    return {
        "status": (
            "healthy"
            if yolo_loaded and rf_loaded
            else "not_ready"
        ),
        "yolo_detection_loaded": yolo_loaded,
        "random_forest_loaded": rf_loaded,
    }


# ============================================================
# FUNGSI VALIDASI GAMBAR
# ============================================================

def load_uploaded_image(uploaded: UploadFile):

    content = uploaded.file.read(
        MAX_IMAGE_BYTES + 1
    )

    if not content:

        raise HTTPException(
            status_code=400,
            detail=f"File kosong: {uploaded.filename}",
        )

    if len(content) > MAX_IMAGE_BYTES:

        raise HTTPException(
            status_code=413,
            detail=(
                f"Ukuran gambar {uploaded.filename} "
                f"melebihi {MAX_IMAGE_SIZE_MB} MB."
            ),
        )

    try:

        with Image.open(
            io.BytesIO(content)
        ) as source:

            source.verify()

        with Image.open(
            io.BytesIO(content)
        ) as source:

            image = ImageOps.exif_transpose(
                source
            ).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"File bukan gambar valid atau "
                f"resolusinya terlalu besar: {uploaded.filename}"
            ),
        )

    return image


# ============================================================
# ENDPOINT PREDIKSI
# ============================================================

@app.post(
    "/predict",
    dependencies=[
        Depends(verify_api_key)
    ],
)
def predict(

    request: Request,

    berat_kedelai_kg: float = Form(...),

    images: list[UploadFile] = File(...),

):

    # ========================================================
    # 1. AMBIL MODEL
    # ========================================================

    yolo_model = request.app.state.yolo_model

    rf_model = request.app.state.rf_model

    # ========================================================
    # 2. VALIDASI INPUT
    # ========================================================

    if not math.isfinite(berat_kedelai_kg):

        raise HTTPException(
            status_code=400,
            detail="Berat kedelai harus berupa angka valid.",
        )

    if berat_kedelai_kg <= 0:

        raise HTTPException(
            status_code=400,
            detail="Berat kedelai harus lebih dari 0 kg.",
        )

    if not images:

        raise HTTPException(
            status_code=400,
            detail="Upload minimal satu gambar kedelai.",
        )

    if len(images) > MAX_IMAGES:

        raise HTTPException(
            status_code=400,
            detail=f"Maksimal {MAX_IMAGES} gambar.",
        )

    # ========================================================
    # 3. INISIALISASI KELAS
    # ========================================================

    model_classes = get_model_class_names(
        yolo_model
    )

    counts = {
        class_name: 0
        for class_name in model_classes.values()
    }

    good_count = 0

    bad_count = 0

    uncertain_count = 0

    classified_count = 0

    images_with_detection = 0

    image_details = []

    # ========================================================
    # 4. YOLO OBJECT DETECTION
    # ========================================================

    for uploaded in images:

        image = load_uploaded_image(
            uploaded
        )

        # ----------------------------------------------------
        # INFERENCE YOLO
        # ----------------------------------------------------

        try:

            with INFERENCE_LOCK:

                results = yolo_model.predict(

                    source=image,

                    imgsz=YOLO_DETECTION_IMAGE_SIZE,

                    conf=YOLO_MIN_CANDIDATE_CONFIDENCE,

                    iou=YOLO_DETECTION_IOU,

                    device=YOLO_DEVICE,

                    verbose=False,

                )

        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Terjadi kesalahan ketika "
                    f"menjalankan YOLO: {str(error)}"
                ),
            )

        result = results[0]

        if result.boxes is None:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Model YOLO tidak menghasilkan "
                    "data bounding box."
                ),
            )

        # ----------------------------------------------------
        # HITUNG DETEKSI
        # ----------------------------------------------------

        detected_in_image = 0

        accepted_in_image = 0

        image_good = 0

        image_bad = 0

        image_uncertain = 0

        image_class_counts = {
            class_name: 0
            for class_name in model_classes.values()
        }

        for box in result.boxes:

            class_index = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            if class_index not in model_classes:

                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Indeks kelas hasil YOLO "
                        "tidak ditemukan."
                    ),
                )

            class_name = model_classes[
                class_index
            ]

            detected_in_image += 1

            # -----------------------------------------------
            # FILTER CONFIDENCE
            # -----------------------------------------------

            if confidence < DETECTION_CONFIDENCE_THRESHOLD:

                uncertain_count += 1

                image_uncertain += 1

                continue

            # -----------------------------------------------
            # KLASIFIKASI GOOD / BAD
            # -----------------------------------------------

            counts[class_name] += 1

            image_class_counts[class_name] += 1

            classified_count += 1

            accepted_in_image += 1

            if is_good_class(class_name):

                good_count += 1

                image_good += 1

            else:

                bad_count += 1

                image_bad += 1

        if accepted_in_image > 0:

            images_with_detection += 1

        # ----------------------------------------------------
        # DETAIL GAMBAR
        # ----------------------------------------------------

        image_details.append({

            "filename": uploaded.filename,

            "detected_count": detected_in_image,

            "classified_count": accepted_in_image,

            "good_count": image_good,

            "bad_count": image_bad,

            "uncertain_count": image_uncertain,

            "jumlah_per_kelas": image_class_counts,

        })

    # ========================================================
    # 5. VALIDASI HASIL DETECTION
    # ========================================================

    if classified_count == 0:

        raise HTTPException(
            status_code=422,
            detail=(
                "Tidak ditemukan biji kedelai "
                "dengan confidence yang memenuhi "
                "batas minimum."
            ),
        )

    # Cakupan dihitung dari jumlah gambar yang
    # memiliki minimal satu deteksi valid.
    coverage = (
        images_with_detection / len(images)
    )

    coverage_percent = coverage * 100

    if coverage < MIN_CONFIDENT_COVERAGE:

        raise HTTPException(
            status_code=422,
            detail={

                "message": (
                    "Cakupan deteksi terlalu rendah. "
                    "Periksa kualitas gambar kedelai."
                ),

                "total_images": len(images),

                "images_with_detection": images_with_detection,

                "coverage_percent": round(
                    coverage_percent,
                    2,
                ),

                "classified_count": classified_count,

                "uncertain_count": uncertain_count,

            },
        )

    # ========================================================
    # 6. MENGHITUNG GOOD DAN BAD
    # ========================================================

    good_percent = (
        good_count / classified_count
    ) * 100

    bad_percent = (
        bad_count / classified_count
    ) * 100

    total_detected = good_count + bad_count

    # ========================================================
    # 7. RANDOM FOREST REGRESSION
    # ========================================================

    rf_input = pd.DataFrame(

        [[
            berat_kedelai_kg,
            good_percent,
        ]],

        columns=FEATURE_COLUMNS,

    )

    try:

        with INFERENCE_LOCK:

            rf_prediction = rf_model.predict(
                rf_input
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Random Forest gagal melakukan prediksi: "
                f"{str(error)}"
            ),
        )

    prediction = np.asarray(
        rf_prediction
    )

    if prediction.shape != (1, 2):

        raise HTTPException(
            status_code=500,
            detail=(
                "Format output Random Forest tidak sesuai. "
                f"Output shape: {prediction.shape}. "
                "API mengharapkan dua target prediksi."
            ),
        )

    hasil_tahu_kg = float(
        prediction[0][0]
    )

    waktu_memasak_jam = float(
        prediction[0][1]
    )

    # ========================================================
    # 8. VALIDASI PREDIKSI
    # ========================================================

    if not math.isfinite(hasil_tahu_kg):

        raise HTTPException(
            status_code=500,
            detail="Prediksi hasil tahu tidak valid.",
        )

    if not math.isfinite(waktu_memasak_jam):

        raise HTTPException(
            status_code=500,
            detail="Prediksi waktu memasak tidak valid.",
        )

    if hasil_tahu_kg < 0:

        raise HTTPException(
            status_code=500,
            detail="Prediksi hasil tahu bernilai negatif.",
        )

    if waktu_memasak_jam <= 0 or waktu_memasak_jam > 24:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediksi waktu memasak berada di luar "
                "rentang yang diizinkan (lebih dari 0 "
                "hingga 24 jam)."
            ),
        )

    waktu_memasak_menit = (
        waktu_memasak_jam * 60
    )

    # ========================================================
    # 9. RESPONSE API
    # ========================================================

    return {

        "success": True,

        "input": {

            "berat_kedelai_kg": round(
                berat_kedelai_kg,
                2,
            ),

            "jumlah_gambar": len(images),

        },

        # Nama key dipertahankan agar kompatibel
        # dengan Laravel yang sudah dibuat.
        "yolo_classification": {

            "jumlah_per_kelas": counts,

            "good_count": good_count,

            "bad_count": bad_count,

            "uncertain_count": uncertain_count,

            "classified_count": classified_count,

            "total_detected": total_detected,

            "good_percent": round(
                good_percent,
                2,
            ),

            "bad_percent": round(
                bad_percent,
                2,
            ),

            "coverage_percent": round(
                coverage_percent,
                2,
            ),

            "images_with_detection": images_with_detection,

        },

        "random_forest_prediction": {

            "hasil_tahu_kg": round(
                hasil_tahu_kg,
                2,
            ),

            "waktu_memasak_jam": round(
                waktu_memasak_jam,
                2,
            ),

            "waktu_memasak_menit": round(
                waktu_memasak_menit,
                2,
            ),

        },

        "image_details": image_details,

    }
