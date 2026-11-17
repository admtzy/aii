
import io
import math
from contextlib import asynccontextmanager

import joblib
import pandas as pd

from PIL import Image, UnidentifiedImageError
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
)
from ultralytics import YOLO

from config import (
    YOLO_MODEL_DIR,
    RF_MODEL_DIR,
    FEATURE_COLUMNS,
    GOOD_CLASS_KEYWORDS,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
)


# ==========================================
# KONFIGURASI API
# ==========================================

MAX_IMAGES = 300
MAX_IMAGE_SIZE_MB = 10
MIN_CONFIDENT_COVERAGE = 0.70
MAX_IMAGE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024


# ==========================================
# HELPER FUNCTION
# ==========================================

def get_model_class_names(model):
    names = model.names

    if isinstance(names, dict):
        return names

    return dict(enumerate(names))


def class_name_from_model(model, class_index):
    names = get_model_class_names(model)
    return str(names[int(class_index)])


def is_good_class(class_name):
    name = class_name.lower()

    return any(
        keyword.lower() in name
        for keyword in GOOD_CLASS_KEYWORDS
    )


# ==========================================
# LOAD MODEL
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    yolo_path = YOLO_MODEL_DIR / "soybean_yolo_best.pt"
    rf_path = RF_MODEL_DIR / "soybean_random_forest.joblib"

    if not yolo_path.exists():
        raise RuntimeError(
            f"Model YOLO tidak ditemukan: {yolo_path}"
        )

    if not rf_path.exists():
        raise RuntimeError(
            f"Model Random Forest tidak ditemukan: {rf_path}"
        )

    if not GOOD_CLASS_KEYWORDS:
        raise RuntimeError(
            "GOOD_CLASS_KEYWORDS belum diatur di config.py"
        )

    print("Memuat model YOLO...")
    app.state.yolo_model = YOLO(str(yolo_path))

    print("Memuat model Random Forest...")
    app.state.rf_model = joblib.load(rf_path)

    class_names = get_model_class_names(
        app.state.yolo_model
    )

    good_classes = [
        name
        for name in class_names.values()
        if is_good_class(str(name))
    ]

    if not good_classes:
        raise RuntimeError(
            "Kelas Good tidak ditemukan pada model YOLO. "
            "Periksa GOOD_CLASS_KEYWORDS di config.py."
        )

    rf_features = list(
        app.state.rf_model.feature_names_in_
    )

    if rf_features != FEATURE_COLUMNS:
        raise RuntimeError(
            f"Fitur RF tidak sesuai. "
            f"Model: {rf_features}, "
            f"Config: {FEATURE_COLUMNS}"
        )

    print("Model berhasil dimuat.")
    print("Kelas Good:", good_classes)

    yield

    print("API dihentikan.")


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Soybean Quality and Tofu Prediction API",
    description=(
        "API klasifikasi kualitas kedelai menggunakan YOLO "
        "dan prediksi hasil tahu menggunakan Random Forest."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ==========================================
# ENDPOINT BERANDA
# ==========================================

@app.get("/")
def home():
    return {
        "application": "Soybean Quality and Tofu Prediction API",
        "status": "running",
        "documentation": "/docs",
    }


# ==========================================
# ENDPOINT HEALTH CHECK
# ==========================================

@app.get("/health")
def health(request: Request):

    return {
        "healthy": True,
        "yolo_loaded": hasattr(
            request.app.state, "yolo_model"
        ),
        "random_forest_loaded": hasattr(
            request.app.state, "rf_model"
        ),
    }


# ==========================================
# ENDPOINT PREDIKSI
# ==========================================

@app.post("/predict")
async def predict(
    request: Request,
    berat_kedelai_kg: float = Form(...),
    images: list[UploadFile] = File(...),
):

    # --------------------------------------
    # VALIDASI BERAT KEDELAI
    # --------------------------------------

    if (
        not math.isfinite(berat_kedelai_kg)
        or berat_kedelai_kg <= 0
    ):
        raise HTTPException(
            status_code=400,
            detail="Berat kedelai harus lebih dari 0 kg."
        )

    # --------------------------------------
    # VALIDASI JUMLAH GAMBAR
    # --------------------------------------

    if not images:
        raise HTTPException(
            status_code=400,
            detail="Minimal satu gambar harus diunggah."
        )

    if len(images) > MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Maksimal {MAX_IMAGES} gambar."
        )

    yolo_model = request.app.state.yolo_model
    rf_model = request.app.state.rf_model

    counts = {
        str(name): 0
        for name in get_model_class_names(
            yolo_model
        ).values()
    }

    uncertain_count = 0
    image_details = []
    total_uploaded = len(images)

    # --------------------------------------
    # PROSES GAMBAR DENGAN YOLO
    # --------------------------------------

    for uploaded_file in images:

        content = await uploaded_file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail=f"File {uploaded_file.filename} kosong."
            )

        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"File {uploaded_file.filename} "
                    f"melebihi batas 10 MB."
                )
            )

        try:
            image = Image.open(
                io.BytesIO(content)
            ).convert("RGB")

        except (
            UnidentifiedImageError,
            OSError,
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"File {uploaded_file.filename} "
                    "bukan gambar yang valid."
                )
            )

        result = yolo_model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )[0]

        class_index = int(
            result.probs.top1
        )

        confidence = float(
            result.probs.top1conf
        )

        class_name = class_name_from_model(
            yolo_model,
            class_index,
        )

        if confidence >= CONFIDENCE_THRESHOLD:

            counts[class_name] += 1
            status = "classified"

        else:

            uncertain_count += 1
            status = "uncertain"

        image_details.append({
            "filename": uploaded_file.filename,
            "predicted_class": class_name,
            "confidence": round(confidence, 4),
            "status": status,
        })

    # --------------------------------------
    # HITUNG PERSENTASE KUALITAS
    # --------------------------------------

    classified_count = sum(counts.values())

    if classified_count == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "Tidak ada gambar dengan "
                "confidence yang memenuhi ambang batas."
            )
        )

    coverage = (
        classified_count / total_uploaded
    )

    if coverage < MIN_CONFIDENT_COVERAGE:
        raise HTTPException(
            status_code=422,
            detail=(
                "Coverage klasifikasi terlalu rendah. "
                "Gunakan gambar yang lebih jelas."
            )
        )

    good_count = sum(
        count
        for name, count in counts.items()
        if is_good_class(name)
    )

    bad_count = classified_count - good_count

    good_percent = (
        good_count / classified_count
    ) * 100

    bad_percent = (
        bad_count / classified_count
    ) * 100

    coverage_percent = coverage * 100

    # --------------------------------------
    # PREDIKSI RANDOM FOREST
    # --------------------------------------

    rf_input = pd.DataFrame(
        [[berat_kedelai_kg, good_percent]],
        columns=FEATURE_COLUMNS,
    )

    prediction = rf_model.predict(rf_input)[0]

    hasil_tahu = float(prediction[0])
    waktu_memasak = float(prediction[1])

    # --------------------------------------
    # RESPONSE API
    # --------------------------------------

    return {
        "success": True,

        "input": {
            "berat_kedelai_kg": berat_kedelai_kg,
            "jumlah_gambar": total_uploaded,
        },

        "yolo_classification": {
            "jumlah_per_kelas": counts,
            "good_count": good_count,
            "bad_count": bad_count,
            "uncertain_count": uncertain_count,
            "classified_count": classified_count,
            "good_percent": round(good_percent, 2),
            "bad_percent": round(bad_percent, 2),
            "coverage_percent": round(
                coverage_percent, 2
            ),
        },

        "random_forest_prediction": {
            "hasil_tahu_kg": round(
                hasil_tahu, 2
            ),
            "waktu_memasak_jam": round(
                waktu_memasak, 2
            ),
            "waktu_memasak_menit": round(
                waktu_memasak * 60, 2
            ),
        },

        "image_details": image_details,
    }
