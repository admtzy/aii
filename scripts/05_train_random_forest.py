
import sys
import json
import csv
from pathlib import Path

# ==========================================
# ROOT PROJECT
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ==========================================
# LIBRARY
# ==========================================

import joblib
import numpy as np

from openpyxl import load_workbook

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from config import (
    PREPARED_REGRESSION_PATH,
    RF_MODEL_DIR,
    RF_RESULT_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    RANDOM_STATE,
)


# ==========================================
# MEMBACA DATASET EXCEL TANPA PANDAS
# ==========================================

def read_excel_data(file_path):

    workbook = load_workbook(
        file_path,
        read_only=True,
        data_only=True
    )

    worksheet = workbook.active

    rows = worksheet.iter_rows(values_only=True)

    headers = next(rows, None)

    if headers is None:
        workbook.close()
        raise ValueError("File Excel tidak memiliki header.")

    headers = [
        str(value).strip() if value is not None else ""
        for value in headers
    ]

    data = []

    for row in rows:

        if all(value is None for value in row):
            continue

        data.append(dict(zip(headers, row)))

    workbook.close()

    return data


# ==========================================
# MEMBAGI DATA TRAIN DAN VALIDATION
# ==========================================

def split_dataset(data):

    train_data = []
    val_data = []

    for row in data:

        split = str(
            row.get("split", "")
        ).strip().lower()

        if split == "train":

            train_data.append(row)

        elif split == "validation":

            val_data.append(row)

    return train_data, val_data


# ==========================================
# KONVERSI FITUR DAN TARGET
# ==========================================

def extract_xy(data):

    X = []
    y = []

    for row_number, row in enumerate(data, start=2):

        try:

            features = [
                float(row[column])
                for column in FEATURE_COLUMNS
            ]

            targets = [
                float(row[column])
                for column in TARGET_COLUMNS
            ]

        except (TypeError, ValueError, KeyError) as error:

            raise ValueError(
                f"Data tidak valid pada baris Excel "
                f"{row_number}: {error}"
            )

        if not all(np.isfinite(features + targets)):

            raise ValueError(
                f"Nilai NaN atau infinity ditemukan "
                f"pada baris Excel {row_number}."
            )

        X.append(features)
        y.append(targets)

    if not X:

        raise ValueError(
            "Tidak ada data yang dapat digunakan."
        )

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=float)
    )


# ==========================================
# NORMALIZED MAE
# ==========================================

def normalized_mae(y_true, y_pred, target_scales):

    errors = mean_absolute_error(
        y_true,
        y_pred,
        multioutput="raw_values"
    )

    normalized_errors = errors / target_scales

    return float(np.mean(normalized_errors))


# ==========================================
# SIMPAN FEATURE IMPORTANCE
# ==========================================

def save_feature_importance(model):

    importance = sorted(
        zip(
            FEATURE_COLUMNS,
            model.feature_importances_
        ),
        key=lambda item: item[1],
        reverse=True
    )

    output_path = (
        RF_RESULT_DIR / "feature_importance.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "feature",
            "importance"
        ])

        for feature, value in importance:

            writer.writerow([
                feature,
                float(value)
            ])

    return importance


# ==========================================
# MAIN
# ==========================================

def main():

    # Memastikan dataset tersedia.
    if not PREPARED_REGRESSION_PATH.exists():

        raise FileNotFoundError(
            "Dataset regresi belum dipersiapkan. "
            "Jalankan 04_prepare_rf_dataset.py."
        )

    # Membaca Excel.
    data = read_excel_data(
        PREPARED_REGRESSION_PATH
    )

    if not data:

        raise ValueError(
            "Dataset regresi kosong."
        )

    # Memeriksa kolom.
    required_columns = (
        FEATURE_COLUMNS
        + TARGET_COLUMNS
        + ["split"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data[0]
    ]

    if missing_columns:

        raise ValueError(
            f"Kolom berikut tidak ditemukan: "
            f"{missing_columns}"
        )

    # Membagi training dan validation.
    train_data, val_data = split_dataset(data)

    if not train_data or not val_data:

        split_counts = {}

        for row in data:

            split = str(
                row.get("split", "kosong")
            ).strip().lower()

            split_counts[split] = (
                split_counts.get(split, 0) + 1
            )

        raise ValueError(
            "Data train dan validation harus terisi. "
            f"Train: {len(train_data)}, "
            f"Validation: {len(val_data)}. "
            f"Pembagian data: {split_counts}"
        )

    print("===================================")
    print("DATASET RANDOM FOREST")
    print("===================================")

    print("Jumlah seluruh data :", len(data))
    print("Data training       :", len(train_data))
    print("Data validation     :", len(val_data))
    print("Fitur               :", FEATURE_COLUMNS)
    print("Target              :", TARGET_COLUMNS)

    # Memisahkan fitur dan target.
    X_train, y_train = extract_xy(train_data)

    X_val, y_val = extract_xy(val_data)

    # Skala target dari data training.
    target_scales = (
        np.max(y_train, axis=0)
        - np.min(y_train, axis=0)
    )

    target_scales[target_scales == 0] = 1.0

    # ======================================
    # KANDIDAT HYPERPARAMETER
    # ======================================

    candidates = [

        {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        },

        {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "max_features": 1.0,
        },

        {
            "n_estimators": 500,
            "max_depth": 20,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": 1.0,
        },

        {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 4,
            "min_samples_leaf": 3,
            "max_features": 0.8,
        },
    ]

    best_score = float("inf")
    best_params = None

    print("\nMemulai pencarian hyperparameter...")

    for index, params in enumerate(
        candidates,
        start=1
    ):

        model = RandomForestRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        val_predictions = model.predict(X_val)

        score = normalized_mae(
            y_val,
            val_predictions,
            target_scales
        )

        print(
            f"Kandidat {index}: "
            f"{params} | "
            f"Normalized MAE: {score:.6f}"
        )

        if score < best_score:

            best_score = score
            best_params = params

    if best_params is None:

        raise RuntimeError(
            "Tidak berhasil memilih hyperparameter."
        )

    print("\nHyperparameter terbaik:", best_params)

    print(
        "Validation normalized MAE:",
        best_score
    )

    # ======================================
    # TRAINING AKHIR TRAIN + VALIDATION
    # ======================================

    train_val_data = train_data + val_data

    X_train_val, y_train_val = extract_xy(
        train_val_data
    )

    final_model = RandomForestRegressor(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    final_model.fit(
        X_train_val,
        y_train_val
    )

    # ======================================
    # DIREKTORI PENYIMPANAN
    # ======================================

    RF_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RF_RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================
    # SIMPAN MODEL
    # ======================================

    model_path = (
        RF_MODEL_DIR
        / "soybean_random_forest.joblib"
    )

    joblib.dump(
        final_model,
        model_path
    )

    # ======================================
    # SIMPAN METADATA
    # ======================================

    metadata = {

        "features": FEATURE_COLUMNS,

        "targets": TARGET_COLUMNS,

        "best_params": best_params,

        "validation_normalized_mae": best_score,

        "random_state": RANDOM_STATE,

        "training_rows": len(train_data),

        "validation_rows": len(val_data),

        "final_fit_rows": len(train_val_data),

    }

    with open(
        RF_MODEL_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # ======================================
    # FEATURE IMPORTANCE
    # ======================================

    importance = save_feature_importance(
        final_model
    )

    # ======================================
    # HASIL TRAINING
    # ======================================

    print("\n===================================")
    print("TRAINING RANDOM FOREST SELESAI")
    print("===================================")

    print("Model disimpan:", model_path)

    print("\nFeature Importance:")

    for feature, value in importance:

        print(
            f"{feature}: {value:.6f}"
        )


if __name__ == "__main__":
    main()
