
import sys
import csv
import json
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
# MEMBACA EXCEL TANPA PANDAS
# ==========================================

def read_excel_without_pandas(file_path):

    workbook = load_workbook(
        file_path,
        read_only=True,
        data_only=True
    )

    worksheet = workbook.active

    rows = worksheet.iter_rows(values_only=True)

    headers = next(rows)

    headers = [
        str(value).strip() if value is not None else ""
        for value in headers
    ]

    data = []

    for row in rows:

        if all(value is None for value in row):
            continue

        record = dict(zip(headers, row))

        data.append(record)

    workbook.close()

    return data


# ==========================================
# FILTER DATA
# ==========================================

def filter_split(data, split_name):

    return [
        row for row in data
        if str(row.get("split", "")).strip().lower()
        == split_name
    ]


# ==========================================
# KONVERSI FITUR DAN TARGET
# ==========================================

def extract_xy(data, feature_columns, target_columns):

    X = []
    y = []

    for index, row in enumerate(data, start=2):

        try:

            feature_values = [
                float(row[column])
                for column in feature_columns
            ]

            target_values = [
                float(row[column])
                for column in target_columns
            ]

        except (TypeError, ValueError, KeyError) as error:

            raise ValueError(
                f"Data tidak valid pada baris Excel {index}: {error}"
            )

        if not all(np.isfinite(feature_values + target_values)):
            raise ValueError(
                f"Ditemukan nilai NaN atau infinity pada baris {index}."
            )

        X.append(feature_values)
        y.append(target_values)

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
        multioutput="raw_values",
    )

    normalized_errors = errors / target_scales

    return float(np.mean(normalized_errors))


# ==========================================
# MENYIMPAN FEATURE IMPORTANCE
# ==========================================

def save_feature_importance(
    feature_names,
    importances,
    output_path
):

    rows = [
        {
            "feature": feature,
            "importance": float(importance)
        }
        for feature, importance
        in zip(feature_names, importances)
    ]

    rows = sorted(
        rows,
        key=lambda item: item["importance"],
        reverse=True
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["feature", "importance"]
        )

        writer.writeheader()
        writer.writerows(rows)

    return rows


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

    # Membaca dataset Excel tanpa Pandas.
    data = read_excel_without_pandas(
        PREPARED_REGRESSION_PATH
    )

    required_columns = (
        FEATURE_COLUMNS
        + TARGET_COLUMNS
        + ["split"]
    )

    if not data:
        raise ValueError("Dataset regresi kosong.")

    missing_columns = [
        column
        for column in required_columns
        if column not in data[0]
    ]

    if missing_columns:

        raise ValueError(
            f"Kolom tidak ditemukan: {missing_columns}"
        )

    # Membagi training dan validation.
    train_data = filter_split(data, "train")

    val_data = filter_split(data, "validation")

    if not train_data or not val_data:

        raise ValueError(
            "Data train dan validation harus terisi. "
            f"Train: {len(train_data)}, "
            f"Validation: {len(val_data)}"
        )

    print("===================================")
    print("DATASET RANDOM FOREST")
    print("===================================")

    print("Jumlah seluruh data :", len(data))
    print("Data training       :", len(train_data))
    print("Data validation    :", len(val_data))
    print("Fitur               :", FEATURE_COLUMNS)
    print("Target              :", TARGET_COLUMNS)

    # Memisahkan fitur dan target.
    X_train, y_train = extract_xy(
        train_data,
        FEATURE_COLUMNS,
        TARGET_COLUMNS
    )

    X_val, y_val = extract_xy(
        val_data,
        FEATURE_COLUMNS,
        TARGET_COLUMNS
    )

    # Skala target dari data training.
    target_scales = (
        np.max(y_train, axis=0)
        - np.min(y_train, axis=0)
    )

    target_scales[target_scales == 0] = 1.0

    # Kandidat hyperparameter.
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
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        val_predictions = model.predict(X_val)

        score = normalized_mae(
            y_val,
            val_predictions,
            target_scales,
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
    # TRAINING AKHIR
    # ======================================

    train_val_data = train_data + val_data

    X_train_val, y_train_val = extract_xy(
        train_val_data,
        FEATURE_COLUMNS,
        TARGET_COLUMNS
    )

    final_model = RandomForestRegressor(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    final_model.fit(
        X_train_val,
        y_train_val
    )

    # ======================================
    # SIMPAN MODEL
    # ======================================

    RF_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RF_RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

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
        FEATURE_COLUMNS,
        final_model.feature_importances_,
        RF_RESULT_DIR / "feature_importance.csv"
    )

    # ======================================
    # HASIL
    # ======================================

    print("\n===================================")
    print("TRAINING RANDOM FOREST SELESAI")
    print("===================================")

    print("Model disimpan:", model_path)

    print("\nFeature Importance:")

    for item in importance:

        print(
            f"{item['feature']}: "
            f"{item['importance']:.6f}"
        )


if __name__ == "__main__":
    main()
