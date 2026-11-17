
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

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from openpyxl import load_workbook, Workbook

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from config import (
    PREPARED_REGRESSION_PATH,
    RF_MODEL_DIR,
    RF_RESULT_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
)


# ==========================================
# MEMBACA DATASET EXCEL
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
# KONVERSI DATA FITUR DAN TARGET
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
                f"Data tidak valid pada baris {row_number}: {error}"
            )

        if not all(np.isfinite(features + targets)):

            raise ValueError(
                f"Nilai tidak valid ditemukan pada baris {row_number}."
            )

        X.append(features)
        y.append(targets)

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=float)
    )


# ==========================================
# MENGHITUNG METRIK
# ==========================================

def calculate_metrics(actual, predicted):

    return {
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),

        "RMSE": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),

        "R2": float(
            r2_score(actual, predicted)
        ),
    }


# ==========================================
# MENYIMPAN HASIL KE CSV
# ==========================================

def save_csv(output_path, headers, rows):

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=headers
        )

        writer.writeheader()
        writer.writerows(rows)


# ==========================================
# MENYIMPAN HASIL KE EXCEL
# ==========================================

def save_excel(output_path, headers, rows):

    workbook = Workbook()

    worksheet = workbook.active

    worksheet.title = "Test Predictions"

    worksheet.append(headers)

    for row in rows:

        worksheet.append([
            row.get(header)
            for header in headers
        ])

    workbook.save(output_path)

    workbook.close()


# ==========================================
# MAIN
# ==========================================

def main():

    # Memastikan direktori hasil tersedia.
    RF_RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Lokasi model.
    model_path = (
        RF_MODEL_DIR
        / "soybean_random_forest.joblib"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "Model Random Forest belum dilatih. "
            "Jalankan 05_train_random_forest.py terlebih dahulu."
        )

    if not PREPARED_REGRESSION_PATH.exists():

        raise FileNotFoundError(
            "Dataset regresi tidak ditemukan."
        )

    # ======================================
    # MEMBACA DATASET
    # ======================================

    data = read_excel_data(
        PREPARED_REGRESSION_PATH
    )

    if not data:

        raise ValueError(
            "Dataset evaluasi kosong."
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
            f"Kolom tidak ditemukan: {missing_columns}"
        )

    # ======================================
    # MEMBAGI DATASET
    # ======================================

    train_val_data = []

    test_data = []

    for row in data:

        split = str(
            row.get("split", "")
        ).strip().lower()

        if split in ["train", "validation"]:

            train_val_data.append(row)

        elif split == "test":

            test_data.append(row)

    if not train_val_data:

        raise ValueError(
            "Data train dan validation tidak ditemukan."
        )

    if not test_data:

        raise ValueError(
            "Data test tidak ditemukan."
        )

    print("===================================")
    print("DATASET EVALUASI RANDOM FOREST")
    print("===================================")

    print(
        "Data train + validation:",
        len(train_val_data)
    )

    print(
        "Data test:",
        len(test_data)
    )

    # ======================================
    # MEMUAT MODEL
    # ======================================

    model = joblib.load(model_path)

    # Memastikan fitur model sesuai.
    if hasattr(model, "n_features_in_"):

        if model.n_features_in_ != len(FEATURE_COLUMNS):

            raise ValueError(
                "Jumlah fitur model tidak sesuai dengan config.py."
            )

    # ======================================
    # PREDIKSI DATA TEST
    # ======================================

    X_test, y_test = extract_xy(test_data)

    predictions = model.predict(X_test)

    if predictions.ndim == 1:

        predictions = predictions.reshape(-1, 1)

    if predictions.shape[1] != len(TARGET_COLUMNS):

        raise ValueError(
            "Jumlah output model tidak sesuai dengan TARGET_COLUMNS."
        )

    metrics_result = {}

    output_rows = []

    output_headers = list(data[0].keys())

    for target in TARGET_COLUMNS:

        prediction_column = f"{target}_prediksi"

        if prediction_column not in output_headers:

            output_headers.append(prediction_column)

    # Membuat salinan hasil test.
    for index, row in enumerate(test_data):

        output_row = dict(row)

        for target_index, target in enumerate(TARGET_COLUMNS):

            output_row[f"{target}_prediksi"] = float(
                predictions[index, target_index]
            )

        output_rows.append(output_row)

    # ======================================
    # EVALUASI SETIAP TARGET
    # ======================================

    X_train_val, y_train_val = extract_xy(
        train_val_data
    )

    for index, target in enumerate(TARGET_COLUMNS):

        actual = y_test[:, index]

        predicted = predictions[:, index]

        metrics = calculate_metrics(
            actual,
            predicted
        )

        # Baseline menggunakan rata-rata train + validation.
        baseline_value = float(
            np.mean(y_train_val[:, index])
        )

        baseline_prediction = np.full(
            len(actual),
            baseline_value
        )

        baseline_metrics = calculate_metrics(
            actual,
            baseline_prediction
        )

        metrics_result[target] = metrics

        metrics_result[target]["baseline"] = (
            baseline_metrics
        )

        print(f"\nTarget: {target}")

        print("Model:")

        print("MAE :", metrics["MAE"])
        print("RMSE:", metrics["RMSE"])
        print("R2  :", metrics["R2"])

        print("Baseline mean:")

        print("MAE :", baseline_metrics["MAE"])
        print("RMSE:", baseline_metrics["RMSE"])
        print("R2  :", baseline_metrics["R2"])

        # ==================================
        # GRAFIK ACTUAL VS PREDICTED
        # ==================================

        plt.figure(figsize=(7, 6))

        plt.scatter(
            actual,
            predicted,
            alpha=0.6
        )

        low = min(
            actual.min(),
            predicted.min()
        )

        high = max(
            actual.max(),
            predicted.max()
        )

        plt.plot(
            [low, high],
            [low, high],
            linestyle="--"
        )

        plt.xlabel("Actual")

        plt.ylabel("Predicted")

        plt.title(
            f"Actual vs Predicted: {target}"
        )

        plt.tight_layout()

        plt.savefig(
            RF_RESULT_DIR
            / f"actual_vs_predicted_{target}.png",
            dpi=300
        )

        plt.close()

    # ======================================
    # SIMPAN METRIK JSON
    # ======================================

    with open(
        RF_RESULT_DIR / "evaluation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics_result,
            file,
            indent=4
        )

    # ======================================
    # SIMPAN HASIL PREDIKSI CSV
    # ======================================

    save_csv(
        RF_RESULT_DIR / "test_predictions.csv",
        output_headers,
        output_rows
    )

    # ======================================
    # SIMPAN HASIL PREDIKSI EXCEL
    # ======================================

    save_excel(
        RF_RESULT_DIR / "test_predictions.xlsx",
        output_headers,
        output_rows
    )

    # ======================================
    # HASIL AKHIR
    # ======================================

    print("\n===================================")
    print("EVALUASI RANDOM FOREST SELESAI")
    print("===================================")

    print(
        "Hasil disimpan di:",
        RF_RESULT_DIR
    )


if __name__ == "__main__":
    main()
