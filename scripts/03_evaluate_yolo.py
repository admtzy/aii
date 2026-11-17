
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

import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml

from ultralytics import YOLO


# ==========================================
# KONFIGURASI AGROTOFU
# ==========================================

from config import (
    YOLO_DETECTION_MODEL_PATH,
    YOLO_DETECTION_RESULT_DIR,
    YOLO_DETECTION_DATA_YAML,
    YOLO_DETECTION_IMAGE_SIZE,
    YOLO_DETECTION_WORKERS,
    YOLO_DETECTION_BATCH_SIZE,
    YOLO_DETECTION_CONFIDENCE,
    YOLO_DETECTION_IOU,
    YOLO_DEVICE,
)


# ==========================================
# KONFIGURASI KELAS
# ==========================================

EXPECTED_CLASS_NAMES = [
    "Good",
    "Bad",
]

NUMBER_OF_CLASSES = 2


# ==========================================
# MEMBACA NAMA KELAS
# ==========================================

def normalize_class_names(names):

    if isinstance(names, dict):
        names = [
            names[key]
            for key in sorted(
                names,
                key=lambda item: int(item)
            )
        ]

    return [
        str(name).strip()
        for name in names
    ]


# ==========================================
# MEMERIKSA MODEL DAN DATASET
# ==========================================

def check_requirements():

    print("\n===================================")
    print("CHECK EVALUATION REQUIREMENTS")
    print("===================================")

    model_path = YOLO_DETECTION_MODEL_PATH
    yaml_path = YOLO_DETECTION_DATA_YAML

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model YOLO tidak ditemukan:\n{model_path}\n"
            "Jalankan training YOLO Good/Bad terlebih dahulu."
        )

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML tidak ditemukan:\n{yaml_path}\n"
            "Jalankan preprocessing terlebih dahulu."
        )

    with open(
        yaml_path,
        "r",
        encoding="utf-8"
    ) as file:

        dataset_config = yaml.safe_load(file)

    if not isinstance(dataset_config, dict):
        raise ValueError(
            "Format dataset YAML tidak valid."
        )

    # Memeriksa nama kelas.
    names = normalize_class_names(
        dataset_config.get("names", [])
    )

    if names != EXPECTED_CLASS_NAMES:
        raise ValueError(
            "Nama kelas dataset tidak sesuai.\n"
            f"Ditemukan: {names}\n"
            f"Seharusnya: {EXPECTED_CLASS_NAMES}"
        )

    if int(dataset_config.get("nc", 0)) != NUMBER_OF_CLASSES:
        raise ValueError(
            "Jumlah kelas dataset harus 2."
        )

    # Memeriksa lokasi dataset.
    dataset_root = Path(
        dataset_config.get(
            "path",
            yaml_path.parent
        )
    )

    if not dataset_root.is_absolute():
        dataset_root = (
            yaml_path.parent / dataset_root
        ).resolve()

    test_images = dataset_root / "test" / "images"
    test_labels = dataset_root / "test" / "labels"

    if not test_images.exists():
        raise FileNotFoundError(
            f"Folder test/images tidak ditemukan:\n{test_images}"
        )

    if not test_labels.exists():
        raise FileNotFoundError(
            f"Folder test/labels tidak ditemukan:\n{test_labels}"
        )

    image_files = [
        path for path in test_images.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png",
            ".bmp", ".webp"
        }
    ]

    if not image_files:
        raise ValueError(
            "Tidak ditemukan gambar pada dataset test."
        )

    # Memeriksa label untuk setiap gambar.
    missing_labels = []

    for image_path in image_files:

        relative_label = (
            image_path.relative_to(test_images)
            .with_suffix(".txt")
        )

        label_path = test_labels / relative_label

        if not label_path.exists():
            missing_labels.append(str(relative_label))

    if missing_labels:
        raise ValueError(
            f"Ditemukan {len(missing_labels)} gambar "
            "tanpa pasangan label."
        )

    print("Model ditemukan :", model_path)
    print("Dataset YAML    :", yaml_path)
    print("Dataset root    :", dataset_root)
    print("Jumlah kelas    :", NUMBER_OF_CLASSES)
    print("Nama kelas      :", names)
    print("Test images     :", len(image_files))
    print("Test labels     :", test_labels)

    return model_path, test_images


# ==========================================
# MEMERIKSA DEVICE
# ==========================================

def check_device():

    print("\n===================================")
    print("DEVICE INFORMATION")
    print("===================================")

    print("PyTorch:", torch.__version__)

    if torch.cuda.is_available():

        print("CUDA: Available")
        print("CUDA Version:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))

        if isinstance(YOLO_DEVICE, int):

            if YOLO_DEVICE >= torch.cuda.device_count():
                raise ValueError(
                    f"GPU index {YOLO_DEVICE} tidak tersedia."
                )

            return YOLO_DEVICE

        return YOLO_DEVICE

    print("CUDA: Not Available")
    print("Evaluasi menggunakan CPU.")

    return "cpu"


# ==========================================
# MEMUAT MODEL
# ==========================================

def load_model(model_path):

    print("\n===================================")
    print("LOADING YOLO GOOD BAD MODEL")
    print("===================================")

    model = YOLO(str(model_path))

    model_names = normalize_class_names(
        model.names
    )

    print("Model berhasil dimuat.")
    print("Model classes:", model_names)

    if model_names != EXPECTED_CLASS_NAMES:
        raise ValueError(
            "Kelas model tidak sesuai dengan dataset.\n"
            f"Model: {model_names}\n"
            f"Dataset: {EXPECTED_CLASS_NAMES}\n"
            "Pastikan menggunakan best.pt hasil training "
            "dua kelas Good/Bad."
        )

    return model


# ==========================================
# EVALUASI DATASET TEST
# ==========================================

def evaluate_model(model, device):

    print("\n===================================")
    print("YOLO GOOD BAD EVALUATION")
    print("===================================")

    metrics = model.val(

        data=str(YOLO_DETECTION_DATA_YAML),

        # Evaluasi menggunakan dataset test.
        split="test",

        imgsz=YOLO_DETECTION_IMAGE_SIZE,
        batch=YOLO_DETECTION_BATCH_SIZE,
        device=device,
        workers=YOLO_DETECTION_WORKERS,

        # Simpan confusion matrix dan grafik.
        plots=True,

        project=str(YOLO_DETECTION_RESULT_DIR),
        name="evaluation",
        exist_ok=True,

        verbose=True,
    )

    return metrics


# ==========================================
# MENGAMBIL METRIK EVALUASI
# ==========================================

def extract_metrics(metrics):

    print("\n===================================")
    print("EVALUATION METRICS")
    print("===================================")

    box_metrics = metrics.box

    # Metrik keseluruhan.
    overall = {
        "precision": float(box_metrics.mp),
        "recall": float(box_metrics.mr),
        "mAP50": float(box_metrics.map50),
        "mAP75": float(box_metrics.map75),
        "mAP50-95": float(box_metrics.map),
    }

    for name, value in overall.items():

        print(
            f"{name:15}: {value:.4f} "
            f"({value * 100:.2f}%)"
        )

    # Metrik masing-masing kelas.
    per_class = []

    precision_values = box_metrics.p
    recall_values = box_metrics.r
    map50_values = box_metrics.ap50
    map_values = box_metrics.ap

    class_indices = box_metrics.ap_class_index

    for position, class_index in enumerate(class_indices):

        class_index = int(class_index)

        if class_index >= len(EXPECTED_CLASS_NAMES):
            continue

        class_name = EXPECTED_CLASS_NAMES[class_index]

        class_result = {
            "class_id": class_index,
            "class_name": class_name,
            "precision": float(precision_values[position]),
            "recall": float(recall_values[position]),
            "mAP50": float(map50_values[position]),
            "mAP50-95": float(map_values[position]),
        }

        per_class.append(class_result)

        print(f"\nKelas: {class_name}")

        for name in [
            "precision",
            "recall",
            "mAP50",
            "mAP50-95",
        ]:

            value = class_result[name]

            print(
                f"{name:15}: {value:.4f} "
                f"({value * 100:.2f}%)"
            )

    return {
        "task": "object_detection",
        "classes": EXPECTED_CLASS_NAMES,
        "overall": overall,
        "per_class": per_class,
    }


# ==========================================
# MENYIMPAN HASIL EVALUASI JSON
# ==========================================

def save_metrics_json(metrics_data, output_dir):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / "detection_metrics.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics_data,
            file,
            indent=4
        )

    print("\nJSON metrics disimpan:", output_path)


# ==========================================
# MENYIMPAN HASIL EVALUASI CSV
# TANPA PANDAS
# ==========================================

def save_metrics_csv(metrics_data, output_dir):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / "detection_metrics.csv"
    )

    fieldnames = [
        "type",
        "class_id",
        "class_name",
        "precision",
        "recall",
        "mAP50",
        "mAP75",
        "mAP50-95",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        # Baris metrik keseluruhan.
        overall = metrics_data["overall"]

        writer.writerow({
            "type": "overall",
            "class_id": "",
            "class_name": "All Classes",
            "precision": overall["precision"],
            "recall": overall["recall"],
            "mAP50": overall["mAP50"],
            "mAP75": overall["mAP75"],
            "mAP50-95": overall["mAP50-95"],
        })

        # Baris metrik masing-masing kelas.
        for item in metrics_data["per_class"]:

            writer.writerow({
                "type": "per_class",
                "class_id": item["class_id"],
                "class_name": item["class_name"],
                "precision": item["precision"],
                "recall": item["recall"],
                "mAP50": item["mAP50"],
                "mAP75": "",
                "mAP50-95": item["mAP50-95"],
            })

    print("CSV metrics disimpan:", output_path)


# ==========================================
# MEMBUAT GRAFIK METRIK KESELURUHAN
# ==========================================

def create_metrics_chart(metrics_data, output_dir):

    print("\nMembuat grafik evaluasi...")

    overall = metrics_data["overall"]

    metric_names = [
        "Precision",
        "Recall",
        "mAP50",
        "mAP75",
        "mAP50-95",
    ]

    metric_values = [
        overall["precision"],
        overall["recall"],
        overall["mAP50"],
        overall["mAP75"],
        overall["mAP50-95"],
    ]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        metric_names,
        metric_values
    )

    plt.title(
        "AgroTofu - YOLO Good Bad Evaluation"
    )

    plt.ylabel("Score")
    plt.xlabel("Evaluation Metrics")
    plt.ylim(0, 1.10)

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    for bar, value in zip(bars, metric_values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    chart_path = (
        output_dir / "detection_metrics_chart.png"
    )

    plt.savefig(
        chart_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Grafik keseluruhan:", chart_path)


# ==========================================
# GRAFIK METRIK PER KELAS
# ==========================================

def create_class_metrics_chart(metrics_data, output_dir):

    per_class = metrics_data["per_class"]

    if not per_class:
        print(
            "Metrik per kelas tidak tersedia."
        )
        return

    class_names = [
        item["class_name"]
        for item in per_class
    ]

    metric_names = [
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95",
    ]

    metric_keys = [
        "precision",
        "recall",
        "mAP50",
        "mAP50-95",
    ]

    x_positions = list(range(len(class_names)))

    width = 0.18

    plt.figure(figsize=(10, 6))

    for index, (metric_name, metric_key) in enumerate(
        zip(metric_names, metric_keys)
    ):

        values = [
            item[metric_key]
            for item in per_class
        ]

        positions = [
            x + (index - 1.5) * width
            for x in x_positions
        ]

        bars = plt.bar(
            positions,
            values,
            width,
            label=metric_name
        )

        for bar, value in zip(bars, values):

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.title(
        "AgroTofu - Good Bad Per-Class Evaluation"
    )

    plt.xlabel("Soybean Quality Classes")
    plt.ylabel("Score")
    plt.xticks(
        x_positions,
        class_names
    )

    plt.ylim(0, 1.15)
    plt.legend()
    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    chart_path = (
        output_dir / "per_class_metrics_chart.png"
    )

    plt.savefig(
        chart_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Grafik per kelas:", chart_path)


# ==========================================
# MENYIMPAN CONTOH PREDIKSI
# ==========================================

def save_test_predictions(
    model,
    device,
    test_images,
    output_dir
):

    print("\n===================================")
    print("SAVING TEST PREDICTIONS")
    print("===================================")

    prediction_dir = output_dir / "test_predictions"

    model.predict(

        source=str(test_images),

        imgsz=YOLO_DETECTION_IMAGE_SIZE,

        conf=YOLO_DETECTION_CONFIDENCE,

        iou=YOLO_DETECTION_IOU,

        device=device,

        save=True,
        save_txt=True,
        save_conf=True,

        project=str(output_dir),

        name="test_predictions",

        exist_ok=True,

        verbose=True,
    )

    print(
        "Hasil visualisasi disimpan:",
        prediction_dir
    )


# ==========================================
# RINGKASAN HASIL
# ==========================================

def print_summary(metrics_data):

    overall = metrics_data["overall"]

    print("\n")
    print("=" * 55)
    print("AGROTOFU YOLO GOOD BAD EVALUATION")
    print("=" * 55)

    print("Kelas: Good dan Bad")

    print(
        "Precision :",
        f"{overall['precision'] * 100:.2f}%"
    )

    print(
        "Recall    :",
        f"{overall['recall'] * 100:.2f}%"
    )

    print(
        "mAP50     :",
        f"{overall['mAP50'] * 100:.2f}%"
    )

    print(
        "mAP75     :",
        f"{overall['mAP75'] * 100:.2f}%"
    )

    print(
        "mAP50-95  :",
        f"{overall['mAP50-95'] * 100:.2f}%"
    )

    print("\nPER CLASS")

    for item in metrics_data["per_class"]:

        print(
            f"{item['class_name']}: "
            f"Precision={item['precision'] * 100:.2f}%, "
            f"Recall={item['recall'] * 100:.2f}%, "
            f"mAP50={item['mAP50'] * 100:.2f}%"
        )

    print("=" * 55)


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n")
    print("=" * 55)
    print("AGROTOFU - YOLO GOOD BAD DETECTION")
    print("MODEL EVALUATION")
    print("=" * 55)

    # 1. Periksa kebutuhan.
    model_path, test_images = check_requirements()

    # 2. Periksa perangkat.
    device = check_device()

    # 3. Muat model dua kelas.
    model = load_model(model_path)

    # 4. Evaluasi dataset test.
    metrics = evaluate_model(model, device)

    # 5. Ambil metrik.
    metrics_data = extract_metrics(metrics)

    # 6. Folder hasil evaluasi.
    output_dir = (
        YOLO_DETECTION_RESULT_DIR
        / "evaluation"
        / "reports"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 7. Simpan JSON.
    save_metrics_json(
        metrics_data,
        output_dir
    )

    # 8. Simpan CSV.
    save_metrics_csv(
        metrics_data,
        output_dir
    )

    # 9. Grafik keseluruhan.
    create_metrics_chart(
        metrics_data,
        output_dir
    )

    # 10. Grafik per kelas.
    create_class_metrics_chart(
        metrics_data,
        output_dir
    )

    # 11. Visualisasi prediksi.
    save_test_predictions(
        model,
        device,
        test_images,
        output_dir
    )

    # 12. Ringkasan.
    print_summary(metrics_data)

    print("\nEvaluasi selesai.")
    print("Lokasi hasil:", output_dir)


if __name__ == "__main__":
    main()
