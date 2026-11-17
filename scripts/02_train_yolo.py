
import sys
import shutil
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

import yaml
import torch
from ultralytics import YOLO


# ==========================================
# KONFIGURASI AGROTOFU
# ==========================================

from config import (
    YOLO_DETECTION_DATA_YAML,
    YOLO_DETECTION_MODEL_DIR,
    YOLO_DETECTION_MODEL_PATH,
    YOLO_DETECTION_RESULT_DIR,
    YOLO_DETECTION_BASE_MODEL,
    YOLO_DETECTION_IMAGE_SIZE,
    YOLO_DETECTION_EPOCHS,
    YOLO_DETECTION_BATCH_SIZE,
    YOLO_DETECTION_PATIENCE,
    YOLO_DETECTION_RANDOM_STATE,
    YOLO_DETECTION_WORKERS,
    YOLO_DEVICE,
    YOLO_AMP,
)


# ==========================================
# MEMERIKSA DATASET
# ==========================================

def check_dataset():

    print("\n===================================")
    print("DATASET CHECK")
    print("===================================")

    if not YOLO_DETECTION_DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML tidak ditemukan:\n"
            f"{YOLO_DETECTION_DATA_YAML}\n\n"
            "Jalankan preprocessing terlebih dahulu."
        )

    with open(
        YOLO_DETECTION_DATA_YAML,
        "r",
        encoding="utf-8"
    ) as file:

        dataset_config = yaml.safe_load(file)

    if not isinstance(dataset_config, dict):
        raise ValueError(
            "Format dataset YAML tidak valid."
        )

    # Validasi jumlah kelas.
    number_of_classes = int(
        dataset_config.get("nc", 0)
    )

    class_names = dataset_config.get("names", [])

    if isinstance(class_names, dict):
        class_names = [
            class_names[key]
            for key in sorted(
                class_names,
                key=lambda item: int(item)
            )
        ]

    if number_of_classes != 2:
        raise ValueError(
            f"Dataset harus memiliki 2 kelas, "
            f"tetapi ditemukan {number_of_classes}."
        )

    if class_names != ["Good", "Bad"]:
        raise ValueError(
            f"Nama kelas tidak sesuai: {class_names}\n"
            "Seharusnya: ['Good', 'Bad']"
        )

    # Membaca lokasi dataset.
    dataset_root = Path(
        dataset_config.get(
            "path",
            YOLO_DETECTION_DATA_YAML.parent
        )
    )

    if not dataset_root.is_absolute():
        dataset_root = (
            YOLO_DETECTION_DATA_YAML.parent
            / dataset_root
        ).resolve()

    print("Dataset YAML :", YOLO_DETECTION_DATA_YAML)
    print("Dataset root :", dataset_root)
    print("Jumlah kelas :", number_of_classes)
    print("Nama kelas   :", class_names)

    # Validasi folder train, valid, dan test.
    required_folders = [
        dataset_root / "train" / "images",
        dataset_root / "train" / "labels",
        dataset_root / "valid" / "images",
        dataset_root / "valid" / "labels",
        dataset_root / "test" / "images",
        dataset_root / "test" / "labels",
    ]

    for folder in required_folders:

        if not folder.exists():
            raise FileNotFoundError(
                f"Folder dataset tidak ditemukan:\n{folder}"
            )

        print("OK:", folder)

    # Pastikan setiap split memiliki gambar.
    for split in ["train", "valid", "test"]:

        image_folder = dataset_root / split / "images"

        images = [
            path for path in image_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {
                ".jpg", ".jpeg", ".png",
                ".bmp", ".webp"
            }
        ]

        if not images:
            raise ValueError(
                f"Tidak ada gambar pada split {split}."
            )

        print(
            f"Jumlah gambar {split}: {len(images)}"
        )

    print("\nDataset Good/Bad berhasil diverifikasi.")

    return dataset_root


# ==========================================
# MEMERIKSA GPU
# ==========================================

def check_gpu():

    print("\n===================================")
    print("GPU INFORMATION")
    print("===================================")

    print("PyTorch Version:", torch.__version__)
    print("CUDA Available :", torch.cuda.is_available())

    if torch.cuda.is_available():

        print("CUDA Version:", torch.version.cuda)
        print("GPU Count:", torch.cuda.device_count())

        for index in range(torch.cuda.device_count()):

            gpu_name = torch.cuda.get_device_name(index)

            gpu_memory = (
                torch.cuda.get_device_properties(index)
                .total_memory / (1024 ** 3)
            )

            print(f"\nGPU {index}: {gpu_name}")
            print("VRAM:", round(gpu_memory, 2), "GB")

        if isinstance(YOLO_DEVICE, int):

            if YOLO_DEVICE >= torch.cuda.device_count():
                raise ValueError(
                    f"GPU index {YOLO_DEVICE} tidak tersedia."
                )

            device = YOLO_DEVICE

        else:
            device = YOLO_DEVICE

        print("\nTraining menggunakan GPU:", device)

        return device

    print("\nGPU tidak tersedia.")
    print("Training menggunakan CPU.")

    return "cpu"


# ==========================================
# MEMERIKSA KONFIGURASI TRAINING
# ==========================================

def show_training_configuration(device):

    print("\n===================================")
    print("YOLO GOOD BAD DETECTION CONFIGURATION")
    print("===================================")

    print("Base Model :", YOLO_DETECTION_BASE_MODEL)
    print("Dataset    :", YOLO_DETECTION_DATA_YAML)
    print("Device     :", device)
    print("Image Size :", YOLO_DETECTION_IMAGE_SIZE)
    print("Epochs     :", YOLO_DETECTION_EPOCHS)
    print("Batch Size :", YOLO_DETECTION_BATCH_SIZE)
    print("Patience   :", YOLO_DETECTION_PATIENCE)
    print("Workers    :", YOLO_DETECTION_WORKERS)
    print(
        "AMP        :",
        YOLO_AMP and torch.cuda.is_available()
    )

    print("Optimizer  : AdamW")
    print("Learning Rate: 0.001")
    print("Classes    : Good, Bad")


# ==========================================
# TRAINING YOLO OBJECT DETECTION
# ==========================================

def train_model(device):

    print("\n===================================")
    print("START YOLO GOOD BAD TRAINING")
    print("===================================")

    # Memuat model dasar.
    model = YOLO(
        YOLO_DETECTION_BASE_MODEL
    )

    print("\nModel berhasil dimuat.")
    print("Base model classes:", model.names)

    # Training dua kelas Good dan Bad.
    results = model.train(

        # Dataset hasil preprocessing.
        data=str(YOLO_DETECTION_DATA_YAML),

        # Parameter utama.
        epochs=YOLO_DETECTION_EPOCHS,
        imgsz=YOLO_DETECTION_IMAGE_SIZE,
        batch=YOLO_DETECTION_BATCH_SIZE,
        patience=YOLO_DETECTION_PATIENCE,

        # Optimizer.
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        cos_lr=True,

        # Pretrained model.
        pretrained=True,
        seed=YOLO_DETECTION_RANDOM_STATE,
        deterministic=True,

        # GPU.
        workers=YOLO_DETECTION_WORKERS,
        device=device,
        amp=(
            YOLO_AMP
            and torch.cuda.is_available()
        ),

        # Hasil training.
        project=str(YOLO_DETECTION_RESULT_DIR),
        name="soybean_good_bad",
        exist_ok=True,

        save=True,
        save_period=-1,

        # Evaluasi dan visualisasi.
        plots=True,
        verbose=True,

        # Augmentasi.
        close_mosaic=10,

        # Penggunaan memori.
        cache=False,
    )

    return model, results


# ==========================================
# MENYIMPAN MODEL TERBAIK
# ==========================================

def save_best_model(model):

    print("\n===================================")
    print("SAVING BEST MODEL")
    print("===================================")

    trainer = getattr(
        model,
        "trainer",
        None
    )

    if trainer is None:
        raise RuntimeError(
            "Trainer Ultralytics tidak ditemukan."
        )

    run_dir = Path(trainer.save_dir)

    best_weight = (
        run_dir / "weights" / "best.pt"
    )

    if not best_weight.exists():
        raise FileNotFoundError(
            f"Best model tidak ditemukan:\n{best_weight}"
        )

    YOLO_DETECTION_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = YOLO_DETECTION_MODEL_PATH

    shutil.copy2(
        best_weight,
        destination
    )

    print("\nModel terbaik berhasil disimpan.")
    print("Sumber      :", best_weight)
    print("Destination :", destination)

    return destination


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("\n")
    print("=" * 60)
    print("AGROTOFU - YOLO GOOD BAD OBJECT DETECTION")
    print("=" * 60)

    # 1. Periksa dataset.
    check_dataset()

    # 2. Periksa GPU.
    device = check_gpu()

    # 3. Tampilkan konfigurasi.
    show_training_configuration(device)

    # 4. Reset statistik GPU.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 5. Jalankan training.
    model, results = train_model(device)

    # 6. Simpan best model.
    destination = save_best_model(model)

    # 7. Informasi GPU.
    if torch.cuda.is_available():

        peak_memory = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 3)
        )

        print(
            "\nPeak GPU Memory:",
            round(peak_memory, 2),
            "GB"
        )

    print("\n===================================")
    print("TRAINING SELESAI")
    print("===================================")

    print("Training Device:", device)
    print("Best Model     :", destination)
    print("Classes        : Good, Bad")

    print("\nModel siap digunakan untuk evaluasi.")

    return results


if __name__ == "__main__":
    main()
