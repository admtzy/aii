
import sys
import shutil
from pathlib import Path
import math
import yaml

# ==========================================
# ROOT PROJECT
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    ROBOFLOW_DATASET_DIR,
    ROBOFLOW_ORIGINAL_YAML,
    YOLO_DETECTION_DATA_YAML,
)

# ==========================================
# KONFIGURASI DATASET
# ==========================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

SPLIT_MAPPING = {
    "train": "train",
    "val": "valid",
    "test": "test",
}

# Folder hasil preprocessing.
# Dataset asli Roboflow tidak diubah.
PROCESSED_DATASET_DIR = (
    ROBOFLOW_DATASET_DIR.parent
    / f"{ROBOFLOW_DATASET_DIR.name}_good_bad"
)

# ==========================================
# MAPPING KELAS ROBOFLOW
# ==========================================

# Mapping berdasarkan nama kelas asli.
# Good = Intact
# Bad = Broken, Immature, Skin Damaged, Spotted

CLASS_MAPPING = {
    "broken": 1,
    "immature": 1,
    "intact": 0,
    "skin damaged": 1,
    "spotted": 1,
}

NEW_CLASS_NAMES = [
    "Good",
    "Bad",
]

NUMBER_OF_CLASSES = 2


# ==========================================
# VALIDASI GAMBAR
# ==========================================

def is_valid_image(image_path):
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image.verify()

        return True

    except Exception:
        return False


# ==========================================
# MEMBACA DATA.YAML ASLI
# ==========================================

def load_original_yaml():

    yaml_path = ROBOFLOW_ORIGINAL_YAML

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"File data.yaml tidak ditemukan:\n{yaml_path}"
        )

    with open(
        yaml_path,
        "r",
        encoding="utf-8"
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Isi data.yaml harus berupa YAML mapping."
        )

    if "names" not in data:
        raise ValueError(
            "Bagian names tidak ditemukan."
        )

    names = data["names"]

    if isinstance(names, dict):
        try:
            names = [
                names[key]
                for key in sorted(
                    names,
                    key=lambda item: int(item)
                )
            ]
        except (TypeError, ValueError):
            raise ValueError(
                "Key names harus berupa ID angka."
            )

    if not isinstance(names, list) or not names:
        raise ValueError(
            "Daftar kelas tidak valid."
        )

    names = [
        str(name).strip()
        for name in names
    ]

    if "nc" in data and int(data["nc"]) != len(names):
        raise ValueError(
            "Jumlah nc tidak sama dengan jumlah names."
        )

    normalized_names = [
        name.lower()
        for name in names
    ]

    expected_names = set(CLASS_MAPPING.keys())

    if set(normalized_names) != expected_names:
        raise ValueError(
            "Nama kelas pada YAML Roboflow tidak sesuai.\n"
            f"Ditemukan: {names}\n"
            f"Seharusnya: {list(CLASS_MAPPING.keys())}"
        )

    print("YAML asli       :", yaml_path)
    print("Kelas asli      :", names)
    print("Kelas baru      :", NEW_CLASS_NAMES)

    return data, names


# ==========================================
# MENCARI GAMBAR
# ==========================================

def get_image_files(image_folder):

    if not image_folder.exists():
        raise FileNotFoundError(
            f"Folder gambar tidak ditemukan:\n{image_folder}"
        )

    images = []

    for path in image_folder.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):

            if is_valid_image(path):
                images.append(path)

    return sorted(images)


# ==========================================
# VALIDASI DAN KONVERSI LABEL YOLO
# ==========================================

def convert_label(
    source_label,
    destination_label,
    number_of_original_classes
):

    if not source_label.exists():
        raise FileNotFoundError(
            f"Label tidak ditemukan:\n{source_label}"
        )

    converted_lines = []

    class_counts = {
        0: 0,
        1: 0,
    }

    with open(
        source_label,
        "r",
        encoding="utf-8"
    ) as file:

        lines = file.readlines()

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        # Label kosong diperbolehkan.
        if not line:
            continue

        values = line.split()

        if len(values) != 5:
            raise ValueError(
                f"Format label salah: {source_label}, "
                f"baris {line_number}."
            )

        try:
            old_class_id = int(values[0])

            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

        except ValueError:
            raise ValueError(
                f"Nilai label bukan angka: {source_label}, "
                f"baris {line_number}."
            )

        coordinates = [
            x_center,
            y_center,
            width,
            height,
        ]

        if not all(
            math.isfinite(value)
            for value in coordinates
        ):
            raise ValueError(
                f"Koordinat NaN/inf: {source_label}, "
                f"baris {line_number}."
            )

        if not (
            0 <= old_class_id < number_of_original_classes
        ):
            raise ValueError(
                f"Class ID tidak valid: {old_class_id} "
                f"pada {source_label}."
            )

        if not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            raise ValueError(
                f"Bounding box di luar rentang YOLO: "
                f"{source_label}, baris {line_number}."
            )

        # Mengubah ID berdasarkan nama kelas.
        # Urutan names berasal dari YAML asli.
        new_class_id = CLASS_MAPPING[
            ORIGINAL_CLASS_NAMES[old_class_id].lower()
        ]

        # Koordinat tidak diubah.
        new_line = (
            f"{new_class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        converted_lines.append(new_line)

        class_counts[new_class_id] += 1

    destination_label.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    destination_label.write_text(
        "\n".join(converted_lines),
        encoding="utf-8"
    )

    return {
        "total": sum(class_counts.values()),
        "good": class_counts[0],
        "bad": class_counts[1],
    }


# ==========================================
# PREPROCESSING SETIAP SPLIT
# ==========================================

def process_split(
    split_name,
    folder_name,
    number_of_original_classes
):

    source_split = ROBOFLOW_DATASET_DIR / folder_name

    source_image_dir = source_split / "images"
    source_label_dir = source_split / "labels"

    output_split = PROCESSED_DATASET_DIR / folder_name

    output_image_dir = output_split / "images"
    output_label_dir = output_split / "labels"

    print("\n" + "=" * 55)
    print(f"PREPROCESSING: {split_name.upper()}")
    print("=" * 55)

    images = get_image_files(source_image_dir)

    if not images:
        raise ValueError(
            f"Tidak ada gambar valid:\n{source_image_dir}"
        )

    if not source_label_dir.exists():
        raise FileNotFoundError(
            f"Folder labels tidak ditemukan:\n"
            f"{source_label_dir}"
        )

    total_boxes = 0
    total_good = 0
    total_bad = 0
    empty_labels = 0

    for index, image_path in enumerate(images, start=1):

        relative_path = image_path.relative_to(
            source_image_dir
        )

        source_label = (
            source_label_dir
            / relative_path.with_suffix(".txt")
        )

        output_image = output_image_dir / relative_path

        output_label = (
            output_label_dir
            / relative_path.with_suffix(".txt")
        )

        # Validasi dan konversi label.
        result = convert_label(
            source_label,
            output_label,
            number_of_original_classes
        )

        # Salin gambar tanpa mengubah isinya.
        output_image.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            image_path,
            output_image
        )

        total_boxes += result["total"]
        total_good += result["good"]
        total_bad += result["bad"]

        if result["total"] == 0:
            empty_labels += 1

        if index % 50 == 0 or index == len(images):
            print(
                f"Memproses: {index}/{len(images)}"
            )

    print("\nHASIL SPLIT:", split_name.upper())
    print("Jumlah gambar :", len(images))
    print("Total objek   :", total_boxes)
    print("Good          :", total_good)
    print("Bad           :", total_bad)
    print("Label kosong  :", empty_labels)

    return {
        "images": len(images),
        "boxes": total_boxes,
        "good": total_good,
        "bad": total_bad,
        "empty": empty_labels,
    }


# ==========================================
# MEMBUAT YAML KHUSUS AGROTOFU
# ==========================================

def create_detection_yaml():

    dataset_yaml = {
        "path": str(
            PROCESSED_DATASET_DIR.resolve()
        ),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": NUMBER_OF_CLASSES,
        "names": NEW_CLASS_NAMES,
    }

    # YAML disimpan sesuai lokasi pada config.py.
    YOLO_DETECTION_DATA_YAML.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        YOLO_DETECTION_DATA_YAML,
        "w",
        encoding="utf-8"
    ) as file:

        yaml.safe_dump(
            dataset_yaml,
            file,
            sort_keys=False,
            allow_unicode=True
        )

    print("\nYAML AgroTofu berhasil dibuat:")
    print(YOLO_DETECTION_DATA_YAML)

    print("\nIsi YAML:")

    print(
        yaml.safe_dump(
            dataset_yaml,
            sort_keys=False,
            allow_unicode=True
        )
    )


# ==========================================
# MAIN
# ==========================================

def main():

    global ORIGINAL_CLASS_NAMES

    print("=" * 60)
    print("PREPROCESSING YOLO OBJECT DETECTION")
    print("AGROTOFU - GOOD BAD CLASSIFICATION")
    print("=" * 60)

    # Membaca YAML asli.
    _, ORIGINAL_CLASS_NAMES = load_original_yaml()

    print("\nDataset asli:")
    print(ROBOFLOW_DATASET_DIR)

    print("\nDataset hasil:")
    print(PROCESSED_DATASET_DIR)

    # Membuat folder output.
    PROCESSED_DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_images = 0
    total_boxes = 0
    total_good = 0
    total_bad = 0

    # Memproses train, valid, dan test.
    for split_name, folder_name in SPLIT_MAPPING.items():

        result = process_split(
            split_name,
            folder_name,
            len(ORIGINAL_CLASS_NAMES)
        )

        total_images += result["images"]
        total_boxes += result["boxes"]
        total_good += result["good"]
        total_bad += result["bad"]

    # Membuat YAML dua kelas.
    create_detection_yaml()

    # ======================================
    # RINGKASAN
    # ======================================

    print("\n" + "=" * 60)
    print("PREPROCESSING SELESAI")
    print("=" * 60)

    print("Total gambar       :", total_images)
    print("Total bounding box :", total_boxes)
    print("Total Good         :", total_good)
    print("Total Bad          :", total_bad)

    print("\nPersentase Good    :",
          f"{total_good / total_boxes * 100:.2f}%")

    print("Persentase Bad     :",
          f"{total_bad / total_boxes * 100:.2f}%")

    print("\nJumlah kelas       :", NUMBER_OF_CLASSES)
    print("Nama kelas         :", NEW_CLASS_NAMES)

    print("\nDataset hasil:")
    print(PROCESSED_DATASET_DIR)

    print("\nFile YAML:")
    print(YOLO_DETECTION_DATA_YAML)

    print("\nDataset siap digunakan untuk training YOLO.")


if __name__ == "__main__":
    main()
