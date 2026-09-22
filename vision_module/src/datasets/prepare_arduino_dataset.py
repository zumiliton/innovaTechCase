#!/usr/bin/env python3

"""
Prepare the Arduino dataset by mapping the original 10 classes
into the 3 target classes:

    MEGA
    UNO
    NANO

Original classes:

    Arduino_Mega_2560       -> MEGA

    Arduino_Uno_R3          -> UNO
    Arduino_Uno_WiFi_Rev2   -> UNO

    Arduino_Nano            -> NANO
    Arduino_Nano_33_BLE     -> NANO
    Arduino_Nano_33_IoT     -> NANO

The following classes are ignored:

    Arduino_Due
    Arduino_Leonardo
    Arduino_Micro
    Arduino_MKR_WiFi_1010

Input:

data/raw/external/arduino.folder/
├── train/
├── valid/
└── test/

Output:

data/processed/arduino_3class/
├── train/
│   ├── MEGA/
│   ├── NANO/
│   └── UNO/
├── val/
│   ├── MEGA/
│   ├── NANO/
│   └── UNO/
└── test/
    ├── MEGA/
    ├── NANO/
    └── UNO/
"""

from pathlib import Path
import shutil


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "arduino.folder"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "arduino_3class"
)


# ============================================================
# Class mapping
# ============================================================

CLASS_MAPPING = {
    # MEGA
    "Arduino_Mega_2560": "MEGA",

    # UNO
    "Arduino_Uno_R3": "UNO",
    "Arduino_Uno_WiFi_Rev2": "UNO",

    # NANO
    "Arduino_Nano": "NANO",
    "Arduino_Nano_33_BLE": "NANO",
    "Arduino_Nano_33_IoT": "NANO",
}


# Classes we want in the final dataset
TARGET_CLASSES = [
    "MEGA",
    "NANO",
    "UNO",
]


# Original split -> output split
SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ARDUINO DATASET PREPARATION")
    print("=" * 70)

    print(f"\nInput:")
    print(f"  {INPUT_DIR}")

    print(f"\nOutput:")
    print(f"  {OUTPUT_DIR}")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_DIR}"
        )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    for split in SPLITS.values():

        for target_class in TARGET_CLASSES:

            output_class_dir = (
                OUTPUT_DIR
                / split
                / target_class
            )

            output_class_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

    # --------------------------------------------------------
    # Process each split
    # --------------------------------------------------------

    total_statistics = {}

    for input_split, output_split in SPLITS.items():

        print("\n" + "-" * 70)
        print(
            f"Processing: "
            f"{input_split} -> {output_split}"
        )
        print("-" * 70)

        split_statistics = {
            "MEGA": 0,
            "NANO": 0,
            "UNO": 0,
        }

        input_split_dir = (
            INPUT_DIR / input_split
        )

        if not input_split_dir.exists():

            print(
                f"WARNING: split does not exist: "
                f"{input_split_dir}"
            )

            continue

        # ----------------------------------------------------
        # Iterate through the classes we actually want
        # ----------------------------------------------------

        for original_class, target_class in (
            CLASS_MAPPING.items()
        ):

            source_dir = (
                input_split_dir
                / original_class
            )

            if not source_dir.exists():

                print(
                    f"WARNING: class not found: "
                    f"{source_dir}"
                )

                continue

            image_files = sorted(
                [
                    path
                    for path in source_dir.rglob("*")
                    if (
                        path.is_file()
                        and path.suffix.lower()
                        in IMAGE_EXTENSIONS
                    )
                ]
            )

            print(
                f"{original_class:25s}"
                f" -> "
                f"{target_class:5s}"
                f" : "
                f"{len(image_files)} images"
            )

            target_dir = (
                OUTPUT_DIR
                / output_split
                / target_class
            )

            # ------------------------------------------------
            # Copy images
            # ------------------------------------------------

            for index, source_file in enumerate(
                image_files
            ):

                # Avoid collisions between images from
                # different original classes that map to
                # the same target class.
                new_filename = (
                    f"{original_class}"
                    f"__"
                    f"{index:06d}"
                    f"{source_file.suffix.lower()}"
                )

                destination = (
                    target_dir
                    / new_filename
                )

                shutil.copy2(
                    source_file,
                    destination,
                )

                split_statistics[
                    target_class
                ] += 1

        total_statistics[
            output_split
        ] = split_statistics

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    grand_total = 0

    for split, statistics in (
        total_statistics.items()
    ):

        split_total = sum(
            statistics.values()
        )

        grand_total += split_total

        print(f"\n{split.upper()}")

        for target_class in TARGET_CLASSES:

            print(
                f"  {target_class:5s}: "
                f"{statistics[target_class]}"
            )

        print(
            f"  TOTAL: {split_total}"
        )

    print("\n" + "-" * 70)

    print(
        f"TOTAL IMAGES: {grand_total}"
    )

    print("\nOutput structure:")

    print(
        f"""
{OUTPUT_DIR}/
├── train/
│   ├── MEGA/
│   ├── NANO/
│   └── UNO/
├── val/
│   ├── MEGA/
│   ├── NANO/
│   └── UNO/
└── test/
    ├── MEGA/
    ├── NANO/
    └── UNO/
"""
    )

    print("Dataset preparation completed.")


if __name__ == "__main__":
    main()