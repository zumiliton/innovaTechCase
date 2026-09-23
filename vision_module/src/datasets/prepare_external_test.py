import shutil
from pathlib import Path


"""
Este script s eencarga de coger las imagenes que nos pasaron 
de ejemplo data/raw/external/Arduino_UNO,MEGA,NANO y genera un unico directorio
flat con todas las iamgenes clasificadas en su etiqueta para la evaluación posterior del
clasificador visual
"""



# PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "external"
)

DEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "external_test_flat"
)


# CLASS CONFIGURATION

CLASSES = {
    "Arduino_mega": "MEGA",
    "Arduino_uno": "UNO",
    "Arduino_nano": "NANO",
}


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".dng",
    ".tif",
    ".tiff",
    ".webp",
}


# PREPARE EXTERNAL TEST SET

def prepare_external_test():
    """
    Flatten the external test dataset into a single directory.

    The original class is preserved in the filename:

        MEGA__image.jpg
        UNO__image.jpg
        NANO__image.jpg

    The original images are copied unchanged.
    """

    # --------------------------------------------------------
    # Validate source
    # --------------------------------------------------------

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"External dataset not found:\n{SOURCE}"
        )

    # --------------------------------------------------------
    # Create destination
    # --------------------------------------------------------

    DEST.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("PREPARING EXTERNAL TEST DATASET")
    print("=" * 60)

    print(f"\nProject root:")
    print(f"  {PROJECT_ROOT}")

    print(f"\nSource:")
    print(f"  {SOURCE}")

    print(f"\nDestination:")
    print(f"  {DEST}")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    class_counts = {
        class_name: 0
        for class_name in CLASSES.values()
    }

    total = 0

    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------

    for folder_name, class_name in CLASSES.items():

        source_dir = SOURCE / folder_name

        if not source_dir.exists():

            print(
                f"[WARNING] Directory not found: "
                f"{source_dir}"
            )

            continue

        image_paths = sorted(
            source_dir.iterdir()
        )

        for image_path in image_paths:

            # Ignore unsupported files
            if (
                not image_path.is_file()
                or image_path.suffix.lower()
                not in VALID_EXTENSIONS
            ):
                continue

            # ------------------------------------------------
            # Preserve ground-truth class in filename
            # ------------------------------------------------

            new_name = (
                f"{class_name}__"
                f"{image_path.name}"
            )

            destination = DEST / new_name

            # ------------------------------------------------
            # Copy without modifying the image
            # ------------------------------------------------

            shutil.copy2(
                image_path,
                destination
            )

            class_counts[class_name] += 1
            total += 1

            print(
                f"{image_path} -> {destination}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXTERNAL TEST DATASET READY")
    print("=" * 60)

    print("\nImages per class:")

    for class_name in CLASSES.values():

        print(
            f"  {class_name}: "
            f"{class_counts[class_name]}"
        )

    print(f"\nTotal images: {total}")

    print(f"\nOutput directory:")
    print(f"  {DEST}")


# MAIN

if __name__ == "__main__":
    prepare_external_test()