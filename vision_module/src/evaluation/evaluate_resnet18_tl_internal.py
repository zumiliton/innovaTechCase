import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
print(PROJECT_ROOT)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "resnet18_arduino_tl"
    / "best_resnet18.pt"
)

# INTERNAL TEST SET
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "arduino_3class"
    / "test"
)


# ============================================================
# Configuration
# ============================================================

CLASS_NAMES = ["MEGA", "NANO", "UNO"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}

IMAGE_SIZE = 224


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")
print(f"Model: {MODEL_PATH}")
print(f"Internal test set: {DATA_DIR}")


# ============================================================
# Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Load model
# ============================================================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES),
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model = model.to(device)
model.eval()


# ============================================================
# Collect test images
# ============================================================

image_paths = []

for class_name in CLASS_NAMES:

    class_dir = DATA_DIR / class_name

    if not class_dir.exists():
        raise RuntimeError(
            f"Class directory not found: {class_dir}"
        )

    for image_path in class_dir.iterdir():

        if image_path.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }:
            image_paths.append(image_path)


if not image_paths:
    raise RuntimeError(
        f"No images found in {DATA_DIR}"
    )

print(f"\nFound {len(image_paths)} internal test images.\n")


# ============================================================
# Evaluate
# ============================================================

y_true = []
y_pred = []


with torch.no_grad():

    for image_path in sorted(image_paths):

        # Ground truth comes from the directory name:
        #
        # test/
        # ├── MEGA/
        # ├── NANO/
        # └── UNO/

        true_label = image_path.parent.name.upper()

        if true_label not in CLASS_TO_IDX:
            print(
                f"WARNING: Unknown class "
                f"{true_label}. Skipping."
            )
            continue

        # Load image
        image = Image.open(image_path).convert("RGB")

        image_tensor = (
            transform(image)
            .unsqueeze(0)
            .to(device)
        )

        # Prediction
        logits = model(image_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_idx = torch.argmax(
            probabilities,
            dim=1,
        ).item()

        predicted_label = IDX_TO_CLASS[predicted_idx]

        confidence = probabilities[
            0,
            predicted_idx
        ].item()

        y_true.append(true_label)
        y_pred.append(predicted_label)

        print(
            f"{image_path.name:40s} "
            f"true={true_label:5s} "
            f"pred={predicted_label:5s} "
            f"conf={confidence:.4f}"
        )


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

report = classification_report(
    y_true,
    y_pred,
    labels=CLASS_NAMES,
    target_names=CLASS_NAMES,
    zero_division=0,
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=CLASS_NAMES,
)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 60)
print("INTERNAL TEST RESULTS")
print("=" * 60)

print(f"\nNumber of images: {len(y_true)}")
print(f"Accuracy:         {accuracy:.4f}")

print("\nConfusion Matrix")
print("Rows = True label")
print("Cols = Predicted label\n")

print(
    "          "
    + "  ".join(
        f"{c:>6}" for c in CLASS_NAMES
    )
)

for label, row in zip(CLASS_NAMES, cm):

    print(
        f"{label:>6}    "
        + "  ".join(
            f"{value:6d}"
            for value in row
        )
    )

print("\nClassification Report\n")

print(report)