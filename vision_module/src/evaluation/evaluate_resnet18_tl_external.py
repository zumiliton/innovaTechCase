import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

"""
Este script mhace el test del clasificador visual (RESNET) respecto a las 40 imagenes de ejemplo
que nos pasan, y es el importante, por que clasificador se entrena con un DATASET distitno, y si tiene buen 
performance aqui significa que generaliza y no hay DOMAIN SHIFT
"""


# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "resnet18_arduino_tl"
    / "best_resnet18.pt"
)

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "external_test_flat"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "resnet18_arduino_tl"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.json"



# Configuration


CLASS_NAMES = ["MEGA", "NANO", "UNO"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}

IMAGE_SIZE = 224



# Device


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"Model: {MODEL_PATH}")
print(f"External images: {DATA_DIR}")



# Transform


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])



# Load model


model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES),
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
)

# Support both:
# 1. checkpoint containing {"model_state_dict": ...}
# 2. raw state_dict
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model = model.to(device)
model.eval()



# Evaluate


image_paths = sorted(
    [
        p
        for p in DATA_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
)

if not image_paths:
    raise RuntimeError(f"No images found in {DATA_DIR}")

print(f"\nFound {len(image_paths)} external images.\n")

y_true = []
y_pred = []
results = []


with torch.no_grad():

    for image_path in image_paths:

        # ----------------------------------------------------
        # Ground-truth from filename
        # Example:
        # MEGA__2831.jpg
        # UNO__1234.jpg
        # NANO__5678.jpg
        # ----------------------------------------------------

        filename = image_path.name
        true_label = filename.split("__")[0].upper()

        if true_label not in CLASS_TO_IDX:
            print(
                f"WARNING: Could not determine label for "
                f"{filename}. Skipping."
            )
            continue

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        logits = model(image_tensor)

        probabilities = torch.softmax(logits, dim=1)

        predicted_idx = torch.argmax(probabilities, dim=1).item()

        predicted_label = IDX_TO_CLASS[predicted_idx]

        confidence = probabilities[0, predicted_idx].item()

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        y_true.append(true_label)
        y_pred.append(predicted_label)

        results.append({
            "filename": filename,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
        })

        print(
            f"{filename:30s} "
            f"true={true_label:5s} "
            f"pred={predicted_label:5s} "
            f"conf={confidence:.4f}"
        )



# Metrics


accuracy = accuracy_score(y_true, y_pred)

report = classification_report(
    y_true,
    y_pred,
    labels=CLASS_NAMES,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

macro_f1 = report["macro avg"]["f1-score"]

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=CLASS_NAMES,
)



# Print results


print("\n" + "=" * 60)
print("EXTERNAL TEST RESULTS")
print("=" * 60)

print(f"\nNumber of images: {len(y_true)}")
print(f"Accuracy:         {accuracy:.4f}")
print(f"Macro F1:         {macro_f1:.4f}")

print("\nConfusion Matrix")
print("Rows = True label")
print("Cols = Predicted label\n")

print("          " + "  ".join(f"{c:>6}" for c in CLASS_NAMES))

for label, row in zip(CLASS_NAMES, cm):
    print(
        f"{label:>6}    "
        + "  ".join(f"{value:6d}" for value in row)
    )

print("\nPer-class metrics")

for label in CLASS_NAMES:
    print(
        f"{label:5s} "
        f"precision={report[label]['precision']:.4f} "
        f"recall={report[label]['recall']:.4f} "
        f"f1={report[label]['f1-score']:.4f}"
    )



# Save predictions CSV


import csv

with open(PREDICTIONS_PATH, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "true_label",
            "predicted_label",
            "confidence",
        ],
    )

    writer.writeheader()
    writer.writerows(results)



# Save metrics JSON


metrics = {
    "num_images": len(y_true),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "classes": CLASS_NAMES,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)


print("\nResults saved to:")
print(f"  Predictions: {PREDICTIONS_PATH}")
print(f"  Metrics:     {METRICS_PATH}")
