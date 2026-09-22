from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PROJECT PATHS
# ============================================================

# evaluate_clip_zero_shot.py
#   -> src/
#       -> evaluation/
#
# parents[0] = src/evaluation
# parents[1] = src
# parents[2] = vision_module

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "external_test_flat"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "003_clip_zero_shot"
    / "clip_external_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "openai/clip-vit-base-patch32"

CLASS_NAMES = [
    "MEGA",
    "UNO",
    "NANO",
]

PROMPTS = [
    "an Arduino Mega board",
    "an Arduino Uno board",
    "an Arduino Nano board",
]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("CLIP ZERO-SHOT ARDUINO EVALUATION")
print("=" * 70)

print(f"Project root: {PROJECT_ROOT}")
print(f"Device:       {device}")
print(f"Model:        {MODEL_NAME}")
print(f"Images:       {IMAGE_DIR}")
print(f"Output:       {OUTPUT_DIR}")
print()


# ============================================================
# VALIDATE INPUT
# ============================================================

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"External test dataset not found:\n"
        f"{IMAGE_DIR}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading CLIP model...")

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

model = CLIPModel.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("CLIP loaded.")
print()


# ============================================================
# TEXT FEATURES
# ============================================================

print("Computing text features...")

text_inputs = processor(
    text=PROMPTS,
    return_tensors="pt",
    padding=True,
)

text_inputs = {
    key: value.to(device)
    for key, value in text_inputs.items()
}

with torch.no_grad():

    text_output = model.text_model(
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
    )

    # CLIP text representation
    text_features = text_output.pooler_output

    # CLIP text projection
    text_features = model.text_projection(
        text_features
    )

    # Normalize
    text_features = (
        text_features
        / text_features.norm(
            dim=-1,
            keepdim=True,
        )
    )

print("Text features ready.")
print()


# ============================================================
# LOAD IMAGES
# ============================================================

image_paths = sorted(
    [
        path
        for path in IMAGE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in VALID_EXTENSIONS
        )
    ]
)

print(f"Images found: {len(image_paths)}")
print()


if not image_paths:
    raise RuntimeError(
        f"No valid images found in:\n{IMAGE_DIR}"
    )


# ============================================================
# ZERO-SHOT PREDICTION
# ============================================================

results = []

for index, image_path in enumerate(
    image_paths,
    start=1,
):

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------
    #
    # Expected filename:
    #
    # MEGA__image.jpg
    # UNO__image.jpg
    # NANO__image.jpg
    #

    ground_truth = (
        image_path.name
        .split("__", 1)[0]
    )

    if ground_truth not in CLASS_NAMES:

        print(
            f"[WARNING] Unknown ground truth: "
            f"{image_path.name}"
        )

        continue

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    image_inputs = processor(
        images=image,
        return_tensors="pt",
    )

    image_inputs = {
        key: value.to(device)
        for key, value in image_inputs.items()
    }

    # --------------------------------------------------------
    # CLIP inference
    # --------------------------------------------------------

    with torch.no_grad():

        image_output = model.vision_model(
            pixel_values=image_inputs[
                "pixel_values"
            ]
        )

        # CLIP image representation
        image_features = (
            image_output.pooler_output
        )

        # CLIP image projection
        image_features = (
            model.visual_projection(
                image_features
            )
        )

        # Normalize
        image_features = (
            image_features
            / image_features.norm(
                dim=-1,
                keepdim=True,
            )
        )

        # Cosine similarity
        similarity = (
            image_features
            @ text_features.T
        )

        # Preserve the original experiment:
        # softmax over the three similarities.
        probabilities = similarity.softmax(
            dim=-1
        )

        predicted_id = (
            similarity.argmax(
                dim=-1
            ).item()
        )

    prediction = CLASS_NAMES[
        predicted_id
    ]

    confidence = probabilities[
        0,
        predicted_id,
    ].item()

    mega_prob = probabilities[
        0,
        0,
    ].item()

    uno_prob = probabilities[
        0,
        1,
    ].item()

    nano_prob = probabilities[
        0,
        2,
    ].item()

    correct = (
        prediction == ground_truth
    )

    results.append(
        {
            "image": image_path.name,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "confidence": confidence,
            "mega_probability": mega_prob,
            "uno_probability": uno_prob,
            "nano_probability": nano_prob,
            "correct": correct,
        }
    )

    print(
        f"[{index:02d}/{len(image_paths)}] "
        f"{image_path.name:55s} "
        f"GT={ground_truth:5s} "
        f"Pred={prediction:5s} "
        f"Conf={confidence:.4f}"
    )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    results
)

if df.empty:
    raise RuntimeError(
        "No valid predictions were generated."
    )


# ============================================================
# GLOBAL ACCURACY
# ============================================================

accuracy = accuracy_score(
    df["ground_truth"],
    df["prediction"],
)

print()
print("=" * 70)
print("GLOBAL RESULTS")
print("=" * 70)

print(
    f"Images evaluated: "
    f"{len(df)}"
)

print(
    f"Correct:         "
    f"{df['correct'].sum()}/{len(df)}"
)

print(
    f"Accuracy:        "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

report = classification_report(
    df["ground_truth"],
    df["prediction"],
    labels=CLASS_NAMES,
    target_names=CLASS_NAMES,
    zero_division=0,
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    df["ground_truth"],
    df["prediction"],
    labels=CLASS_NAMES,
)

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("Rows = Ground Truth")
print("Cols = Prediction")
print()

print(
    f"{'':>12}"
    + "".join(
        f"{name:>10}"
        for name in CLASS_NAMES
    )
)

for i, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{class_name:>12}"
        + "".join(
            f"{cm[i, j]:>10}"
            for j in range(
                len(CLASS_NAMES)
            )
        )
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "predictions.csv"
)

df.to_csv(
    csv_path,
    index=False,
)

print()
print(
    f"CSV saved to: "
    f"{csv_path}"
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

im = ax.imshow(cm)

ax.set_xticks(
    range(len(CLASS_NAMES))
)

ax.set_yticks(
    range(len(CLASS_NAMES))
)

ax.set_xticklabels(
    CLASS_NAMES
)

ax.set_yticklabels(
    CLASS_NAMES
)

ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "Ground Truth"
)

ax.set_title(
    "CLIP Zero-Shot - Arduino External Test"
)

for i in range(
    len(CLASS_NAMES)
):

    for j in range(
        len(CLASS_NAMES)
    ):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
        )

fig.colorbar(
    im,
    ax=ax,
)

plt.tight_layout()

cm_path = (
    OUTPUT_DIR
    / "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    f"Confusion matrix saved to: "
    f"{cm_path}"
)


# ============================================================
# ERROR ANALYSIS
# ============================================================

errors = df[
    df["correct"] == False
]

print()
print("=" * 70)
print(
    f"ERROR ANALYSIS "
    f"({len(errors)} errors)"
)
print("=" * 70)

if len(errors) == 0:

    print(
        "No classification errors."
    )

else:

    print(
        errors[
            [
                "image",
                "ground_truth",
                "prediction",
                "confidence",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

print(
    df["prediction"]
    .value_counts()
    .reindex(
        CLASS_NAMES,
        fill_value=0,
    )
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)