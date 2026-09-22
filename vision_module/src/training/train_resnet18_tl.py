#!/usr/bin/env python3
from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "arduino_3class"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "resnet18_arduino_tl"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "resnet18_arduino_tl"
)

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

CLASS_NAMES = [
    "MEGA",
    "NANO",
    "UNO",
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Transforms
# ============================================================

def get_transforms():

    train_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=15
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])

    eval_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])

    return train_transform, eval_transform


# ============================================================
# Model
# ============================================================

def create_model():

    print("\nLoading pretrained ResNet18...")

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # ImageNet classifier -> 3 Arduino classes
    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        len(CLASS_NAMES),
    )

    return model


# ============================================================
# Train
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    loss = (
        running_loss / total
    )

    accuracy = (
        correct / total
    )

    return loss, accuracy


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

    loss = (
        running_loss / total
    )

    accuracy = (
        correct / total
    )

    return (
        loss,
        accuracy,
        all_labels,
        all_predictions,
    )


# ============================================================
# Test
# ============================================================

@torch.no_grad()
def evaluate_test(
    model,
    loader,
    device,
):

    model.eval()

    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    confusion = confusion_matrix(
        all_labels,
        all_predictions,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": confusion.tolist(),
        "num_samples": len(all_labels),
    }


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print("RESNET18 - ARDUINO 3-CLASS CLASSIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"\nPrepared dataset not found:\n"
            f"{DATASET_DIR}\n\n"
            f"Run first:\n"
            f"python "
            f"src/datasets/"
            f"prepare_arduino_3class.py"
        )

    train_dir = (
        DATASET_DIR / "train"
    )

    val_dir = (
        DATASET_DIR / "val"
    )

    test_dir = (
        DATASET_DIR / "test"
    )

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    (
        train_transform,
        eval_transform,
    ) = get_transforms()

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=eval_transform,
    )

    print("\nClasses:")

    for name, index in (
        train_dataset.class_to_idx.items()
    ):

        print(
            f"  {name:5s} -> {index}"
        )

    print("\nDataset sizes:")

    print(
        f"  Train: {len(train_dataset)}"
    )

    print(
        f"  Val:   {len(val_dataset)}"
    )

    print(
        f"  Test:  {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()

    model = model.to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # LR scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / "best_resnet18.pt"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_accuracy = -1.0

    history = []

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
        )

        (
            val_loss,
            val_accuracy,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(
            val_accuracy
        )

        learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": learning_rate,
        })

        print(
            f"Epoch "
            f"{epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Train Acc: "
            f"{train_accuracy:.4f} | "
            f"Val Loss: "
            f"{val_loss:.4f} | "
            f"Val Acc: "
            f"{val_accuracy:.4f}"
        )

        # Save best checkpoint
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_accuracy":
                        val_accuracy,
                    "class_names":
                        CLASS_NAMES,
                },
                checkpoint_path,
            )

            print(
                f"  -> Best model saved "
                f"(val_acc="
                f"{val_accuracy:.4f})"
            )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = (
        RESULTS_DIR
        / "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best validation accuracy: "
        f"{checkpoint['val_accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST")
    print("=" * 70)

    test_metrics = evaluate_test(
        model,
        test_loader,
        device,
    )

    print(
        f"\nAccuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("\nConfusion matrix:")

    print(
        np.array(
            test_metrics[
                "confusion_matrix"
            ]
        )
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification report:")

    report = test_metrics[
        "classification_report"
    ]

    for class_name in CLASS_NAMES:

        metrics = report[class_name]

        print(
            f"{class_name:5s} | "
            f"Precision: "
            f"{metrics['precision']:.4f} | "
            f"Recall: "
            f"{metrics['recall']:.4f} | "
            f"F1: "
            f"{metrics['f1-score']:.4f}"
        )

    # --------------------------------------------------------
    # Save test metrics
    # --------------------------------------------------------

    metrics_path = (
        RESULTS_DIR
        / "test_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            test_metrics,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"\nCheckpoint:"
        f"\n  {checkpoint_path}"
    )

    print(
        f"\nTraining history:"
        f"\n  {history_path}"
    )

    print(
        f"\nTest metrics:"
        f"\n  {metrics_path}"
    )


if __name__ == "__main__":
    main()