from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


CLASS_NAMES = ["MEGA", "NANO", "UNO"]

DEFAULT_MODEL_PATH = (
    Path("models")
    / "vision"
    / "resnet18_target_adaptation"
    / "best.pt"
)


class ArduinoClassifier:

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            len(CLASS_NAMES),
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        # Compatible with checkpoints saved either directly
        # or inside a {"model_state_dict": ...} dictionary.
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")

        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)

        confidence, predicted_idx = torch.max(probabilities, dim=1)

        board = CLASS_NAMES[predicted_idx.item()]
        confidence = confidence.item()

        return {
            "board": board,
            "confidence": confidence,
        }