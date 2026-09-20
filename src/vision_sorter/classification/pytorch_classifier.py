"""Optional TorchScript crop classifier with an explicit input contract.

The supplied model must expect RGB float32 NCHW in [0, 1], resized to size,
and return one logit per supplied label. Only load trusted model artifacts.
"""
from pathlib import Path
import cv2
from vision_sorter.image_processing.reader import Image
from vision_sorter.image_processing.preprocessing import validate


class PyTorchClassifier:
    def __init__(self, model_path: Path, labels: list[str], size: int = 224):
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if not labels or len(set(labels)) != len(labels) or size < 1:
            raise ValueError("Unique class labels and a positive input size are required")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Optional PyTorch dependency missing; install .[ml] deliberately") from exc
        self.torch = torch
        self.model = torch.jit.load(str(model_path), map_location="cpu").eval()
        self.labels = list(labels)
        self.size = size

    def predict(self, crop: Image) -> tuple[str, float]:
        validate(crop)
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError("Classifier expects a BGR crop with three channels")
        rgb = cv2.cvtColor(cv2.resize(crop, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
        with self.torch.inference_mode():
            logits = self.model(tensor)
            if tuple(logits.shape) != (1, len(self.labels)) or not self.torch.isfinite(logits).all():
                raise ValueError("Model output must be finite logits with shape [1, number of labels]")
            probability = self.torch.softmax(logits, dim=1)[0]
        index = int(probability.argmax().item())
        return self.labels[index], float(probability[index].item())
