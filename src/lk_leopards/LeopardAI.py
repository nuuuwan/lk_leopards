import json
import os

import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional as tv_functional
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from lk_leopards.Leopard import Leopard

FINGERPRINTS_DIR = os.path.join("data", "finger_prints")
FACES_DIR = os.path.join("images", "faces")

# COCO label indices that represent animals (bird=16 … giraffe=25).
# Leopards are most commonly detected under label 17 (cat).
_ANIMAL_LABELS = frozenset(range(16, 26))
_DETECT_THRESHOLD = 0.3
# Fraction of the bounding-box height retained as the "face" / head region.
_FACE_HEIGHT_RATIO = 0.55

console = Console()

_TRANSFORM = tv_transforms.Compose(
    [
        tv_transforms.Resize(256),
        tv_transforms.CenterCrop(224),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class LeopardAI:
    """Image-embedding model for leopard identification.

    Uses EfficientNet-B0 (pretrained on ImageNet) as a feature extractor.
    Produces a 1280-dimensional L2-normalised embedding per image.
    PyTorch-only: no TensorFlow dependency.
    """

    MODEL_NAME = "EfficientNet-B0"
    DETECTOR_NAME = "FasterRCNN-MobileNetV3-Large-FPN"
    EMBEDDING_DIM = 1280

    def __init__(self):
        self._model = None
        self._detector = None

    def _get_model(self) -> torch.nn.Module:
        if self._model is None:
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT
            m = tv_models.efficientnet_b0(weights=weights)
            m.classifier = torch.nn.Identity()
            m.eval()
            self._model = m
        return self._model

    def _get_detector(self) -> torch.nn.Module:
        if self._detector is None:
            weights = (
                tv_models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            )
            m = tv_models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=weights
            )
            m.eval()
            self._detector = m
        return self._detector

    def _crop_face(self, img: Image.Image) -> Image.Image:
        """Detect the leopard and return a crop of the head/face region.

        Uses FasterRCNN to locate the animal bounding box, then retains only
        the top FACE_HEIGHT_RATIO fraction (the head) of that box.
        Falls back to an upper-center crop if no animal is detected.
        """
        detector = self._get_detector()
        # to_tensor: HxWxC PIL → CxHxW float [0,1] as expected by the detector
        tensor = tv_functional.to_tensor(img).unsqueeze(0)
        with torch.no_grad():
            preds = detector(tensor)[0]

        boxes = preds["boxes"]  # (N, 4)
        scores = preds["scores"]  # (N,)
        labels = preds["labels"]  # (N,)

        # Keep animal detections above the confidence threshold
        keep = [
            i
            for i in range(len(scores))
            if scores[i].item() >= _DETECT_THRESHOLD
            and labels[i].item() in _ANIMAL_LABELS
        ]

        w, h = img.size
        if keep:
            best = max(keep, key=lambda i: scores[i].item())
            x1, y1, x2, y2 = (c.item() for c in boxes[best])
            # Clamp to image bounds
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            # Keep only the top fraction as the face/head
            face_y2 = y1 + (y2 - y1) * _FACE_HEIGHT_RATIO
            crop = img.crop((int(x1), int(y1), int(x2), int(face_y2)))
        else:
            # Fallback: upper-centre region of the image
            crop = img.crop(
                (int(w * 0.1), int(h * 0.05), int(w * 0.9), int(h * 0.6))
            )

        # Safety: if the crop is too small for the transform, use the full image
        cw, ch = crop.size
        if cw < 32 or ch < 32:
            return img
        return crop

    @staticmethod
    def _face_image_path(original_image_path: str) -> str:
        """Derive the face image path for a given original image path.

        images/original/KLF0001/image_1.jpeg -> images/faces/KLF0001/image_1.jpeg
        """
        parts = original_image_path.replace("\\", "/").split("/")
        # parts: ['images', 'original', '<id>', '<filename>']
        if len(parts) >= 4 and parts[0] == "images" and parts[1] == "original":
            return os.path.join("images", "faces", *parts[2:])
        # Fallback: put alongside the original with a _face suffix
        stem, ext = os.path.splitext(original_image_path)
        return f"{stem}_face{ext}"

    def embed_image(self, image_path: str) -> list[float]:
        """Return a normalised 1280-dim embedding for the given image.

        Expects an already-cropped face image for best results.
        """
        model = self._get_model()
        img = Image.open(image_path).convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].tolist()

    def build_faces(self):
        """Extract face crops from all original images and save to images/faces/."""
        leopards = sorted(Leopard.list_all(), key=lambda l: l.id)
        all_images = [
            (leopard, ip)
            for leopard in leopards
            for ip in leopard.image_path_list
        ]

        console.print(
            Panel.fit(
                f"[bold cyan]LeopardAI — Extract Faces[/bold cyan]\n"
                f"Detector: [green]{self.DETECTOR_NAME}[/green]\n"
                f"Leopards: [bold]{len(leopards)}[/bold]  |  "
                f"Images: [bold]{len(all_images)}[/bold]  |  "
                f"Output: [dim]{FACES_DIR}/[/dim]",
                title="[bold]Starting[/bold]",
            )
        )

        console.print("[dim]Loading detector weights...[/dim]")
        self._get_detector()
        console.print("[green]✓[/green] Detector ready.")

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        saved = 0
        skipped = 0
        with progress:
            task_id = progress.add_task(
                "Extracting faces", total=len(all_images)
            )
            for leopard, image_path in all_images:
                image_name = os.path.basename(image_path)
                progress.update(
                    task_id,
                    description=f"[cyan]{leopard.id}[/cyan] [dim]{image_name}[/dim]",
                )
                face_path = self._face_image_path(image_path)
                os.makedirs(os.path.dirname(face_path), exist_ok=True)
                try:
                    img = Image.open(image_path).convert("RGB")
                    face = self._crop_face(img)
                    face.save(face_path)
                    console.log(
                        f"[green]✓[/green] {leopard.id}/{image_name} "
                        f"→ face {face.size[0]}×{face.size[1]} → {face_path}"
                    )
                    saved += 1
                except Exception as e:
                    console.log(
                        f"[yellow]⚠[/yellow] Skipping {leopard.id}/{image_name}: {e}"
                    )
                    skipped += 1
                progress.advance(task_id)

        console.print(
            Panel.fit(
                f"[bold green]✓ Done![/bold green] "
                f"Saved [bold]{saved}[/bold] face images to [dim]{FACES_DIR}/[/dim]"
                + (
                    f"  ([yellow]{skipped} skipped[/yellow])"
                    if skipped
                    else ""
                ),
                title="[bold]Complete[/bold]",
            )
        )

    def get_face_fingerprints(self, leopard: Leopard) -> list[list[float]]:
        """Return one embedding per image for the given leopard."""
        fingerprints = []
        for image_path in leopard.image_path_list:
            try:
                fingerprints.append(self.embed_image(image_path))
            except Exception as e:
                console.print(f"[yellow]⚠ Skipping {image_path}:[/yellow] {e}")
        return fingerprints

    @staticmethod
    def _fingerprint_path(leopard_id: str, image_path: str) -> str:
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(FINGERPRINTS_DIR, leopard_id, f"{image_stem}.json")

    def build_fingerprints(self):
        """Compute embeddings for every face image and write to
        data/finger_prints/<leopard_id>/<image_stem>.json.

        Reads from images/faces/ (produced by build_faces()). Falls back to
        the original image path if the face image does not exist yet.
        """
        leopards = sorted(Leopard.list_all(), key=lambda l: l.id)
        all_images = [
            (leopard, ip)
            for leopard in leopards
            for ip in leopard.image_path_list
        ]

        console.print(
            Panel.fit(
                f"[bold cyan]LeopardAI — Build Fingerprints[/bold cyan]\n"
                f"Source: [dim]{FACES_DIR}/[/dim]\n"
                f"Embedder: [green]{self.MODEL_NAME}[/green]  |  "
                f"Embedding dim: [green]{self.EMBEDDING_DIM}[/green]\n"
                f"Leopards: [bold]{len(leopards)}[/bold]  |  "
                f"Images: [bold]{len(all_images)}[/bold]  |  "
                f"Output: [dim]{FINGERPRINTS_DIR}/[/dim]",
                title="[bold]Starting[/bold]",
            )
        )

        console.print("[dim]Loading embedder weights...[/dim]")
        self._get_model()
        console.print("[green]✓[/green] Embedder ready.")

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            task_id = progress.add_task(
                "Embedding images", total=len(all_images)
            )
            for leopard, image_path in all_images:
                image_name = os.path.basename(image_path)
                progress.update(
                    task_id,
                    description=f"[cyan]{
                        leopard.id}[/cyan] [dim]{image_name}[/dim]",
                )
                out_path = self._fingerprint_path(leopard.id, image_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                embedding: list[float] = []
                # Use the pre-extracted face image; fall back to original if missing
                face_path = self._face_image_path(image_path)
                src_path = (
                    face_path if os.path.exists(face_path) else image_path
                )
                try:
                    embedding = self.embed_image(src_path)
                    src_label = "face" if src_path == face_path else "original"
                    console.log(
                        f"[green]✓[/green] {leopard.id}/{image_name} "
                        f"({src_label}) → {len(embedding)}-dim embedding"
                    )
                except Exception as e:
                    console.log(
                        f"[yellow]⚠[/yellow] Skipping {leopard.id}/{image_name}: {e}"
                    )
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(embedding, f, indent=4)
                progress.advance(task_id)

        console.print(
            Panel.fit(
                f"[bold green]✓ Done![/bold green] Wrote embeddings for "
                f"[bold]{len(all_images)}[/bold] images to "
                f"[dim]{FINGERPRINTS_DIR}/[/dim]",
                title="[bold]Complete[/bold]",
            )
        )
