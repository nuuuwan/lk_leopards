import json
import os

import cv2
import numpy as np
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
FACE_DETECTED_DIR = os.path.join("images", "face_detected")

# ── FasterRCNN body-detection parameters ─────────────────────────────────────
# COCO animal category IDs 16–25 (cat, dog, horse, sheep, cow, elephant,
# bear, zebra, giraffe, and catch-all); all plausible for a leopard.
_ANIMAL_LABELS = frozenset(range(16, 26))
# Minimum confidence score to keep a detection.
_BODY_SCORE_THRESHOLD = 0.6
# Maximum width/height ratio of the body bounding box.
# A portrait box (ratio < this) means the animal is upright / facing camera.
# A wide box (ratio > this) means the animal is walking sideways.
_MAX_BODY_RATIO = 1.1
# Minimum Laplacian variance of the head crop.
# Low values indicate blur or heavy foliage occlusion.
_MIN_LAPLACIAN_VAR = 150.0
# Laplacian variance reference value used to normalise sharpness to [0, 1].
_SHARPNESS_REF = 600.0
# Minimum composite precision score (0–1) to save a detection.
_MIN_PRECISION = 0.5
# Fraction of the bounding-box height used for the head crop.
_HEAD_HEIGHT_FRAC = 0.55
# Proportional horizontal padding around the head crop.
_HEAD_PAD = 0.12
# Extra vertical padding above the head crop (for ears / top of head).
_HEAD_PAD_TOP = 0.18

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

    Face detection: FasterRCNN body localisation, portrait ratio gate,
                    and Laplacian sharpness gate.
    Embedding:      EfficientNet-B0, 1280-dim L2-normalised vectors.
    """

    BODY_DETECTOR_NAME = "FasterRCNN-MobileNetV3-Large-FPN"
    MODEL_NAME = "EfficientNet-B0"
    EMBEDDING_DIM = 1280

    def __init__(self):
        self._model = None
        self._body_detector = None

    # ── Model loading ──────────────────────────────────────────────────────

    def _get_model(self) -> torch.nn.Module:
        if self._model is None:
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT
            m = tv_models.efficientnet_b0(weights=weights)
            m.classifier = torch.nn.Identity()
            m.eval()
            self._model = m
        return self._model

    def _get_body_detector(self) -> torch.nn.Module:
        if self._body_detector is None:
            weights = (
                tv_models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            )
            m = tv_models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=weights
            )
            m.eval()
            self._body_detector = m
        return self._body_detector

    # ── Face detection ─────────────────────────────────────────────────────

    def _compute_frontal_head_bbox(
        self, img: Image.Image
    ) -> tuple[tuple[int, int, int, int], float] | None:
        """Return ((x1, y1, x2, y2), precision) or None.

        precision is a composite [0, 1] score:
          0.4 * detection_confidence
        + 0.4 * normalised_sharpness   (lap_var / _SHARPNESS_REF, capped at 1)
        + 0.2 * portrait_score         (1 − aspect_ratio / _MAX_BODY_RATIO)

        Returns None when any hard gate fails or precision < _MIN_PRECISION.
        """
        iw, ih = img.size
        tensor = tv_functional.to_tensor(img).unsqueeze(0)
        with torch.no_grad():
            pred = self._get_body_detector()(tensor)[0]

        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]

        keep = [
            i
            for i in range(len(scores))
            if scores[i].item() >= _BODY_SCORE_THRESHOLD
            and labels[i].item() in _ANIMAL_LABELS
        ]
        if not keep:
            return None

        best = max(keep, key=lambda i: scores[i].item())
        detection_score = scores[best].item()
        x1, y1, x2, y2 = (c.item() for c in boxes[best])
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(iw), x2), min(float(ih), y2)
        bw, bh = x2 - x1, y2 - y1

        # Gate 1 — portrait ratio: wide box → animal is walking sideways.
        aspect_ratio = bw / max(bh, 1)
        if aspect_ratio > _MAX_BODY_RATIO:
            return None

        # Head crop: upper portion of the portrait bounding box.
        head_y2 = y1 + min(bw * 1.15, bh * _HEAD_HEIGHT_FRAC)
        px = bw * _HEAD_PAD
        py_top = bh * _HEAD_PAD_TOP
        hx1 = max(0.0, x1 - px)
        hy1 = max(0.0, y1 - py_top)
        hx2 = min(float(iw), x2 + px)
        hy2 = min(float(ih), head_y2 + px)
        head = img.crop((int(hx1), int(hy1), int(hx2), int(hy2)))

        # Gate 2 — sharpness: reject blurry or foliage-occluded crops.
        gray = cv2.cvtColor(np.array(head), cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < _MIN_LAPLACIAN_VAR:
            return None

        # Composite precision score.
        sharpness_norm = min(lap_var / _SHARPNESS_REF, 1.0)
        portrait_score = max(0.0, 1.0 - aspect_ratio / _MAX_BODY_RATIO)
        precision = round(
            0.4 * detection_score
            + 0.4 * sharpness_norm
            + 0.2 * portrait_score,
            4,
        )
        if precision < _MIN_PRECISION:
            return None

        return (int(hx1), int(hy1), int(hx2), int(hy2)), precision

    def detect_frontal_face(self, img: Image.Image) -> Image.Image | None:
        """Return a padded head crop when the leopard is facing the camera.

        Returns ``None`` when no qualifying frontal face is found.
        """
        result = self._compute_frontal_head_bbox(img)
        if result is None:
            return None
        bbox, _ = result
        return img.crop(bbox)

    @staticmethod
    def _face_image_path(original_image_path: str) -> str:
        """Map images/original/<id>/<f> -> images/faces/<id>/<f>."""
        parts = original_image_path.replace("\\", "/").split("/")
        if len(parts) >= 4 and parts[0] == "images" and parts[1] == "original":
            return os.path.join("images", "faces", *parts[2:])
        stem, ext = os.path.splitext(original_image_path)
        return f"{stem}_face{ext}"

    @staticmethod
    def _face_detected_image_path(original_image_path: str) -> str:
        """Map images/original/<id>/<f> -> images/face_detected/<id>/<f>."""
        parts = original_image_path.replace("\\", "/").split("/")
        if len(parts) >= 4 and parts[0] == "images" and parts[1] == "original":
            return os.path.join("images", "face_detected", *parts[2:])
        stem, ext = os.path.splitext(original_image_path)
        return f"{stem}_face_detected{ext}"

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
        """Detect frontal leopard faces in every original image.

        Saves a padded face crop to images/faces/<id>/<image> only when:
        - the cat-face Haar cascade fires on the full image, AND
        - both eyes are confirmed visible in the upper face ROI.

        Images where no qualifying frontal face is detected are skipped
        entirely (no file is written).
        """
        leopards = sorted(Leopard.list_all(), key=lambda l: l.id)
        all_images = [
            (leopard, ip)
            for leopard in leopards
            for ip in leopard.image_path_list
        ]

        console.print(
            Panel.fit(
                f"[bold cyan]LeopardAI — Extract Faces[/bold cyan]\n"
                f"Method: [green]FasterRCNN body detection + portrait ratio + sharpness gate[/green]\n"
                f"Leopards: [bold]{len(leopards)}[/bold]  |  "
                f"Images: [bold]{len(all_images)}[/bold]  |  "
                f"Output: [dim]{FACES_DIR}/[/dim]",
                title="[bold]Starting[/bold]",
            )
        )

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
                "Detecting faces", total=len(all_images)
            )
            for leopard, image_path in all_images:
                image_name = os.path.basename(image_path)
                progress.update(
                    task_id,
                    description=f"[cyan]{leopard.id}[/cyan] [dim]{image_name}[/dim]",
                )
                try:
                    img = Image.open(image_path).convert("RGB")
                    face = self.detect_frontal_face(img)
                    if face is None:
                        console.log(
                            f"[dim]— {leopard.id}/{image_name}: no frontal face[/dim]"
                        )
                        skipped += 1
                    else:
                        face_path = self._face_image_path(image_path)
                        os.makedirs(os.path.dirname(face_path), exist_ok=True)
                        face.save(face_path)
                        console.log(
                            f"[green]✓[/green] {leopard.id}/{image_name} "
                            f"→ face {face.size[0]}×{face.size[1]} → {face_path}"
                        )
                        saved += 1
                except Exception as e:
                    console.log(
                        f"[yellow]⚠[/yellow] Error on {leopard.id}/{image_name}: {e}"
                    )
                    skipped += 1
                progress.advance(task_id)

        console.print(
            Panel.fit(
                f"[bold green]✓ Done![/bold green] "
                f"Saved [bold]{saved}[/bold] face images  |  "
                f"{skipped} skipped (no frontal face detected)",
                title="[bold]Complete[/bold]",
            )
        )

    def build_face_detected(self):
        """Save original images annotated with a bounding box around the face.

        For every original image where a frontal leopard face is detected,
        writes a copy of the full original image with a green rectangle drawn
        around the detected head region to images/face_detected/<id>/<image>.
        Images with no qualifying frontal face are skipped.
        """
        leopards = sorted(Leopard.list_all(), key=lambda lp: lp.id)
        all_images = [
            (leopard, ip)
            for leopard in leopards
            for ip in leopard.image_path_list
        ]

        console.print(
            Panel.fit(
                f"[bold cyan]LeopardAI — Face Detected Bounding Boxes"
                f"[/bold cyan]\n"
                f"Leopards: [bold]{len(leopards)}[/bold]  |  "
                f"Images: [bold]{len(all_images)}[/bold]  |  "
                f"Output: [dim]{FACE_DETECTED_DIR}/[/dim]",
                title="[bold]Starting[/bold]",
            )
        )

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
        precision_log: dict[str, float] = {}
        with progress:
            task_id = progress.add_task(
                "Drawing bounding boxes", total=len(all_images)
            )
            for leopard, image_path in all_images:
                image_name = os.path.basename(image_path)
                progress.update(
                    task_id,
                    description=(
                        f"[cyan]{leopard.id}[/cyan] "
                        f"[dim]{image_name}[/dim]"
                    ),
                )
                try:
                    img = Image.open(image_path).convert("RGB")
                    result = self._compute_frontal_head_bbox(img)
                    if result is None:
                        console.log(
                            f"[dim]— {leopard.id}/{image_name}: "
                            f"no frontal face or precision too low[/dim]"
                        )
                        skipped += 1
                    else:
                        (x1, y1, x2, y2), precision = result
                        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        cv2.rectangle(
                            img_cv, (x1, y1), (x2, y2), (0, 255, 0), 3
                        )
                        out_path = self._face_detected_image_path(image_path)
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        cv2.imwrite(out_path, img_cv)
                        key = f"{leopard.id}/{image_name}"
                        precision_log[key] = precision
                        console.log(
                            f"[green]✓[/green] {leopard.id}/{image_name}"
                            f" precision={precision:.4f}"
                            f" → bbox ({x1},{y1},{x2},{y2})"
                        )
                        saved += 1
                except Exception as e:
                    console.log(
                        f"[yellow]⚠[/yellow] Error on "
                        f"{leopard.id}/{image_name}: {e}"
                    )
                    skipped += 1
                progress.advance(task_id)

        precision_path = os.path.join(FACE_DETECTED_DIR, "precision.json")
        os.makedirs(FACE_DETECTED_DIR, exist_ok=True)
        with open(precision_path, "w", encoding="utf-8") as f:
            json.dump(precision_log, f, indent=4)
        console.log(
            f"[green]✓[/green] Precision scores written → {precision_path}"
        )

        console.print(
            Panel.fit(
                f"[bold green]✓ Done![/bold green] "
                f"Saved [bold]{saved}[/bold] annotated images  |  "
                f"{skipped} skipped (low precision or no face detected)  |  "
                f"Min precision: [bold]{_MIN_PRECISION}[/bold]",
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

        Reads from images/faces/ (produced by build_faces()). Images that
        have no corresponding face file are skipped entirely.
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
                # Only embed if a face image was produced by build_faces().
                face_path = self._face_image_path(image_path)
                if not os.path.exists(face_path):
                    console.log(
                        f"[dim]— {leopard.id}/{image_name}: no face image, skipping[/dim]"
                    )
                    progress.advance(task_id)
                    continue
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                embedding: list[float] = []
                try:
                    embedding = self.embed_image(face_path)
                    console.log(
                        f"[green]✓[/green] {leopard.id}/{image_name} "
                        f"→ {len(embedding)}-dim embedding"
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
