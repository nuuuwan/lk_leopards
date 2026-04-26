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

# ── Detection tuning parameters ───────────────────────────────────────────────
# These control which images are accepted or rejected during face detection.
# Adjust them if you're getting too many missed detections (raise thresholds)
# or too many false positives (lower thresholds).

# Which COCO category the detector must match for a detection to be accepted.
# 17 = cat — the closest COCO class to a leopard.  Detections with any other
# label (e.g. dog, giraffe) are discarded even if the score is high.
_ANIMAL_LABELS = frozenset({17})  # 17 = cat

# How confident the detector must be before we trust a body detection.
# Range: 0.0 – 1.0.  Higher → fewer but more reliable detections.
# Lower → more detections, but also more false positives.
# Try 0.7 if too many images are being skipped; try 0.95 to reduce noise.
_BODY_SCORE_THRESHOLD = 0.25

# Maximum allowed width-to-height ratio of the body bounding box.
# A value of 1 means only accept boxes that are taller than they are wide
# (i.e. the leopard is upright / facing the camera).
# Increase (e.g. 1.5) to also include animals photographed at an angle;
# decrease (e.g. 0.8) to be stricter about frontal poses.
_MAX_BODY_RATIO = 1

# How sharp the head crop must be (measured by Laplacian variance).
# Higher → only crisp, well-focused head regions are accepted.
# Lower → blurrier images pass through (useful if your photos are soft).
# Typical range to experiment with: 50–150.
_MIN_LAPLACIAN_VAR = 150

# What fraction of the body box height becomes the head crop.
# 0.55 means the top 55 % of the detected body box is treated as the head.
# Increase slightly (e.g. 0.65) if the crop cuts off the chin;
# decrease (e.g. 0.45) if it includes too much of the neck/chest.
_HEAD_HEIGHT_FRAC = 1

# Extra horizontal padding added on each side of the head crop, as a
# fraction of the body-box width.  0.12 adds 12 % on each side.
# Increase if ears are being clipped; decrease if background is distracting.
_HEAD_PAD = 0.0

# Extra padding added above the head crop (for ears and the top of the head),
# as a fraction of the body-box height.  0.18 adds 18 % above.
# Increase if the top of the head is cut off in saved crops.
_HEAD_PAD_TOP = 0.0

# Whole-image sharpness threshold checked before running the detector.
# Images blurrier than this are skipped immediately (saves time).
# Lower → fewer images skipped up front; higher → faster pipeline overall.
# Set to 0 to disable this early exit entirely.
_GLOBAL_BLUR_THRESHOLD = 0

# Number of face images processed together in one GPU/CPU forward pass
# during the embedding step.  Larger batches are faster on GPU/MPS but
# use more memory.  Reduce to 4 or 8 if you run out of memory.
_EMBED_BATCH_SIZE = 16

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
        # self._device = torch.device(
        #     "cuda"
        #     if torch.cuda.is_available()
        #     else "mps" if torch.backends.mps.is_available() else "cpu"
        # )
        self._device = torch.device("cpu")
        self._model = None
        self._body_detector = None

    # ── Model loading ──────────────────────────────────────────────────────

    def _get_model(self) -> torch.nn.Module:
        if self._model is None:
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT
            m = tv_models.efficientnet_b0(weights=weights)
            m.classifier = torch.nn.Identity()
            m.eval()
            self._model = m.to(self._device)
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
            self._body_detector = m.to(self._device)
        return self._body_detector

    # ── Face detection ─────────────────────────────────────────────────────

    def _compute_frontal_head_bbox(
        self, img: Image.Image
    ) -> tuple[tuple[int, int, int, int], float] | tuple[None, str]:
        """Return ((x1, y1, x2, y2), score) or (None, reason) on rejection.

        Uses FasterRCNN body detection + portrait-ratio gate + Laplacian
        sharpness gate.  The score is the FasterRCNN detection confidence.
        """
        # Cheap early exit: skip heavily blurred images before running
        # the expensive FasterRCNN forward pass.
        gray_full = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        global_var = float(cv2.Laplacian(gray_full, cv2.CV_64F).var())
        if global_var < _GLOBAL_BLUR_THRESHOLD:
            return None, (
                f"image too blurry "
                f"(var={global_var:.1f} < {_GLOBAL_BLUR_THRESHOLD})"
            )

        iw, ih = img.size
        tensor = tv_functional.to_tensor(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            pred = self._get_body_detector()(tensor)[0]

        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]

        n = len(scores)
        # Only consider detections whose label is in the allowed set.
        label_pass = [
            i for i in range(n) if labels[i].item() in _ANIMAL_LABELS
        ]
        # Of those, keep only the ones that also pass the score threshold.
        keep = [
            i for i in label_pass if scores[i].item() >= _BODY_SCORE_THRESHOLD
        ]

        if not keep:
            if not label_pass:
                return None, "no allowed-label detections"
            # Allowed label found but score too low — report its best score.
            best_i = max(label_pass, key=lambda i: scores[i].item())
            best_score = round(scores[best_i].item(), 4)
            return None, (
                f"best allowed-label score {best_score} "
                f"< threshold {_BODY_SCORE_THRESHOLD}"
            )

        best = max(keep, key=lambda i: scores[i].item())
        score = round(scores[best].item(), 4)
        x1, y1, x2, y2 = (c.item() for c in boxes[best])
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(iw), x2), min(float(ih), y2)
        bw, bh = x2 - x1, y2 - y1

        # Gate 1 — portrait ratio: wide box → animal is walking sideways.
        ratio = round(bw / max(bh, 1), 3)
        if ratio > _MAX_BODY_RATIO:
            return None, (
                f"body box too wide " f"(ratio={ratio} > {_MAX_BODY_RATIO})"
            )

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
            return None, (
                f"head too blurry "
                f"(var={lap_var:.1f} < {_MIN_LAPLACIAN_VAR})"
            )

        return (int(hx1), int(hy1), int(hx2), int(hy2)), score

    def detect_frontal_face(self, img: Image.Image) -> Image.Image | None:
        """Return a padded head crop when the leopard is facing the camera.

        Returns ``None`` when no qualifying frontal face is found.
        """
        bbox, _ = self._compute_frontal_head_bbox(img)
        if bbox is None:
            return None
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
        tensor = _TRANSFORM(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = model(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].tolist()

    def build_faces(self, force_rebuild: bool = False):
        """Detect frontal leopard faces in every original image.

        Saves a padded face crop to images/faces/<id>/<image> only when a
        qualifying frontal face is detected.  Already-processed images are
        skipped unless ``force_rebuild=True``.
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
                face_path = self._face_image_path(image_path)
                if not force_rebuild and os.path.exists(face_path):
                    console.log(
                        f"[dim]— {leopard.id}/{image_name}: cached[/dim]"
                    )
                    progress.advance(task_id)
                    continue
                try:
                    img = Image.open(image_path).convert("RGB")
                    face = self.detect_frontal_face(img)
                    if face is None:
                        console.log(
                            f"[dim]— {leopard.id}/{image_name}: no frontal face[/dim]"
                        )
                        skipped += 1
                    else:
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

    def build_face_detected(
        self,
        force_rebuild: bool = False,
        max_images: int | None = None,
    ):
        """Save original images annotated with a bounding box around the face.

        For every original image where a frontal leopard face is detected,
        writes a copy of the full original image with a green rectangle drawn
        around the detected head region to images/face_detected/<id>/<image>.
        Already-processed images are skipped unless ``force_rebuild=True``.
        Pass ``max_images`` to limit processing to the first N images.
        """
        leopards = sorted(Leopard.list_all(), key=lambda lp: lp.id)
        all_images = [
            (leopard, ip)
            for leopard in leopards
            for ip in leopard.image_path_list
        ]
        if max_images is not None:
            all_images = all_images[:max_images]

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
                out_path = self._face_detected_image_path(image_path)
                if not force_rebuild and os.path.exists(out_path):
                    progress.advance(task_id)
                    continue
                try:
                    img = Image.open(image_path).convert("RGB")
                    result = self._compute_frontal_head_bbox(img)
                    bbox, info = result
                    if bbox is None:
                        console.log(
                            f"[dim]— {leopard.id}/{image_name}: "
                            f"{info}[/dim]"
                        )
                        skipped += 1
                    else:
                        x1, y1, x2, y2 = bbox
                        precision = info
                        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        cv2.rectangle(
                            img_cv, (x1, y1), (x2, y2), (0, 255, 0), 3
                        )
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        cv2.imwrite(out_path, img_cv)
                        key = f"{leopard.id}/{image_name}"
                        precision_log[key] = {
                            "score": precision,
                            "bbox": [x1, y1, x2, y2],
                        }
                        console.log(
                            f"[green]✓[/green] {leopard.id}/{image_name}"
                            f" score={precision:.4f}"
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
                f"Min precision: [bold]{_BODY_SCORE_THRESHOLD}[/bold]",
                title="[bold]Complete[/bold]",
            )
        )

    def build_faces_from_detected(
        self,
        force_rebuild: bool = False,
        max_images: int | None = None,
    ):
        """Crop face regions from original images using saved bbox coordinates.

        Reads bbox coordinates from images/face_detected/precision.json
        (written by build_face_detected()), crops that region from the
        original image, and saves to images/faces/<id>/<image>.
        Already-processed images are skipped unless ``force_rebuild=True``.
        """
        precision_path = os.path.join(FACE_DETECTED_DIR, "precision.json")
        if not os.path.exists(precision_path):
            console.print(
                f"[red]✗[/red] {precision_path} not found — "
                "run build_face_detected() first."
            )
            return

        with open(precision_path, encoding="utf-8") as f:
            precision_log: dict = json.load(f)

        entries = list(precision_log.items())
        if max_images is not None:
            entries = entries[:max_images]

        console.print(
            Panel.fit(
                f"[bold cyan]LeopardAI — Extract Faces from Detections"
                f"[/bold cyan]\n"
                f"Source: [dim]{FACE_DETECTED_DIR}/precision.json[/dim]  |  "
                f"Entries: [bold]{len(entries)}[/bold]\n"
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
                "Cropping faces", total=len(entries)
            )
            for key, entry in entries:
                # key is "<leopard_id>/<image_name>"
                leopard_id, image_name = key.split("/", 1)
                original_path = os.path.join(
                    "images", "original", leopard_id, image_name
                )
                face_path = os.path.join(
                    FACES_DIR, leopard_id, image_name
                )
                progress.update(
                    task_id,
                    description=(
                        f"[cyan]{leopard_id}[/cyan] "
                        f"[dim]{image_name}[/dim]"
                    ),
                )
                if not force_rebuild and os.path.exists(face_path):
                    console.log(
                        f"[dim]— {key}: cached[/dim]"
                    )
                    progress.advance(task_id)
                    continue
                try:
                    x1, y1, x2, y2 = entry["bbox"]
                    img = Image.open(original_path).convert("RGB")
                    face = img.crop((x1, y1, x2, y2))
                    os.makedirs(os.path.dirname(face_path), exist_ok=True)
                    face.save(face_path)
                    console.log(
                        f"[green]✓[/green] {key} "
                        f"→ {face.size[0]}×{face.size[1]} → {face_path}"
                    )
                    saved += 1
                except Exception as e:
                    console.log(
                        f"[yellow]⚠[/yellow] Error on {key}: {e}"
                    )
                    skipped += 1
                progress.advance(task_id)

        console.print(
            Panel.fit(
                f"[bold green]✓ Done![/bold green] "
                f"Saved [bold]{saved}[/bold] face crops  |  "
                f"{skipped} errors",
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

    def build_fingerprints(self, force_rebuild: bool = False):
        """Compute embeddings for every face image and write to
        data/finger_prints/<leopard_id>/<image_stem>.json.

        Reads from images/faces/ (produced by build_faces()). Images without
        a face file, or already embedded unless ``force_rebuild=True``, are
        skipped.  Face images are processed in batches of ``_EMBED_BATCH_SIZE``
        for GPU/MPS throughput.
        """
        leopards = sorted(Leopard.list_all(), key=lambda leo: leo.id)
        all_images = [
            (leopard, ip)
            for leopard in leopards
            for ip in leopard.image_path_list
        ]

        console.print("[dim]Loading embedder weights...[/dim]")
        model = self._get_model()
        console.print(
            f"[green]✓[/green] Embedder ready "
            f"([dim]device: {self._device}[/dim])"
        )

        # Collect work: (leopard, image_path, face_path, out_path)
        work: list[tuple] = []
        for leopard, image_path in all_images:
            face_path = self._face_image_path(image_path)
            if not os.path.exists(face_path):
                continue
            out_path = self._fingerprint_path(leopard.id, image_path)
            if not force_rebuild and os.path.exists(out_path):
                continue
            work.append((leopard, image_path, face_path, out_path))

        console.print(
            Panel.fit(
                f"[bold cyan]LeopardAI — Build Fingerprints[/bold cyan]\n"
                f"Source: [dim]{FACES_DIR}/[/dim]\n"
                f"Embedder: [green]{self.MODEL_NAME}[/green]  |  "
                f"Dim: [green]{self.EMBEDDING_DIM}[/green]  |  "
                f"Batch: [green]{_EMBED_BATCH_SIZE}[/green]  |  "
                f"Device: [green]{self._device}[/green]\n"
                f"To embed: [bold]{len(work)}[/bold]  |  "
                f"Output: [dim]{FINGERPRINTS_DIR}/[/dim]",
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

        with progress:
            task_id = progress.add_task("Embedding images", total=len(work))
            for batch_start in range(0, len(work), _EMBED_BATCH_SIZE):
                batch = work[batch_start : batch_start + _EMBED_BATCH_SIZE]
                imgs = [
                    Image.open(face_path).convert("RGB")
                    for _, _, face_path, _ in batch
                ]
                tensors = torch.stack([_TRANSFORM(i) for i in imgs]).to(
                    self._device
                )
                with torch.no_grad():
                    feats = model(tensors)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                for (leopard, image_path, _, out_path), feat in zip(
                    batch, feats
                ):
                    embedding = feat.tolist()
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(embedding, f, indent=4)
                    image_name = os.path.basename(image_path)
                    console.log(
                        f"[green]✓[/green] {leopard.id}/{image_name} "
                        f"→ {len(embedding)}-dim embedding"
                    )
                    progress.advance(task_id)

        console.print(
            Panel.fit(
                f"[bold green]✓ Done![/bold green] Wrote embeddings for "
                f"[bold]{len(work)}[/bold] images to "
                f"[dim]{FINGERPRINTS_DIR}/[/dim]",
                title="[bold]Complete[/bold]",
            )
        )
