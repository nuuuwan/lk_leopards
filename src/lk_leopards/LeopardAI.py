import json
import os

import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           SpinnerColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)

from lk_leopards.Leopard import Leopard

FINGERPRINTS_DIR = os.path.join("data", "finger_prints")

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
    EMBEDDING_DIM = 1280

    def __init__(self):
        self._model = None

    def _get_model(self) -> torch.nn.Module:
        if self._model is None:
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT
            m = tv_models.efficientnet_b0(weights=weights)
            m.classifier = torch.nn.Identity()
            m.eval()
            self._model = m
        return self._model

    def embed_image(self, image_path: str) -> list[float]:
        """Return a normalised 1280-dim embedding for a single image."""
        model = self._get_model()
        img = Image.open(image_path).convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].tolist()

    def get_face_fingerprints(self, leopard: Leopard) -> list[list[float]]:
        """Return one embedding per image for the given leopard."""
        fingerprints = []
        for image_path in leopard.image_path_list:
            try:
                fingerprints.append(self.embed_image(image_path))
            except Exception as e:
                console.print(
                    f"[yellow]⚠ Skipping {image_path}:[/yellow] {e}"
                )
        return fingerprints

    @staticmethod
    def _fingerprint_path(leopard_id: str, image_path: str) -> str:
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(
            FINGERPRINTS_DIR, leopard_id, f"{image_stem}.json"
        )

    def build_fingerprints(self):
        """Compute embeddings for every leopard image and write to
        data/finger_prints/<leopard_id>/<image_stem>.json.
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
                f"Model: [green]{self.MODEL_NAME}[/green]  |  "
                f"Embedding dim: [green]{self.EMBEDDING_DIM}[/green]\n"
                f"Leopards: [bold]{len(leopards)}[/bold]  |  "
                f"Images: [bold]{len(all_images)}[/bold]  |  "
                f"Output: [dim]{FINGERPRINTS_DIR}/[/dim]",
                title="[bold]Starting[/bold]",
            )
        )

        console.print("[dim]Loading model weights...[/dim]")
        self._get_model()
        console.print("[green]✓[/green] Model ready.")

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
                try:
                    embedding = self.embed_image(image_path)
                    console.log(
                        f"[green]✓[/green] {leopard.id}/{image_name} "
                        f"→ {len(embedding)}-dim embedding"
                    )
                except Exception as e:
                    console.log(
                        f"[yellow]⚠[/yellow] Skipping {
                            leopard.id}/{image_name}: {e}"
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
