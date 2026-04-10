import json
import os

from deepface import DeepFace
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

console = Console()


class LeopardAI:
    MODEL_NAME = "Facenet512"
    DETECTOR_BACKEND = "retinaface"

    def get_face_fingerprints(self, leopard: Leopard) -> list[list[float]]:
        """Return face embedding vectors for all images of a leopard.

        Each image may yield zero or more face embeddings. All found embeddings
        are collected and returned as a flat list.  enforce_detection=False is
        used so that a result is still produced when the detector cannot locate
        a face region (e.g. partial or distant shots).
        """
        fingerprints: list[list[float]] = []
        for image_path in leopard.image_path_list:
            try:
                results = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.MODEL_NAME,
                    detector_backend=self.DETECTOR_BACKEND,
                    enforce_detection=False,
                )
                for result in results:
                    fingerprints.append(result["embedding"])
            except Exception as e:
                console.print(f"[yellow]⚠ Skipping {image_path}:[/yellow] {e}")
        return fingerprints

    def _fingerprint_path(self, leopard_id: str, image_path: str) -> str:
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(FINGERPRINTS_DIR, leopard_id, f"{image_stem}.json")

    def build_fingerprints(self):
        """Compute face fingerprints per image and write to
        data/finger_prints/<leopard_id>/<image_stem>.json.

        Each file contains a list of 512-dim embedding vectors (one per
        detected face in that image).
        """
        leopards = sorted(Leopard.list_all(), key=lambda l: l.id)
        all_images = [
            (leopard, image_path)
            for leopard in leopards
            for image_path in leopard.image_path_list
        ]

        console.print(
            Panel.fit(
                f"[bold cyan]LeopardAI — Build Fingerprints[/bold cyan]\n"
                f"Model: [green]{self.MODEL_NAME}[/green]  |  "
                f"Detector: [green]{self.DETECTOR_BACKEND}[/green]\n"
                f"Leopards: [bold]{len(leopards)}[/bold]  |  "
                f"Images: [bold]{len(all_images)}[/bold]  |  "
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
            task = progress.add_task(
                "Fingerprinting images", total=len(all_images)
            )
            for leopard, image_path in all_images:
                image_name = os.path.basename(image_path)
                progress.update(
                    task,
                    description=f"[cyan]{leopard.id}[/cyan] [dim]{image_name}[/dim]",
                )
                out_path = self._fingerprint_path(leopard.id, image_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                embeddings: list[list[float]] = []
                try:
                    results = DeepFace.represent(
                        img_path=image_path,
                        model_name=self.MODEL_NAME,
                        detector_backend=self.DETECTOR_BACKEND,
                        enforce_detection=False,
                    )
                    embeddings = [r["embedding"] for r in results]
                    console.log(
                        f"[green]✓[/green] {leopard.id}/{image_name} → {len(embeddings)} embedding(s)"
                    )
                except Exception as e:
                    console.log(
                        f"[yellow]⚠[/yellow] Skipping {leopard.id}/{image_name}: {e}"
                    )
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(embeddings, f, indent=4)
                progress.advance(task)

        console.print(
            Panel.fit(
                f"[bold green]✓ Done![/bold green] Wrote fingerprints for [bold]{len(all_images)}[/bold] images to [dim]{FINGERPRINTS_DIR}/[/dim]",
                title="[bold]Complete[/bold]",
            )
        )
