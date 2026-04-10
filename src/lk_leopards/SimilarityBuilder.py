import json
import os

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

FINGERPRINTS_DIR = os.path.join("data", "finger_prints")
SIMILARITY_PATH = os.path.join("data", "similarity.json")

console = Console()


class SimilarityBuilder:
    """Compute pairwise cosine similarity between all leopard image fingerprints.

    Embeddings are already L2-normalised, so cosine similarity == dot product.
    For each image, the TOP_N most similar *other* images are stored.
    """

    TOP_N = 5

    def _load_all_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Return (keys, matrix) where keys[i] == 'KLF0001/image_1' and
        matrix[i] is the 1280-dim embedding for that image."""
        keys: list[str] = []
        vectors: list[list[float]] = []

        for leopard_id in sorted(os.listdir(FINGERPRINTS_DIR)):
            leopard_dir = os.path.join(FINGERPRINTS_DIR, leopard_id)
            if not os.path.isdir(leopard_dir):
                continue
            for fname in sorted(os.listdir(leopard_dir)):
                if not fname.endswith(".json"):
                    continue
                key = f"{leopard_id}/{os.path.splitext(fname)[0]}"
                fp = os.path.join(leopard_dir, fname)
                with open(fp, encoding="utf-8") as f:
                    vec = json.load(f)
                keys.append(key)
                vectors.append(vec)

        mat = np.array(vectors, dtype=np.float32)
        return keys, mat

    def build(self) -> dict:
        """Compute top-5 most similar images for every fingerprint.

        Returns a dict keyed by 'leopard_id/image_stem', each value being a
        list of TOP_N dicts: {"image": <key>, "score": <float>}.
        """
        console.print("[dim]Loading fingerprint files...[/dim]")
        keys, mat = self._load_all_embeddings()
        n = len(keys)
        console.print(
            f"[green]✓[/green] Loaded [bold]{n}[/bold] embeddings "
            f"({mat.shape[1]}-dim each)."
        )

        console.print(
            "[dim]Computing pairwise cosine similarity matrix...[/dim]"
        )
        # Embeddings are already L2-normalised → dot product == cosine similarity
        sim: np.ndarray = mat @ mat.T  # (N × N)
        console.print(f"[green]✓[/green] Similarity matrix: {n}×{n}")

        result: dict = {}
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        with progress:
            task_id = progress.add_task("Ranking similarities", total=n)
            for i, key in enumerate(keys):
                scores = sim[i]
                # Exclude self; sort descending by score
                sorted_idx = np.argsort(-scores)
                top: list[dict] = []
                for j in sorted_idx:
                    if j == i:
                        continue
                    top.append(
                        {
                            "image": keys[j],
                            "score": round(float(scores[j]), 6),
                        }
                    )
                    if len(top) == self.TOP_N:
                        break
                result[key] = top
                progress.advance(task_id)

        return result

    def write(self) -> dict:
        """Build similarity data and write to data/similarity.json."""
        console.print(
            Panel.fit(
                "[bold cyan]SimilarityBuilder[/bold cyan]\n"
                f"Top-N: [green]{self.TOP_N}[/green]  |  "
                f"Output: [dim]{SIMILARITY_PATH}[/dim]",
                title="[bold]Starting[/bold]",
            )
        )

        result = self.build()

        os.makedirs(os.path.dirname(SIMILARITY_PATH), exist_ok=True)
        with open(SIMILARITY_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        console.print(
            Panel.fit(
                f"[bold green]✓ Done![/bold green] Wrote similarities for "
                f"[bold]{len(result)}[/bold] images to [dim]{SIMILARITY_PATH}[/dim]",
                title="[bold]Complete[/bold]",
            )
        )
        return result
