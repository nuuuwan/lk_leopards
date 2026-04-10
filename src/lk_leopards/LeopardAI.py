import json
import os

from deepface import DeepFace

from lk_leopards.Leopard import Leopard

FINGERPRINTS_PATH = os.path.join("data", "finger_prints.json")


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
                print(f"[LeopardAI] Skipping {image_path}: {e}")
        return fingerprints

    def build_fingerprints(self):
        """Compute face fingerprints for every leopard and write to data/finger_prints.json.

        Output format:
        {
            "KLF1": [[...embedding...], ...],
            "KLM2": [...],
            ...
        }
        """
        leopards = sorted(Leopard.list_all(), key=lambda l: l.id)
        data: dict[str, list[list[float]]] = {}
        for leopard in leopards:
            print(f"[LeopardAI] Processing {leopard.id} ({leopard.name})...")
            data[leopard.id] = self.get_face_fingerprints(leopard)
        with open(FINGERPRINTS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"[LeopardAI] Wrote {FINGERPRINTS_PATH}")
