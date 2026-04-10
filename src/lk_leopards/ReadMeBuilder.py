import json
import os

from lk_leopards.Leopard import Leopard

README_PATH = "README.md"
SIMILARITY_PATH = os.path.join("data", "similarity.json")


class ReadMeBuilder:
    def __init__(self):
        self.leopards = sorted(Leopard.list_all(), key=lambda l: l.id)

    def _summary_section(self) -> str:
        total = len(self.leopards)
        males = sum(1 for l in self.leopards if l.gender == "M")
        females = sum(1 for l in self.leopards if l.gender == "F")
        all_zones = sorted({z for l in self.leopards for z in l.zone_list})

        lines = [
            "## Summary",
            "",
            f"| Metric | Count |",
            f"| --- | --- |",
            f"| Total leopards | {total} |",
            f"| Males | {males} |",
            f"| Females | {females} |",
            f"| Zones | {', '.join(all_zones)} |",
        ]
        return "\n".join(lines)

    def _leopard_row(self, leopard: Leopard) -> str:
        first_image = (
            f'<img src="{leopard.image_path_list[0]}" width="100"/>'
            if leopard.image_path_list
            else ""
        )
        gender_label = "Male" if leopard.gender == "M" else "Female"
        zones = ", ".join(leopard.zone_list)
        return (
            f"| {leopard.id} | {leopard.name} | {gender_label} "
            f"| {zones} | {leopard.date_first_seen} "
            f"| {leopard.date_last_seen} | {first_image} |"
        )

    def _similarity_section(self) -> str:
        """Build a summary of the top cross-leopard similar image pairs."""
        if not os.path.exists(SIMILARITY_PATH):
            return ""

        with open(SIMILARITY_PATH, encoding="utf-8") as f:
            similarity: dict = json.load(f)

        # Collect unique cross-leopard pairs (different leopard IDs) with
        # highest scores
        seen: set[frozenset] = set()
        pairs: list[tuple[float, str, str]] = []
        for src_key, matches in similarity.items():
            src_leopard = src_key.split("/")[0]
            for match in matches:
                tgt_key: str = match["image"]
                tgt_leopard = tgt_key.split("/")[0]
                if src_leopard == tgt_leopard:
                    continue
                pair = frozenset([src_key, tgt_key])
                if pair in seen:
                    continue
                seen.add(pair)
                pairs.append((match["score"], src_key, tgt_key))

        pairs.sort(key=lambda x: -x[0])
        top_pairs = pairs[:10]

        lines = [
            "## Top Similar Pairs (Cross-Leopard)",
            "",
            "The 10 image pairs with the highest cosine similarity score "
            "that belong to *different* leopards.",
            "",
            "| Score | Image A | Image B |",
            "| --- | --- | --- |",
        ]
        for score, a, b in top_pairs:
            lines.append(f"| {score:.4f} | {a} | {b} |")

        return "\n".join(lines)

    def _leopards_table_section(self) -> str:
        header = (
            "## Leopards\n"
            "\n"
            "| ID | Name | Gender | Zone(s) | First Seen | Last Seen | Image |\n"
            "| --- | --- | --- | --- | --- | --- | --- |"
        )
        rows = [self._leopard_row(l) for l in self.leopards]
        return "\n".join([header] + rows)

    def build(self) -> str:
        similarity_section = self._similarity_section()
        sections = [
            "# lk_leopards",
            "",
            "Catalogue of leopards observed in Kumana National Park, Sri Lanka.",
            "",
            self._summary_section(),
            "",
        ]
        if similarity_section:
            sections += [similarity_section, ""]
        sections += [
            self._leopards_table_section(),
            "",
        ]
        return "\n".join(sections)

    def write(self):
        content = self.build()
        with open(README_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Wrote {README_PATH}")
