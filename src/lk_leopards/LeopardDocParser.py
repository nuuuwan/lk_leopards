import os
import re

import fitz  # PyMuPDF

from lk_leopards.Leopard import Leopard


class LeopardDocParser:
    PDF_PATH = os.path.join("docs", "Leopards of Kumana - Field Guide.pdf")
    FIRST_PAGE = 16  # 1-indexed, inclusive
    LAST_PAGE = 103  # 1-indexed, inclusive
    FIRST_TABLE_PAGE = 13  # 1-indexed, pages with first/last seen table
    LAST_TABLE_PAGE = 15

    RE_TITLE = re.compile(r"^\d+\.\s+(KL[FM]\d+)\s*[–\-]\s*(.*)")
    RE_ZONE = re.compile(r"^Zone\s*[–\-]\s*(.*)")
    RE_DATE_TOKEN = re.compile(r"(?:\*\*|\d{2})/\d{2}/\d{4}|\d{4}")
    RE_DATE_FULL = re.compile(r"^(\*\*|\d{2})/(\d{2})/(\d{4})$")
    RE_SEQ_NUM = re.compile(r"^\d{1,2}$")
    RE_RAW_ID = re.compile(r"(KL[FM])(\d+)")
    TABLE_HEADER_LINES = {
        "#",
        "ID",
        "Zone",
        "First Seen",
        "Last Seen",
        "Mother",
    }

    def __init__(self, pdf_path: str = None):
        self.pdf_path = pdf_path or self.PDF_PATH

    def parse(self) -> list[Leopard]:
        os.makedirs(os.path.join("data", "leopards"), exist_ok=True)
        doc = fitz.open(self.pdf_path)
        dates = self._parse_date_table(doc)
        leopards = []
        for page_num in range(self.FIRST_PAGE - 1, self.LAST_PAGE):
            leopard = self._parse_page(doc[page_num], doc)
            if leopard is not None:
                first_seen, last_seen = dates.get(leopard.id, ("", ""))
                leopard.date_first_seen = first_seen
                leopard.date_last_seen = last_seen
                leopard.write()
                leopards.append(leopard)
        return leopards

    def _pad_id(self, raw_id: str) -> str:
        """Pad the numeric part of a leopard ID to 4 digits: KLF1 → KLF0001."""
        return self.RE_RAW_ID.sub(
            lambda m: m.group(1) + m.group(2).zfill(4), raw_id
        )

    def _parse_date_table(self, doc) -> dict[str, tuple[str, str]]:
        """Parse pages 13–15 and return {leopard_id: (first_seen, last_seen)}."""
        lines = []
        for page_num in range(
            self.FIRST_TABLE_PAGE - 1, self.LAST_TABLE_PAGE
        ):
            for line in doc[page_num].get_text().split("\n"):
                line = line.strip()
                if (
                    line
                    and line not in self.TABLE_HEADER_LINES
                    and not line.startswith("Leopards of Kumana")
                    and not line.startswith("P a g e")
                ):
                    lines.append(line)

        dates: dict[str, tuple[str, str]] = {}
        i = 0
        while i < len(lines):
            if self.RE_SEQ_NUM.match(lines[i]):
                # Collect lines belonging to this record
                j = i + 1
                while j < len(lines) and not self.RE_SEQ_NUM.match(lines[j]):
                    j += 1
                record_lines = lines[i + 1: j]

                # Extract leopard ID
                leopard_id = None
                for rl in record_lines:
                    m = re.search(r"(KL[FM]\d+)", rl)
                    if m:
                        leopard_id = self._pad_id(m.group(1))
                        break

                if leopard_id:
                    tokens = self.RE_DATE_TOKEN.findall(
                        " ".join(record_lines)
                    )
                    dates[leopard_id] = (
                        (
                            self._format_date(tokens[0])
                            if len(tokens) > 0
                            else ""
                        ),
                        (
                            self._format_date(tokens[1])
                            if len(tokens) > 1
                            else ""
                        ),
                    )
                i = j
            else:
                i += 1

        return dates

    def _format_date(self, token: str) -> str:
        """Convert dd/mm/yyyy (or **/mm/yyyy) to yyyy-mm-dd; year-only stays as-is."""
        m = self.RE_DATE_FULL.match(token)
        if m:
            dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
            return f"{yyyy}-{mm}-{dd}"
        return token  # year-only or unrecognised

    def _parse_page(self, page, doc) -> Leopard | None:
        lines = [line.strip() for line in page.get_text().split("\n")]

        # --- Find title line: "N. KL[FM]ID – Name (Sinhala)" ---
        leopard_id = None
        name = None
        for line in lines:
            m = self.RE_TITLE.match(line)
            if m:
                leopard_id = self._pad_id(m.group(1))
                name_raw = m.group(2).strip()
                name = re.split(r"\s*\(", name_raw)[0].strip()
                break

        if not leopard_id:
            return None

        gender = "F" if leopard_id.startswith("KLF") else "M"

        # --- Locate the "Correlation" table header ---
        corr_idx = next(
            (i for i, line in enumerate(lines) if line == "Correlation"), None
        )
        if corr_idx is None:
            return None

        table_lines = [line for line in lines[corr_idx + 1:] if line]

        # --- Find "Zone – X" line ---
        zone_idx = next(
            (
                i
                for i, line in enumerate(table_lines)
                if self.RE_ZONE.match(line)
            ),
            None,
        )
        if zone_idx is None:
            return None

        zone_list = [
            z.strip()
            for z in self.RE_ZONE.match(table_lines[zone_idx])
            .group(1)
            .split(",")
            if z.strip()
        ]

        # --- Location: lines between end-of-ID-cell and Zone ---
        # The ID cell ends after the Sinhala name (first line containing "(")
        loc_lines = []
        in_id_cell = True
        for line in table_lines[:zone_idx]:
            if in_id_cell:
                if "(" in line:
                    in_id_cell = False
            else:
                loc_lines.append(line)
        location_details = " ".join(loc_lines)

        # --- Correlation: lines after Zone until "Get more details" ---
        corr_lines = []
        for line in table_lines[zone_idx + 1:]:
            if line.startswith("Get more details"):
                break
            corr_lines.append(line)
        correlation_details = "\n".join(corr_lines)

        # --- Mother ID from "Cub of KLxNN" ---
        mother_id = ""
        for line in corr_lines:
            m = re.search(r"Cub of\s+(KL[FM]\d+)", line)
            if m:
                mother_id = self._pad_id(m.group(1))
                break

        # --- Extract and save images ---
        image_path_list = self._extract_images(page, leopard_id, doc)

        return Leopard(
            id=leopard_id,
            name=name,
            gender=gender,
            location_details=location_details,
            correlation_details=correlation_details,
            image_path_list=image_path_list,
            zone_list=zone_list,
            date_first_seen="",
            date_last_seen="",
            mother_id=mother_id,
        )

    def _extract_images(self, page, leopard_id: str, doc) -> list[str]:
        image_dir = os.path.join("images", leopard_id)
        os.makedirs(image_dir, exist_ok=True)
        image_paths = []
        for i, img_ref in enumerate(page.get_images(full=True)):
            xref = img_ref[0]
            img_data = doc.extract_image(xref)
            ext = img_data["ext"]
            path = os.path.join(image_dir, f"image_{i + 1}.{ext}")
            with open(path, "wb") as f:
                f.write(img_data["image"])
            image_paths.append(path)
        return image_paths
