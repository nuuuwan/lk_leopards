import json
import os
from dataclasses import dataclass


@dataclass
class Leopard:
    id: str
    name: str
    gender: str  # M or F
    location_details: str
    correlation_details: str
    image_path_list: list[str]
    zone_list: list[str]
    date_first_seen: str
    date_last_seen: str
    mother_id: str

    @property
    def json_path(self):
        return os.path.join("data", "leopards", f"{self.id}.json")

    def write(self):
        with open(self.json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
