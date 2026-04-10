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

    DIR_DATA_LEOPARDS = os.path.join("data", "leopards")

    @property
    def json_path(self):
        return os.path.join(self.DIR_DATA_LEOPARDS, f"{self.id}.json")

    def write(self):
        with open(self.json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    @classmethod
    def from_file(cls, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def list_all(cls):
        leopard_dir = cls.DIR_DATA_LEOPARDS
        leopard_files = [
            f for f in os.listdir(leopard_dir) if f.endswith(".json")
        ]
        return [
            cls.from_file(os.path.join(leopard_dir, f)) for f in leopard_files
        ]
