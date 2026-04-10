from dataclasses import dataclass


@dataclass
class Leopard:
    id: str
    name: str
    gender: str  # M or F
    location_details: str
    correlation_details: str
    image_path_list: list[str]
    zone: str
    date_first_seen: str
    date_last_seen: str
    mother_id: str
