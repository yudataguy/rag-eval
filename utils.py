import json
from typing import Dict, Any


def load_json(file) -> Dict[str, Any]:
    return json.load(file)


def validate_json(data: Dict[str, Any]) -> bool:
    required_keys = ["query", "generate", "retrieval"]
    return all(key in data for key in required_keys)
