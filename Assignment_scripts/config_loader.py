# config_loader.py
import yaml
from typing import Dict, Any

def load_defaults(path: str = "defaults.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("defaults", {})