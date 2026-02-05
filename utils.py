"""Small JSON/file utilities shared across training and analysis scripts."""

import json
import os
from typing import Any, Dict


def _json_default(value: Any) -> Any:
    """Serialize common array/scalar objects (e.g., NumPy) to JSON-friendly types."""

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_json(path: str) -> Dict[str, Any]:
    """Load and return a JSON dictionary from disk."""

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_dict_to_json(data: Dict[str, Any], output_folder: str, filename: str, indent: int = 2) -> str:
    """Write a dictionary to `<output_folder>/<filename>` and return the output path."""

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, default=_json_default)
    return output_path
