#!/usr/bin/env python3
"""Validate task JSON files against docs/task_schema.json."""

import json
import sys
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ImportError:
    print("Missing dependency: jsonschema")
    print("Install with: pip install -r requirements.txt")
    sys.exit(2)

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = BASE_DIR / "docs" / "task_schema.json"
TASK_DIR = BASE_DIR / "tasks"


def main():
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    errors = []
    for task_file in sorted(TASK_DIR.glob("*.json")):
        data = json.loads(task_file.read_text())
        file_errors = sorted(validator.iter_errors(data), key=lambda e: str(e.path))
        for err in file_errors:
            path = ".".join(str(p) for p in err.path) or "<root>"
            errors.append(f"{task_file.name}: {path}: {err.message}")

    if errors:
        print("Task schema validation failed:")
        for line in errors:
            print(f" - {line}")
        return 1

    print(f"Validated {len(list(TASK_DIR.glob('*.json')))} task files against task_schema.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
