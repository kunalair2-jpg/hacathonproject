import json
import os
from env.models import Email

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "emails.json")


def _load_all() -> list[Email]:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Email(**e) for e in raw]


def get_easy_task() -> list[Email]:
    return [e for e in _load_all() if e.difficulty == "easy"]


def get_medium_task() -> list[Email]:
    return [e for e in _load_all() if e.difficulty in ("easy", "medium")]


def get_hard_task() -> list[Email]:
    return _load_all()
