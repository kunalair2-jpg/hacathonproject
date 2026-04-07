from pydantic import BaseModel, ConfigDict
from typing import List, Optional


class Email(BaseModel):
    id: str
    subject: str
    body: str
    sender: str
    priority: str          # "spam" | "urgent" | "normal"
    expected_action: str   # "archive" | "reply" | "escalate"
    difficulty: str = "easy"
    thread_id: Optional[str] = None

    model_config = ConfigDict(extra="ignore")


class Action(BaseModel):
    action_type: str              # "classify" | "reply" | "archive" | "escalate"
    priority_label: Optional[str] = None  # used when action_type == "classify"
    content: Optional[str] = None         # reply body or auxiliary text

    model_config = ConfigDict(extra="ignore")
