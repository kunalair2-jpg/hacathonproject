from pydantic import BaseModel, ConfigDict, Field
from typing import Optional


class Email(BaseModel):
    id: str
    subject: str
    body: str
    sender: str
    priority: str          # "spam" | "urgent" | "normal"
    priority_level: int    # 1 (low) – 5 (critical)
    category: str          # "billing" | "technical" | "spam" | "urgent" | "complaint" | "inquiry" | "feedback"
    routing: str           # "spam-filter" | "finance-team" | "engineering-team" | "incident-response" | etc.
    expected_action: str   # "archive" | "reply" | "escalate"
    difficulty: str = "easy"
    thread_id: Optional[str] = None

    model_config = ConfigDict(extra="ignore")

    @property
    def length_bucket(self) -> str:
        n = len(self.body)
        if n < 80:   return "short"
        if n < 200:  return "medium"
        return "long"


class Action(BaseModel):
    """
    All fields an agent can submit in one step.
    action_type is required; all others are optional enrichment.
    """
    action_type: str              = Field(..., description="classify | reply | archive | escalate")
    priority_label: Optional[str] = Field(None, description="spam | urgent | normal — used when classifying")
    priority_level: Optional[int] = Field(None, description="1-5 urgency score — used when classifying")
    category_tag: Optional[str]   = Field(None, description="billing | technical | spam | urgent | complaint | inquiry | feedback")
    routing: Optional[str]        = Field(None, description="target queue/team name")
    content: Optional[str]        = Field(None, description="Reply draft text — used when action_type=reply")

    model_config = ConfigDict(extra="ignore")
