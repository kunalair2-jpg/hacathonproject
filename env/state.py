from typing import List, Optional
from .models import Email


class EmailState:
    """Tracks episode state: current inbox cursor, thread memory, step count."""

    def __init__(self, emails: List[Email]) -> None:
        self._emails = emails
        self.current_index: int = 0
        self.done: bool = False
        self.steps: int = 0
        self.thread_memory: List[str] = []  # recent thread IDs (working memory)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def total_emails(self) -> int:
        return len(self._emails)

    def get_current_email(self) -> Optional[Email]:
        if self.done or self.current_index >= len(self._emails):
            return None
        return self._emails[self.current_index]

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def advance(self) -> None:
        email = self.get_current_email()
        if email and email.thread_id:
            if email.thread_id not in self.thread_memory:
                self.thread_memory.append(email.thread_id)
            # Keep last 5 thread IDs (sliding window memory)
            self.thread_memory = self.thread_memory[-5:]

        self.current_index += 1
        if self.current_index >= len(self._emails):
            self.done = True

    def reset(self) -> None:
        self.current_index = 0
        self.done = False
        self.steps = 0
        self.thread_memory = []
