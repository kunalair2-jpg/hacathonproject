"""
Core OpenEnv Environment: EmailEnv
Implements the standard step() / reset() / state() API.
"""
import logging
from typing import Any

from .state import EmailState
from .reward import compute_reward
from tasks.data import get_easy_task, get_medium_task, get_hard_task

logger = logging.getLogger(__name__)

VALID_TASKS = ("easy", "medium", "hard")


class EmailEnv:
    """
    A real-world email triage and response environment.

    The agent receives a structured observation for each email in the inbox
    and must decide:
      1. (Optional) classify: label spam / urgent / normal
      2.             act:      reply | archive | escalate

    Rewards are dense and shaped to encourage:
      - Correct triage priority classification (+0.30)
      - Correct workflow action (+0.50)
      - High-quality reply generation (+0.20)
      - Efficiency: compounding -0.01 penalty per step
    """

    def __init__(self, task_name: str = "easy") -> None:
        if task_name not in VALID_TASKS:
            logger.warning("Unknown task '%s', defaulting to 'easy'.", task_name)
            task_name = "easy"

        self.task_name = task_name
        self._state: EmailState = self._build_state(task_name)

    # ------------------------------------------------------------------
    # OpenEnv required public API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset the environment to the start of a new episode."""
        self._state.reset()
        logger.info("Episode reset — task=%s, emails=%d", self.task_name, self._state.total_emails)
        return self._observe()

    def state(self) -> dict:
        """Return the full internal environment state (meta-information)."""
        email = self._state.get_current_email()
        return {
            "task": self.task_name,
            "total_emails": self._state.total_emails,
            "emails_remaining": self._state.total_emails - self._state.current_index,
            "steps_taken": self._state.steps,
            "done": self._state.done,
            "current_email_id": email.id if email else None,
            "thread_memory": list(self._state.thread_memory),
        }

    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        """
        Process one agent action.

        Args:
            action: dict with keys:
                - action_type (str):    "classify" | "reply" | "archive" | "escalate"
                - priority_label (str): used only when action_type == "classify"
                - content (str):        reply body (used only when action_type == "reply")

        Returns:
            observation (dict), reward (float), done (bool), info (dict)
        """
        if self._state.done:
            return self._observe(), 0.0, True, {"reason": "episode_already_done"}

        # Coerce to dict
        if not isinstance(action, dict):
            return self._observe(), -1.0, self._state.done, {"reason": "invalid_action_type"}

        email = self._state.get_current_email()
        if email is None:
            return self._observe(), 0.0, True, {"reason": "no_email_available"}

        reward, reason = compute_reward(action, email, self._state.steps)
        self._state.steps += 1

        # Consume the email only on a final action (not on classify which is a free sub-step)
        action_type = action.get("action_type", "noop")
        if action_type in ("reply", "archive", "escalate"):
            self._state.advance()

        done = self._state.done
        obs  = self._observe()

        logger.debug("step=%d action=%s reward=%.3f done=%s", self._state.steps, action_type, reward, done)
        return obs, reward, done, {"reason": reason}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _observe(self) -> dict:
        email = self._state.get_current_email()
        return {
            "email": email.model_dump() if email else None,
            "inbox_remaining": self._state.total_emails - self._state.current_index,
            "step_count": self._state.steps,
            "thread_memory": list(self._state.thread_memory),
        }

    @staticmethod
    def _build_state(task_name: str) -> EmailState:
        loaders = {
            "easy":   get_easy_task,
            "medium": get_medium_task,
            "hard":   get_hard_task,
        }
        return EmailState(loaders[task_name]())
