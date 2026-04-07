"""
Graders evaluate a recorded trajectory (list of step dicts) and return a
normalised score in [0.0, 1.0].

Each step dict must have:
  { "reward": float, "done": bool, "action": dict, "email": dict }
"""
from __future__ import annotations
from typing import List


def grade_easy(trajectory: List[dict]) -> float:
    """
    Easy grader: only checks if the agent chose the correct action.
    Score = fraction of steps with reward > 0.
    """
    if not trajectory:
        return 0.0
    positive = sum(1 for s in trajectory if s.get("reward", 0.0) > 0.0)
    return round(positive / len(trajectory), 4)


def grade_medium(trajectory: List[dict]) -> float:
    """
    Medium grader: checks action correctness + classification accuracy.
    Partial credit for classify steps.
    """
    if not trajectory:
        return 0.0
    total_score = sum(
        max(0.0, s.get("reward", 0.0)) for s in trajectory
    )
    # Normalise: best possible reward per step ≈ 0.8 (action+quality)
    best_possible = len(trajectory) * 0.8
    return round(min(total_score / best_possible, 1.0), 4)


def grade_hard(trajectory: List[dict]) -> float:
    """
    Hard grader: full rubric.
    - 30% weight: classification accuracy
    - 40% weight: correct action
    - 30% weight: reply quality
    Averaged and normalised to [0, 1].
    """
    if not trajectory:
        return 0.0

    classify_ok   = 0
    action_ok     = 0
    quality_ok    = 0
    classify_total = 0
    action_total   = 0
    quality_total  = 0

    for step in trajectory:
        action    = step.get("action", {})
        email     = step.get("email", {})
        atype     = action.get("action_type", "")
        reward    = step.get("reward", 0.0)

        if atype == "classify":
            classify_total += 1
            if action.get("priority_label") == email.get("priority"):
                classify_ok += 1
        elif atype in ("reply", "archive", "escalate"):
            action_total += 1
            if atype == email.get("expected_action"):
                action_ok += 1
            if atype == "reply":
                quality_total += 1
                content = action.get("content", "") or ""
                if len(content.strip()) >= 40:
                    quality_ok += 1

    c_score = (classify_ok  / classify_total)  if classify_total  else 0.5  # neutral if no classify steps
    a_score = (action_ok    / action_total)     if action_total    else 0.0
    q_score = (quality_ok   / quality_total)    if quality_total   else 0.5  # neutral if no replies

    final = 0.30 * c_score + 0.40 * a_score + 0.30 * q_score
    return round(min(final, 1.0), 4)
