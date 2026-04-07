"""
Reward shaper for the Email-Triage environment.
Returns (reward: float, reason: str) — both are logged per step.
Reward is clipped to [-1.0, 1.0].
"""
from .models import Email

# Weight constants (must sum to roughly ≤ 1.0 for max achievable)
W_CLASSIFY = 0.30
W_ACTION   = 0.50
W_QUALITY  = 0.20
W_EFFICIENCY_PER_STEP = 0.01   # subtracted each step (compounding)


def compute_reward(action: dict, email: Email, steps: int) -> tuple[float, str]:
    action_type   = action.get("action_type", "noop")
    priority_label = action.get("priority_label", "")
    content        = action.get("content", "") or ""

    reward = 0.0
    reasons: list[str] = []

    # ------------------------------------------------------------------
    # 1. Classification sub-reward (agent may call classify before acting)
    # ------------------------------------------------------------------
    if action_type == "classify":
        if priority_label == email.priority:
            reward += W_CLASSIFY
            reasons.append(f"+classify_correct({email.priority})")
        else:
            reward -= W_CLASSIFY * 0.5         # partial penalty
            reasons.append(f"-classify_wrong(expected={email.priority},got={priority_label})")
        # Classify alone does not consume the email — return immediately
        reward = _clip(reward)
        return reward, ", ".join(reasons)

    # ------------------------------------------------------------------
    # 2. Core action reward
    # ------------------------------------------------------------------
    if action_type == email.expected_action:
        reward += W_ACTION
        reasons.append(f"+action_correct({action_type})")
    elif action_type in ("archive", "reply", "escalate"):
        reward -= W_ACTION * 0.6               # wrong action penalty
        reasons.append(f"-action_wrong(expected={email.expected_action},got={action_type})")
    else:
        # noop or unknown
        reward -= W_ACTION * 0.3
        reasons.append(f"-invalid_action({action_type})")

    # ------------------------------------------------------------------
    # 3. Reply quality bonus
    # ------------------------------------------------------------------
    if action_type == "reply":
        if len(content.strip()) >= 40:
            reward += W_QUALITY
            reasons.append("+reply_quality_high")
        elif len(content.strip()) >= 15:
            reward += W_QUALITY * 0.5
            reasons.append("+reply_quality_medium")
        else:
            reward -= W_QUALITY * 0.5
            reasons.append("-reply_too_short")

    # ------------------------------------------------------------------
    # 4. Efficiency penalty: slow agents lose points
    # ------------------------------------------------------------------
    step_penalty = W_EFFICIENCY_PER_STEP * min(steps, 30)
    reward -= step_penalty
    if step_penalty > 0:
        reasons.append(f"-efficiency({step_penalty:.2f})")

    return _clip(reward), ", ".join(reasons)


def _clip(r: float) -> float:
    return max(-1.0, min(1.0, r))
