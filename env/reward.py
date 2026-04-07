"""
Multi-signal reward function for the Email-Triage OpenEnv environment.

Four components (matching the winning blueprint):
  1. Priority accuracy   — 0.40 weight  (off-by-one tolerant)
  2. Category F1         — 0.30 weight  (spam misclassification penalised extra)
  3. Routing correctness — 0.20 weight  (did it go to the right team?)
  4. Speed / efficiency  — 0.10 weight  (fewer steps = better)

Total reward clipped to [-1.0, 1.0].
"""
from .models import Email

# ---- Weights (must sum to 1.0) ----
W_PRIORITY  = 0.40
W_CATEGORY  = 0.30
W_ROUTING   = 0.20
W_SPEED     = 0.10
MAX_STEPS   = 30   # after this, speed bonus is 0


def compute_reward(action: dict, email: Email, steps: int) -> tuple[float, str]:
    action_type   = action.get("action_type", "noop")
    content       = action.get("content", "") or ""
    reasons: list[str] = []
    reward = 0.0

    # ------------------------------------------------------------------
    # CLASSIFY sub-step (free — does not consume email)
    # Returns early with partial reward signal only.
    # ------------------------------------------------------------------
    if action_type == "classify":
        sub = 0.0
        # Priority label
        pred_priority = action.get("priority_label", "")
        if pred_priority == email.priority:
            sub += W_PRIORITY * 0.5
            reasons.append(f"+priority_label_correct({email.priority})")
        else:
            sub -= W_PRIORITY * 0.3
            reasons.append(f"-priority_label_wrong(exp={email.priority},got={pred_priority})")

        # Priority level (1-5), off-by-one tolerance
        pred_level = action.get("priority_level")
        if pred_level is not None:
            diff = abs(int(pred_level) - email.priority_level)
            if diff == 0:
                sub += W_PRIORITY * 0.5
                reasons.append(f"+priority_level_exact({email.priority_level})")
            elif diff == 1:
                sub += W_PRIORITY * 0.2   # partial credit
                reasons.append(f"+priority_level_close(exp={email.priority_level},got={pred_level})")
            else:
                sub -= W_PRIORITY * 0.3
                reasons.append(f"-priority_level_far(exp={email.priority_level},got={pred_level})")

        # Category tag (weighted — spam misclassification is costly)
        pred_cat = action.get("category_tag", "")
        if pred_cat == email.category:
            sub += W_CATEGORY
            reasons.append(f"+category_correct({email.category})")
        else:
            penalty = W_CATEGORY * 0.8 if email.category == "spam" else W_CATEGORY * 0.4
            sub -= penalty
            reasons.append(f"-category_wrong(exp={email.category},got={pred_cat},penalty={penalty:.2f})")

        return _clip(sub), ", ".join(reasons)

    # ------------------------------------------------------------------
    # FINAL ACTION: archive | reply | escalate
    # ------------------------------------------------------------------

    # 1. Priority accuracy (inferred from action choice)
    if action_type == email.expected_action:
        reward += W_PRIORITY
        reasons.append(f"+priority_action_correct({action_type})")
    else:
        reward -= W_PRIORITY * 0.6
        reasons.append(f"-priority_action_wrong(exp={email.expected_action},got={action_type})")

    # 2. Category / semantic correctness
    pred_cat = action.get("category_tag", "")
    if pred_cat == email.category:
        reward += W_CATEGORY
        reasons.append(f"+category_match({email.category})")
    elif pred_cat:  # wrong but tried
        penalty = W_CATEGORY * 0.8 if email.category == "spam" else W_CATEGORY * 0.4
        reward -= penalty
        reasons.append(f"-category_mismatch(exp={email.category},got={pred_cat})")
    # no penalty if category not provided (partial agent)

    # 3. Routing correctness
    pred_routing = action.get("routing", "")
    if pred_routing == email.routing:
        reward += W_ROUTING
        reasons.append(f"+routing_correct({email.routing})")
    elif pred_routing:
        reward -= W_ROUTING * 0.5
        reasons.append(f"-routing_wrong(exp={email.routing},got={pred_routing})")

    # 4. Reply quality (counted under speed/quality component)
    if action_type == "reply":
        length = len(content.strip())
        if length >= 60:
            reward += W_SPEED
            reasons.append("+reply_quality_high")
        elif length >= 25:
            reward += W_SPEED * 0.5
            reasons.append("+reply_quality_medium")
        else:
            reward -= W_SPEED * 0.5
            reasons.append("-reply_too_short")
    else:
        # Speed bonus for non-reply actions
        speed_ratio = max(0.0, 1.0 - steps / MAX_STEPS)
        speed_bonus = W_SPEED * speed_ratio
        reward += speed_bonus
        if speed_bonus > 0.01:
            reasons.append(f"+speed_bonus({speed_bonus:.2f})")

    return _clip(reward), ", ".join(reasons)


def _clip(r: float) -> float:
    return round(max(-1.0, min(1.0, r)), 4)
