"""
inference.py — Email Triage Agent Inference Script
=====================================================
MANDATORY env vars:
  API_BASE_URL  — LLM API base URL  (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    — model identifier  (e.g. meta-llama/Llama-3.1-8B-Instruct)
  HF_TOKEN      — Hugging Face API key
"""
import os
import re
import json
import textwrap
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

MAX_STEPS    = 20
TEMPERATURE  = 0.2
MAX_TOKENS   = 300
TASK_NAME    = os.getenv("TASK_NAME", "hard")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert email triage AI agent. You will receive an email and must
    decide how to handle it.

    You MUST reply with a single valid JSON object — no prose, no markdown.

    Choose one of the following action_type values:
      - "classify"  → set priority_label to "spam", "urgent", or "normal"
      - "archive"   → for spam or low-priority mail (no content needed)
      - "escalate"  → for urgent issues needing human escalation (no content needed)
      - "reply"     → write a professional reply in the "content" field (≥40 chars)

    Decision rules:
      - Suspicious sender / prize claims / gift cards → "spam" → archive
      - Production outages / legal / executive deadlines → "urgent" → escalate
      - HR newsletters / low-priority notices → "normal" → archive
      - Meeting reminders / colleague requests → "normal" → reply

    Response format (choose one):
    {"action_type": "classify", "priority_label": "spam"}
    {"action_type": "archive"}
    {"action_type": "escalate"}
    {"action_type": "reply", "content": "Dear ..., <your professional reply here>"}
""").strip()


def build_user_prompt(obs: dict, step: int, history: list[str]) -> str:
    email = obs.get("email") or {}
    inbox_left = obs.get("inbox_remaining", "?")
    memory     = obs.get("thread_memory", [])

    return textwrap.dedent(f"""
        Step {step}  |  Inbox remaining: {inbox_left}
        Thread memory: {memory}

        --- EMAIL ---
        ID:       {email.get("id", "?")}
        From:     {email.get("sender", "?")}
        Subject:  {email.get("subject", "?")}
        Body:
        {email.get("body", "")}
        -------------

        Recent history:
        {chr(10).join(history[-4:]) or "None"}

        Reply with a single JSON action.
    """).strip()


def parse_action(text: str) -> dict:
    """Extract a JSON action dict from the model response."""
    # Try to find a JSON block
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback
    return {"action_type": "archive"}


def run_episode(client: OpenAI, env_url: str | None = None) -> float:
    """
    Run one full episode.
    If env_url is set, talk to the remote HTTP server.
    Otherwise instantiate the environment locally.
    """
    import requests  # kept scoped so inference.py works in Docker without requests

    history: list[str] = []
    total_reward = 0.0

    # --- Reset ---
    resp = requests.post(f"{env_url}/reset", json={"task_name": TASK_NAME}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()["observation"]
    done = resp.json().get("done", False)
    print(f"\n🚀 Episode started — task={TASK_NAME}")

    for step in range(1, MAX_STEPS + 1):
        if done or obs.get("email") is None:
            print("✅ Episode done early.")
            break

        user_prompt = build_user_prompt(obs, step, history)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            raw = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"⚠️  LLM request failed at step {step}: {exc}")
            raw = '{"action_type": "archive"}'

        action = parse_action(raw)
        print(f"  Step {step:02d}  action={action}")

        # --- Step ---
        step_resp = requests.post(f"{env_url}/step", json={"action": action}, timeout=30)
        step_resp.raise_for_status()
        payload     = step_resp.json()
        obs         = payload["observation"]
        reward      = payload["reward"]
        done        = payload["done"]
        info        = payload.get("info", {})
        total_reward += reward

        history.append(f"step={step} action={action.get('action_type')} reward={reward:+.3f} {info.get('reason','')}")
        print(f"         reward={reward:+.3f}  done={done}  reason={info.get('reason','')}")

    print(f"\n📊 Episode complete — total_reward={total_reward:.3f}")
    return total_reward


def main() -> None:
    if not API_KEY:
        print("⚠️  HF_TOKEN / API_KEY not set. Skipping live inference.")
        print("   Set API_BASE_URL, MODEL_NAME, HF_TOKEN then re-run.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env_url = os.getenv("ENV_URL", "http://localhost:8000")
    run_episode(client, env_url=env_url)


if __name__ == "__main__":
    main()
