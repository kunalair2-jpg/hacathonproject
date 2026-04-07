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
# System prompt - OPTIMIZED FOR MAXIMUM REWARD EFFICIENCY
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert, hyper-efficient email triage AI agent. You will receive an email 
    and must extract full metadata and execute an action in a SINGLE pass for maximum efficiency.

    You MUST reply with exactly ONE valid JSON object — no prose, no reasoning, no markdown.

    Your JSON object must ALWAYS contain these fields:
      - "action_type": "archive" | "escalate" | "reply"
      - "priority_level": 1 to 5 (1=low, 5=critical)
      - "category_tag": "billing" | "technical" | "spam" | "inquiry" | "complaint" | "feedback" | "urgent"
      - "routing": e.g., "spam-filter", "finance-team", "engineering-team", "incident-response", "customer-success"
      - "content": (Only if action_type="reply") A highly professional, concise email response (≥40 characters)

    Decision rules for maximum efficiency:
      - Suspicious sender/promos → "archive", priority 1, category "spam", routing "spam-filter"
      - Production down/escalations → "escalate", priority 5, category "technical" or "urgent", routing "incident-response"
      - Questions about platform/help → "reply", priority 3, category "inquiry", routing "customer-success"
      - Billing issues/invoice requests → "reply", priority 3, category "billing", routing "finance-team"

    Output Example:
    {
      "action_type": "reply",
      "priority_level": 3,
      "category_tag": "billing",
      "routing": "finance-team",
      "content": "Hello, thank you for reaching out. We have received your invoice inquiry and our finance team will process it shortly. Please let us know if you have any other questions."
    }
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
