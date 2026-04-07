"""
FastAPI server wrapping EmailEnv for OpenEnv multi-mode deployment.
Exposes the standard /reset, /step, /state HTTP endpoints.
"""
import logging
import os
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env.environment import EmailEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Email Triage Agent — OpenEnv API",
    description="Real-world email triage environment for AI agents. Supports reset/step/state.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TASK_NAME = os.environ.get("TASK_NAME", "hard")
env = EmailEnv(task_name=TASK_NAME)
logger.info("Environment loaded — task=%s", TASK_NAME)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: Optional[str] = None  # allows overriding task at runtime


class ActionPayload(BaseModel):
    action_type: str = Field(..., description="classify | reply | archive | escalate")
    priority_label: Optional[str] = Field(None, description="spam | urgent | normal (for classify)")
    content: Optional[str]        = Field(None, description="Reply body text (for reply actions)")


class StepRequest(BaseModel):
    action: ActionPayload


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", summary="Reset the environment to a new episode")
async def reset_env(req: ResetRequest = ResetRequest()):
    global env
    task = req.task_name or TASK_NAME
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Invalid task_name '{task}'. Must be easy|medium|hard.")
    if task != env.task_name:
        env = EmailEnv(task_name=task)
    obs = env.reset()
    return {"observation": obs, "reward": 0.0, "done": False, "info": {}}


@app.post("/step", summary="Submit an agent action and receive next observation + reward")
async def step_env(req: StepRequest):
    action_dict = req.action.model_dump()
    obs, reward, done, info = env.step(action_dict)
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state", summary="Return the current internal environment state")
async def get_state():
    return {"state": env.state(), "reward": 0.0, "done": env._state.done, "info": {}}


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "task": env.task_name}


# ---------------------------------------------------------------------------
# Entry-point (referenced in pyproject.toml [project.scripts])
# ---------------------------------------------------------------------------

def main() -> None:
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
