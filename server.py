"""
FastAPI server exposing the Email Triage OpenEnv environment
over HTTP so HF Spaces and the openenv validate tool can hit it.

Endpoints:
  GET  /                 → health check
  POST /reset            → reset(task_id) → Observation
  POST /step             → step(Action)   → obs, reward, done, info
  GET  /state            → state()        → EpisodeState
  GET  /tasks            → list tasks
  GET  /action_space     → action space description
  GET  /observation_space → observation space description
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import Action, ActionType, EpisodeState, Observation, Reward
from environment import EmailTriageEnv

app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "A real-world email triage environment for training and evaluating AI agents. "
        "Agents must prioritize, label, reply to, or escalate customer support emails."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-session; for multi-session use a session map)
_env: Optional[EmailTriageEnv] = None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task1_basic_triage"


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {
        "status": "ok",
        "environment": "Email Triage OpenEnv",
        "version": "1.0.0",
        "tasks": EmailTriageEnv.VALID_TASKS,
    }


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest):
    global _env
    try:
        _env = EmailTriageEnv(task_id=request.task_id)
        obs = _env.reset()
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    try:
        obs, reward, done, info = _env.step(request.action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


@app.get("/state", response_model=EpisodeState)
def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return _env.state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "task1_basic_triage",
                "difficulty": "easy",
                "description": "Assign priority and label to emails. Choose correct action type.",
                "num_emails": 5,
                "max_steps": 5,
            },
            {
                "id": "task2_draft_replies",
                "difficulty": "medium",
                "description": "Handle escalated issues: draft replies, escalate legal/fraud cases.",
                "num_emails": 5,
                "max_steps": 5,
            },
            {
                "id": "task3_sla_constrained",
                "difficulty": "hard",
                "description": "SLA-constrained triage: enterprise, regulated industry, legal cases.",
                "num_emails": 5,
                "max_steps": 5,
            },
        ]
    }


@app.get("/action_space")
def action_space():
    env = EmailTriageEnv()
    return {"description": env.action_space_description}


@app.get("/observation_space")
def observation_space():
    env = EmailTriageEnv()
    return {"description": env.observation_space_description}
