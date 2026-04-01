"""
Email Triage OpenEnv Environment
Full OpenEnv spec: step() / reset() / state()
"""

from __future__ import annotations

import copy
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

# Make sibling packages importable
sys.path.insert(0, os.path.dirname(__file__))

from models import Action, ActionType, EpisodeState, Observation, Reward
from graders.graders import grade

# Lazy import dataset to avoid circular issues
def _load_registry():
    from data.emails import TASK_REGISTRY
    return TASK_REGISTRY


class EmailTriageEnv:
    """
    OpenEnv-compliant Email Triage environment.

    Simulates a customer-support inbox where an AI agent must:
      - Assign priority and labels to incoming emails
      - Draft appropriate replies or escalate where needed
      - Respect SLA constraints (Task 3)

    Three tasks of increasing difficulty:
      task1_basic_triage   — Easy
      task2_draft_replies  — Medium
      task3_sla_constrained — Hard
    """

    VALID_TASKS = [
        "task1_basic_triage",
        "task2_draft_replies",
        "task3_sla_constrained",
    ]

    def __init__(self, task_id: str = "task1_basic_triage") -> None:
        if task_id not in self.VALID_TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Choose from: {self.VALID_TASKS}"
            )
        self.task_id = task_id
        self._registry = _load_registry()
        self._task_cfg = self._registry[task_id]

        # Episode state (initialised by reset())
        self._emails: List[Any] = []
        self._email_index: int = 0
        self._step: int = 0
        self._done: bool = True
        self._cumulative_reward: float = 0.0
        self._action_history: List[Dict[str, Any]] = []
        self._episode_rewards: List[float] = []

    # -----------------------------------------------------------------------
    # OpenEnv interface
    # -----------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the first observation."""
        cfg = self._task_cfg
        self._emails = list(cfg["emails"])
        self._email_index = 0
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._action_history = []
        self._episode_rewards = []
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process one action.

        Returns:
            observation: next observation (or final if done)
            reward:      Reward model for this step
            done:        whether episode is complete
            info:        diagnostic dict
        """
        if self._done:
            raise RuntimeError("Environment is done. Call reset() first.")

        current_email = self._emails[self._email_index]
        cfg = self._task_cfg

        reward = grade(
            task_id=self.task_id,
            email_id=current_email.id,
            action=action,
            ground_truth=cfg["ground_truth"],
            context=cfg.get("context", {}),
            email_metadata=current_email.metadata,
        )

        self._cumulative_reward += reward.total
        self._episode_rewards.append(reward.total)
        self._step += 1

        history_entry = {
            "step": self._step,
            "email_id": current_email.id,
            "email_subject": current_email.subject,
            "action_type": action.action_type.value,
            "priority": action.priority.value if action.priority else None,
            "label": action.label.value if action.label else None,
            "reward": reward.total,
        }
        self._action_history.append(history_entry)

        self._email_index += 1
        max_steps = cfg["max_steps"]

        if self._email_index >= len(self._emails) or self._step >= max_steps:
            self._done = True

        if self._done:
            obs = self._make_final_observation()
        else:
            obs = self._make_observation()

        info = {
            "email_id": current_email.id,
            "step": self._step,
            "reward_breakdown": reward.breakdown,
            "cumulative_reward": self._cumulative_reward,
            "done": self._done,
        }

        return obs, reward, self._done, info

    def state(self) -> EpisodeState:
        """Return the current episode state."""
        current_email_id = None
        if not self._done and self._email_index < len(self._emails):
            current_email_id = self._emails[self._email_index].id

        return EpisodeState(
            task_id=self.task_id,
            step=self._step,
            max_steps=self._task_cfg["max_steps"],
            done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            emails_processed=self._email_index,
            action_history=copy.deepcopy(self._action_history),
            current_email_id=current_email_id,
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _make_observation(self) -> Observation:
        cfg = self._task_cfg
        email = self._emails[self._email_index]
        remaining = len(self._emails) - self._email_index
        return Observation(
            email=email,
            inbox_size=remaining,
            step=self._step,
            max_steps=cfg["max_steps"],
            task_id=self.task_id,
            task_description=cfg["description"],
            history=copy.deepcopy(self._action_history),
            context=cfg.get("context", {}),
        )

    def _make_final_observation(self) -> Observation:
        """Return a terminal observation with a sentinel email."""
        from models import Email
        cfg = self._task_cfg
        sentinel = Email(
            id="__done__",
            subject="Episode complete",
            sender="system@openenv",
            sender_domain="openenv",
            body=(
                f"Episode complete. "
                f"Processed {self._email_index} emails. "
                f"Total reward: {self._cumulative_reward:.4f}"
            ),
            received_at="",
        )
        return Observation(
            email=sentinel,
            inbox_size=0,
            step=self._step,
            max_steps=cfg["max_steps"],
            task_id=self.task_id,
            task_description=cfg["description"],
            history=copy.deepcopy(self._action_history),
            context=cfg.get("context", {}),
        )

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    @property
    def action_space_description(self) -> str:
        return (
            "Action fields:\n"
            "  action_type: triage | draft_reply | escalate | archive | mark_spam\n"
            "  priority:    urgent | high | medium | low | spam  (required for triage)\n"
            "  label:       billing | technical | complaint | inquiry | refund |\n"
            "               account | feedback | other       (required for triage)\n"
            "  reply_text:  str (required for draft_reply)\n"
            "  escalation_reason: str (required for escalate)\n"
            "  notes:       str (optional internal note)"
        )

    @property
    def observation_space_description(self) -> str:
        return (
            "Observation fields:\n"
            "  email:            Email object (id, subject, sender, body, metadata)\n"
            "  inbox_size:       int — remaining emails in inbox\n"
            "  step:             int — current step\n"
            "  max_steps:        int — episode limit\n"
            "  task_id:          str\n"
            "  task_description: str — natural language goal\n"
            "  history:          list of previous actions + rewards\n"
            "  context:          dict with SLA rules, templates (Task 3 only)"
        )
