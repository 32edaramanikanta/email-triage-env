"""
Typed Pydantic models for the Email Triage OpenEnv environment.
Observation, Action, Reward — all fully typed per OpenEnv spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPAM = "spam"


class ActionType(str, Enum):
    TRIAGE = "triage"          # Assign priority + label
    DRAFT_REPLY = "draft_reply"  # Write a reply
    ESCALATE = "escalate"      # Escalate to human agent
    ARCHIVE = "archive"        # Archive without reply
    MARK_SPAM = "mark_spam"    # Mark as spam


class Label(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    COMPLAINT = "complaint"
    INQUIRY = "inquiry"
    REFUND = "refund"
    ACCOUNT = "account"
    FEEDBACK = "feedback"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Email data model
# ---------------------------------------------------------------------------

class Email(BaseModel):
    id: str
    subject: str
    sender: str
    sender_domain: str
    body: str
    received_at: str          # ISO timestamp string
    has_attachment: bool = False
    thread_length: int = 1    # Number of messages in thread
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each step."""

    email: Email
    inbox_size: int = Field(..., description="Total emails remaining in inbox")
    step: int = Field(..., description="Current step within episode")
    max_steps: int = Field(..., description="Maximum steps allowed")
    task_id: str = Field(..., description="Active task identifier")
    task_description: str = Field(..., description="Natural language task goal")
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Previous actions taken this episode",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra task-specific context (SLA rules, templates, etc.)",
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """What the agent can do."""

    action_type: ActionType
    priority: Optional[Priority] = None
    label: Optional[Label] = None
    reply_text: Optional[str] = Field(
        None,
        description="Draft reply body (required for draft_reply action)",
        max_length=2000,
    )
    escalation_reason: Optional[str] = Field(
        None,
        description="Reason for escalation (required for escalate action)",
        max_length=500,
    )
    notes: Optional[str] = Field(
        None,
        description="Internal notes about the decision",
        max_length=500,
    )

    def model_post_init(self, __context: Any) -> None:
        if self.action_type == ActionType.DRAFT_REPLY and not self.reply_text:
            raise ValueError("reply_text is required for draft_reply action")
        if self.action_type == ActionType.ESCALATE and not self.escalation_reason:
            raise ValueError("escalation_reason is required for escalate action")
        if self.action_type == ActionType.TRIAGE:
            if not self.priority:
                raise ValueError("priority is required for triage action")
            if not self.label:
                raise ValueError("label is required for triage action")


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Structured reward with partial credit signals."""

    total: float = Field(..., ge=0.0, le=1.0, description="Overall reward 0–1")
    priority_score: float = Field(0.0, ge=0.0, le=1.0)
    label_score: float = Field(0.0, ge=0.0, le=1.0)
    reply_quality_score: float = Field(0.0, ge=0.0, le=1.0)
    escalation_score: float = Field(0.0, ge=0.0, le=1.0)
    penalty: float = Field(0.0, ge=0.0, le=1.0, description="Penalty deducted")
    breakdown: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Episode state (returned by state())
# ---------------------------------------------------------------------------

class EpisodeState(BaseModel):
    task_id: str
    step: int
    max_steps: int
    done: bool
    cumulative_reward: float
    emails_processed: int
    action_history: List[Dict[str, Any]]
    current_email_id: Optional[str]
