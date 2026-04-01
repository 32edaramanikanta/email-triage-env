"""
Deterministic graders for all three Email Triage tasks.
Each grader returns a score in [0.0, 1.0] with partial credit signals.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from models import Action, ActionType, Priority, Label, Reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PRIORITY_DISTANCE: Dict[str, int] = {
    "urgent": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "spam": 0,
}


def priority_score(predicted: Optional[str], expected: str) -> float:
    """Partial credit based on priority distance; exact match = 1.0."""
    if predicted is None:
        return 0.0
    if predicted == expected:
        return 1.0
    dist = abs(PRIORITY_DISTANCE.get(predicted, 0) - PRIORITY_DISTANCE.get(expected, 0))
    if dist == 1:
        return 0.5
    if dist == 2:
        return 0.2
    return 0.0


def label_score(predicted: Optional[str], expected: str) -> float:
    return 1.0 if predicted == expected else 0.0


def action_score(predicted: Optional[str], expected: str) -> float:
    return 1.0 if predicted == expected else 0.0


def reply_quality_score(reply_text: Optional[str], must_include: list[str], must_avoid: list[str]) -> float:
    """Score reply quality based on required/forbidden keywords."""
    if not reply_text:
        return 0.0
    reply_lower = reply_text.lower()
    score = 0.0
    if must_include:
        hits = sum(1 for kw in must_include if kw.lower() in reply_lower)
        score += 0.6 * (hits / len(must_include))
    else:
        score += 0.6

    if must_avoid:
        misses = sum(1 for kw in must_avoid if kw.lower() in reply_lower)
        score += 0.2 * (1.0 - misses / len(must_avoid))
    else:
        score += 0.2

    # Minimum length check
    word_count = len(reply_text.split())
    if word_count >= 30:
        score += 0.2
    elif word_count >= 15:
        score += 0.1

    return min(score, 1.0)


def escalation_quality_score(escalation_reason: Optional[str], keywords: list[str]) -> float:
    """Score escalation reason for relevant keywords."""
    if not escalation_reason:
        return 0.0
    reason_lower = escalation_reason.lower()
    hits = sum(1 for kw in keywords if kw.lower() in reason_lower)
    base = hits / len(keywords) if keywords else 1.0
    # Bonus for length
    word_count = len(escalation_reason.split())
    length_bonus = 0.15 if word_count >= 10 else 0.0
    return min(base + length_bonus, 1.0)


# ---------------------------------------------------------------------------
# Task 1 grader — Basic Triage
# ---------------------------------------------------------------------------

def grade_task1(email_id: str, action: Action, ground_truth: Dict[str, Any]) -> Reward:
    """
    Task 1: Assign priority + label + correct action type.
    Weights: priority 40%, label 40%, action 20%.
    """
    gt = ground_truth.get(email_id, {})
    weights = {"priority": 0.4, "label": 0.4, "action": 0.2}

    p_score = priority_score(
        action.priority.value if action.priority else None,
        gt.get("priority", ""),
    )
    l_score = label_score(
        action.label.value if action.label else None,
        gt.get("label", ""),
    )
    a_score = action_score(action.action_type.value, gt.get("action", ""))

    total = (
        weights["priority"] * p_score
        + weights["label"] * l_score
        + weights["action"] * a_score
    )

    return Reward(
        total=round(total, 4),
        priority_score=p_score,
        label_score=l_score,
        reply_quality_score=0.0,
        escalation_score=0.0,
        penalty=0.0,
        breakdown={
            "expected_priority": gt.get("priority"),
            "got_priority": action.priority.value if action.priority else None,
            "expected_label": gt.get("label"),
            "got_label": action.label.value if action.label else None,
            "expected_action": gt.get("action"),
            "got_action": action.action_type.value,
        },
    )


# ---------------------------------------------------------------------------
# Task 2 grader — Draft Replies
# ---------------------------------------------------------------------------

def grade_task2(email_id: str, action: Action, ground_truth: Dict[str, Any]) -> Reward:
    """
    Task 2: Triage + quality of reply/escalation.
    Weights: priority 20%, label 20%, action 30%, reply_quality 30%.
    """
    gt = ground_truth.get(email_id, {})
    weights = {"priority": 0.2, "label": 0.2, "action": 0.3, "reply_quality": 0.3}

    p_score = priority_score(
        action.priority.value if action.priority else None,
        gt.get("priority", ""),
    )
    l_score = label_score(
        action.label.value if action.label else None,
        gt.get("label", ""),
    )
    a_score = action_score(action.action_type.value, gt.get("action", ""))

    # Reply quality
    rq_score = 0.0
    esc_score = 0.0

    if gt.get("action") == "draft_reply":
        must_include = gt.get("reply_must_include", [])
        must_avoid = gt.get("reply_must_avoid", [])
        rq_score = reply_quality_score(action.reply_text, must_include, must_avoid)
    elif gt.get("action") == "escalate":
        esc_keywords = gt.get("escalation_keywords", [])
        esc_score = escalation_quality_score(action.escalation_reason, esc_keywords)
        rq_score = esc_score  # Use escalation quality as the "reply quality" slot

    total = (
        weights["priority"] * p_score
        + weights["label"] * l_score
        + weights["action"] * a_score
        + weights["reply_quality"] * rq_score
    )

    return Reward(
        total=round(total, 4),
        priority_score=p_score,
        label_score=l_score,
        reply_quality_score=rq_score,
        escalation_score=esc_score,
        penalty=0.0,
        breakdown={
            "expected_action": gt.get("action"),
            "got_action": action.action_type.value,
            "reply_text_provided": bool(action.reply_text),
            "escalation_reason_provided": bool(action.escalation_reason),
        },
    )


# ---------------------------------------------------------------------------
# Task 3 grader — SLA-Constrained Hard Triage
# ---------------------------------------------------------------------------

def grade_task3(
    email_id: str,
    action: Action,
    ground_truth: Dict[str, Any],
    context: Dict[str, Any],
    email_metadata: Dict[str, Any],
) -> Reward:
    """
    Task 3: SLA-constrained hard triage.
    Weights: priority 25%, label 15%, action 35%, escalation_quality 25%.
    Extra penalty for missing legal/security SLA violations.
    """
    gt = ground_truth.get(email_id, {})
    weights = {"priority": 0.25, "label": 0.15, "action": 0.35, "escalation_quality": 0.25}

    p_score = priority_score(
        action.priority.value if action.priority else None,
        gt.get("priority", ""),
    )
    l_score = label_score(
        action.label.value if action.label else None,
        gt.get("label", ""),
    )
    a_score = action_score(action.action_type.value, gt.get("action", ""))

    esc_keywords = gt.get("escalation_keywords", [])
    esc_score = 0.0
    if action.action_type == ActionType.ESCALATE and esc_keywords:
        esc_score = escalation_quality_score(action.escalation_reason, esc_keywords)
    elif gt.get("action") == "escalate" and action.action_type != ActionType.ESCALATE:
        # Failed to escalate a required escalation — penalize
        esc_score = 0.0

    penalty = 0.0
    if gt.get("sla_violation_if_not_escalated") and action.action_type != ActionType.ESCALATE:
        penalty = 0.15  # SLA penalty

    total = (
        weights["priority"] * p_score
        + weights["label"] * l_score
        + weights["action"] * a_score
        + weights["escalation_quality"] * esc_score
        - penalty
    )
    total = max(0.0, min(1.0, total))

    return Reward(
        total=round(total, 4),
        priority_score=p_score,
        label_score=l_score,
        reply_quality_score=0.0,
        escalation_score=esc_score,
        penalty=penalty,
        breakdown={
            "expected_action": gt.get("action"),
            "got_action": action.action_type.value,
            "sla_violated": gt.get("sla_violation_if_not_escalated", False)
            and action.action_type != ActionType.ESCALATE,
            "escalation_keywords_hit": [
                kw for kw in esc_keywords
                if action.escalation_reason and kw.lower() in action.escalation_reason.lower()
            ],
        },
    )


# ---------------------------------------------------------------------------
# Unified grader dispatcher
# ---------------------------------------------------------------------------

def grade(
    task_id: str,
    email_id: str,
    action: Action,
    ground_truth: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    email_metadata: Optional[Dict[str, Any]] = None,
) -> Reward:
    if task_id == "task1_basic_triage":
        return grade_task1(email_id, action, ground_truth)
    elif task_id == "task2_draft_replies":
        return grade_task2(email_id, action, ground_truth)
    elif task_id == "task3_sla_constrained":
        return grade_task3(
            email_id, action, ground_truth,
            context or {},
            email_metadata or {},
        )
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
