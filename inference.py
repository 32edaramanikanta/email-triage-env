"""
Inference Script — Email Triage OpenEnv
=======================================
Hackathon submission inference script.

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier.
    HF_TOKEN       Your Hugging Face / API key.

Usage:
    python inference.py

Runs all 3 tasks and prints reproducible baseline scores.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Make local modules importable from this script's directory
sys.path.insert(0, os.path.dirname(__file__))

from models import Action, ActionType, Label, Priority
from environment import EmailTriageEnv

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
MAX_TOKENS = 300
TEMPERATURE = 0.0   # Deterministic for reproducibility
RETRY_LIMIT = 2

TASKS = [
    "task1_basic_triage",
    "task2_draft_replies",
    "task3_sla_constrained",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert customer support triage agent. Your job is to process incoming
customer emails and decide how to handle each one.

For every email you receive, you must respond with a single JSON object (no markdown,
no explanation, just raw JSON) containing your triage decision.

Schema:
{
  "action_type": "triage" | "draft_reply" | "escalate" | "archive" | "mark_spam",
  "priority": "urgent" | "high" | "medium" | "low" | "spam",
  "label": "billing" | "technical" | "complaint" | "inquiry" | "refund" | "account" | "feedback" | "other",
  "reply_text": "<your reply if action_type is draft_reply, else null>",
  "escalation_reason": "<reason if action_type is escalate, else null>",
  "notes": "<optional internal notes>"
}

Guidelines:
- URGENT: Immediate risk to business, data breach, system down, legal threat
- HIGH: Enterprise customer issues, payment failures, account suspension
- MEDIUM: Non-critical technical issues, billing questions from paying customers
- LOW: General inquiries, feature requests, vendor outreach
- SPAM: Unsolicited marketing, scams, phishing

Action rules:
- triage: Use when you only need to classify (priority + label), no reply needed
- draft_reply: Write a professional, empathetic reply addressing the customer's issue
- escalate: Use for legal threats, security incidents, fraud, or issues needing human judgment
- archive: Low-value emails that don't need a response
- mark_spam: Clearly unsolicited / malicious email

Always output valid JSON. No extra text.
""").strip()


def build_user_prompt(obs) -> str:
    email = obs.email
    context_str = ""
    if obs.context:
        context_str = f"\n\nContext/SLA Rules:\n{json.dumps(obs.context, indent=2)}"

    history_str = ""
    if obs.history:
        last = obs.history[-3:]
        history_str = "\n\nRecent history:\n" + "\n".join(
            f"  Step {h['step']}: {h['email_id']} → {h['action_type']} "
            f"(reward: {h['reward']:.2f})"
            for h in last
        )

    return textwrap.dedent(f"""
Task: {obs.task_description}
Emails remaining in inbox: {obs.inbox_size}
Step: {obs.step + 1} / {obs.max_steps}
{history_str}

=== INCOMING EMAIL ===
Email ID: {email.id}
From: {email.sender} <{email.sender_domain}>
Subject: {email.subject}
Received: {email.received_at}
Thread length: {email.thread_length}
Has attachment: {email.has_attachment}
Metadata: {json.dumps(email.metadata, indent=2)}

Body:
{email.body}
{context_str}

Respond with JSON only.
""").strip()


# ---------------------------------------------------------------------------
# Parse model JSON output into Action
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> Optional[Action]:
    """Parse LLM JSON output into a typed Action."""
    text = response_text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                return None
        else:
            return None

    # Coerce string → enum
    action_type_str = data.get("action_type", "archive")
    try:
        action_type = ActionType(action_type_str)
    except ValueError:
        action_type = ActionType.ARCHIVE

    priority = None
    if data.get("priority"):
        try:
            priority = Priority(data["priority"])
        except ValueError:
            priority = Priority.LOW

    label = None
    if data.get("label"):
        try:
            label = Label(data["label"])
        except ValueError:
            label = Label.OTHER

    # For triage action, we need both priority and label
    if action_type == ActionType.TRIAGE and not priority:
        priority = Priority.LOW
    if action_type == ActionType.TRIAGE and not label:
        label = Label.OTHER

    # For non-triage, set defaults if missing
    if action_type != ActionType.TRIAGE:
        if not priority:
            priority = Priority.LOW
        if not label:
            label = Label.OTHER

    try:
        return Action(
            action_type=action_type,
            priority=priority,
            label=label,
            reply_text=data.get("reply_text") or None,
            escalation_reason=data.get("escalation_reason") or None,
            notes=data.get("notes") or None,
        )
    except Exception as e:
        # Try to salvage with minimal valid action
        try:
            if action_type == ActionType.DRAFT_REPLY and not data.get("reply_text"):
                action_type = ActionType.ARCHIVE
            if action_type == ActionType.ESCALATE and not data.get("escalation_reason"):
                action_type = ActionType.ESCALATE
                data["escalation_reason"] = "Escalating due to complexity."
            return Action(
                action_type=action_type,
                priority=priority or Priority.LOW,
                label=label or Label.OTHER,
                reply_text=data.get("reply_text") or None,
                escalation_reason=data.get("escalation_reason") or None,
            )
        except Exception:
            return Action(
                action_type=ActionType.ARCHIVE,
                priority=Priority.LOW,
                label=Label.OTHER,
            )


# ---------------------------------------------------------------------------
# Fallback action when model fails
# ---------------------------------------------------------------------------

def fallback_action() -> Action:
    return Action(
        action_type=ActionType.ARCHIVE,
        priority=Priority.LOW,
        label=Label.OTHER,
        notes="Fallback: model request failed.",
    )


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"TASK: {task_id}")
    print("=" * 60)

    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()

    step_results = []
    done = False

    while not done:
        if obs.email.id == "__done__":
            break

        user_prompt = build_user_prompt(obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response_text = ""
        for attempt in range(RETRY_LIMIT + 1):
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
                break
            except Exception as exc:
                print(f"  [attempt {attempt+1}] Model error: {exc}")
                if attempt == RETRY_LIMIT:
                    response_text = ""
                time.sleep(1)

        action = parse_action(response_text) if response_text else fallback_action()
        if action is None:
            action = fallback_action()

        obs, reward, done, info = env.step(action)

        step_results.append({
            "email_id": info["email_id"],
            "action_type": action.action_type.value,
            "priority": action.priority.value if action.priority else None,
            "label": action.label.value if action.label else None,
            "reward": reward.total,
            "priority_score": reward.priority_score,
            "label_score": reward.label_score,
            "reply_quality_score": reward.reply_quality_score,
            "escalation_score": reward.escalation_score,
            "penalty": reward.penalty,
        })

        print(
            f"  [{info['email_id']}] action={action.action_type.value} "
            f"priority={action.priority.value if action.priority else '-'} "
            f"label={action.label.value if action.label else '-'} "
            f"→ reward={reward.total:.3f}"
        )

    final_state = env.state()
    n = len(step_results)
    task_score = final_state.cumulative_reward / n if n > 0 else 0.0

    print(f"\n  Task score: {task_score:.4f}  (cumulative={final_state.cumulative_reward:.4f}, steps={n})")
    return {
        "task_id": task_id,
        "task_score": round(task_score, 4),
        "cumulative_reward": round(final_state.cumulative_reward, 4),
        "num_emails": n,
        "steps": step_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable.")
        sys.exit(1)
    if not MODEL_NAME:
        print("ERROR: Set MODEL_NAME environment variable.")
        sys.exit(1)

    print(f"Email Triage OpenEnv — Baseline Inference")
    print(f"Model:    {MODEL_NAME}")
    print(f"Endpoint: {API_BASE_URL}")
    print(f"Tasks:    {TASKS}")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_results = []
    start_time = time.time()

    for task_id in TASKS:
        result = run_task(client, task_id)
        all_results.append(result)

    elapsed = time.time() - start_time

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("BASELINE SCORES SUMMARY")
    print("=" * 60)
    for r in all_results:
        diff = {"task1_basic_triage": "EASY", "task2_draft_replies": "MEDIUM", "task3_sla_constrained": "HARD"}
        tag = diff.get(r["task_id"], "")
        print(f"  {tag:6s} | {r['task_id']:30s} | score={r['task_score']:.4f}")

    total_avg = sum(r["task_score"] for r in all_results) / len(all_results)
    print(f"\n  OVERALL AVERAGE SCORE: {total_avg:.4f}")
    print(f"  Elapsed time: {elapsed:.1f}s")

    # Save results to JSON
    output = {
        "model": MODEL_NAME,
        "overall_score": round(total_avg, 4),
        "tasks": all_results,
        "elapsed_seconds": round(elapsed, 1),
    }
    output_path = os.path.join(os.path.dirname(__file__), "baseline_scores.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
