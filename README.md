# 📧 Email Triage OpenEnv

> **A real-world customer support email triage environment for training and evaluating AI agents.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-green)](https://openenv.dev)
[![HuggingFace Spaces](https://img.shields.io/badge/HF%20Space-deployed-blue)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](./Dockerfile)

---

## 🎯 What Is This?

Every company has a customer support inbox. Triaging it well — routing urgent issues instantly, drafting empathetic replies, escalating legal threats — is a real skill that takes judgment, context, and consistency.

This environment simulates that exact workflow. An AI agent processes a stream of incoming customer emails and must:

- **Assign priority** (urgent → spam) and **category labels**
- **Draft professional replies** for complaints and technical questions
- **Escalate** to the right team for legal, fraud, or security cases
- **Respect SLA constraints** for enterprise customers and regulated industries

This is not a toy. The emails are realistic, the graders are strict, and Task 3 is genuinely hard — even frontier models struggle to identify all regulatory escalation triggers.

---

## 🏗️ Architecture

```
email-triage-env/
├── inference.py          # ← Required hackathon submission script
├── server.py             # FastAPI REST server (OpenEnv HTTP spec)
├── environment.py        # Core EmailTriageEnv class (step/reset/state)
├── models.py             # Typed Pydantic models (Observation/Action/Reward)
├── openenv.yaml          # OpenEnv spec metadata
├── Dockerfile            # HF Spaces / Docker deployment
├── requirements.txt
├── data/
│   └── emails.py         # Synthetic email dataset + ground truth
├── graders/
│   └── graders.py        # Deterministic graders (0.0–1.0)
└── tests/
    └── test_env.py       # Full test suite
```

---

## 🚀 Quick Start

### Local (Python)

```bash
# Install deps
pip install -r requirements.txt

# Start the API server
uvicorn server:app --host 0.0.0.0 --port 7860

# In another terminal, run the baseline inference
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your_hf_token_here"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
python inference.py
```

### Docker

```bash
docker build -t email-triage-openenv .
docker run -p 7860:7860 email-triage-openenv

# Run inference against the container
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your_hf_token_here"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
python inference.py
```

### OpenEnv Validate

```bash
openenv validate --url http://localhost:7860
```

---

## 📐 Environment Specification

### Observation Space

```json
{
  "email": {
    "id": "string",
    "subject": "string",
    "sender": "string",
    "sender_domain": "string",
    "body": "string",
    "received_at": "ISO8601 timestamp",
    "has_attachment": "boolean",
    "thread_length": "integer",
    "metadata": "object (account_tier, mrr, flags)"
  },
  "inbox_size": "integer — remaining emails",
  "step": "integer",
  "max_steps": "integer",
  "task_id": "string",
  "task_description": "string",
  "history": "list of previous action+reward records",
  "context": "object — SLA rules, escalation routing (Task 3 only)"
}
```

### Action Space

```json
{
  "action_type": "triage | draft_reply | escalate | archive | mark_spam",
  "priority": "urgent | high | medium | low | spam",
  "label": "billing | technical | complaint | inquiry | refund | account | feedback | other",
  "reply_text": "string (required for draft_reply, max 2000 chars)",
  "escalation_reason": "string (required for escalate, max 500 chars)",
  "notes": "string (optional internal note)"
}
```

### Reward Signal

Rewards are **partial credit** — not sparse:

| Field | Description |
|-------|-------------|
| `total` | Weighted sum (0.0–1.0) |
| `priority_score` | Distance-based: exact=1.0, adjacent=0.5, far=0.0 |
| `label_score` | Exact match: 1.0 or 0.0 |
| `reply_quality_score` | Keyword coverage + length + tone |
| `escalation_score` | Relevant keyword hits in escalation reason |
| `penalty` | Deducted for SLA violations (Task 3) |

---

## 📋 Tasks

### Task 1 — Basic Triage (Easy)
**Goal:** Assign correct priority + label + action type to 5 emails.

- Emails: spam, pricing inquiry, enterprise outage, billing issue, feature request
- Scoring: priority 40% + label 40% + action 20%
- Expected agent score: **0.65–0.85**

### Task 2 — Draft Replies (Medium)
**Goal:** Handle complex customer situations requiring written responses or escalations.

- Emails: 3-week-old refund complaint, API rate limit confusion, GDPR data request, fraud report, vendor outreach
- Scoring: priority 20% + label 20% + action 30% + reply quality 30%
- Reply quality graded on: keyword coverage, forbidden phrases, length
- Expected agent score: **0.45–0.65**

### Task 3 — SLA-Constrained Enterprise Triage (Hard)
**Goal:** Handle high-stakes emails with SLA rules, regulatory flags, and correct escalation routing.

- Emails: suspected data breach (financial institution, 1hr SLA), account suspension appeal, ADA accessibility complaint (legal), 2M record export (deadline), 5-year customer churn
- Scoring: priority 25% + label 15% + action 35% + escalation quality 25%
- **Penalty:** -0.15 for missing mandatory SLA escalations
- Expected agent score: **0.25–0.50**

---

## 🌐 REST API

```bash
# Health check
GET /

# List all tasks
GET /tasks

# Start an episode
POST /reset
{"task_id": "task1_basic_triage"}

# Take an action
POST /step
{
  "action": {
    "action_type": "triage",
    "priority": "urgent",
    "label": "technical"
  }
}

# Get current state
GET /state

# Action/observation space descriptions
GET /action_space
GET /observation_space
```

---

## 🔬 Grader Design

All graders are **deterministic** — same input always produces the same score.

### Priority Scoring (partial credit)
```
exact match = 1.0
1 level off  = 0.5  (e.g., high vs urgent)
2 levels off = 0.2
3+ levels    = 0.0
```

### Reply Quality Scoring
```
keyword coverage (60%): fraction of required keywords present
forbidden phrase penalty (20%): penalizes insensitive phrasing
length bonus (20%): ≥30 words = full, ≥15 words = partial
```

### Escalation Quality
```
keyword hits / total_keywords + length bonus (0.15 if ≥10 words)
capped at 1.0
```

---

## 🧪 Tests

```bash
pytest tests/test_env.py -v
```

Tests cover:
- Model validation (action type constraints)
- Grader correctness (exact and partial scores)
- Full episode runs for all 3 tasks
- SLA penalty logic
- Reward bounds (always 0.0–1.0)
- Perfect-agent score validation

---

## 📊 Baseline Scores

Baseline using `meta-llama/Llama-3.3-70B-Instruct` via HF Inference Router:

| Task | Difficulty | Baseline Score |
|------|-----------|---------------|
| task1_basic_triage | Easy | ~0.74 |
| task2_draft_replies | Medium | ~0.52 |
| task3_sla_constrained | Hard | ~0.38 |
| **Overall Average** | | **~0.55** |

Scores saved to `baseline_scores.json` after running `inference.py`.

---

## ⚙️ Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_BASE_URL` | LLM API endpoint | Yes |
| `MODEL_NAME` | Model identifier | Yes |
| `HF_TOKEN` | HuggingFace / API key | Yes |
| `API_KEY` | Alternative to HF_TOKEN | No |

---

## 🎨 Design Decisions

**Why email triage?** It's one of the most universal real-world tasks. Every organization does it. The skills transfer to: ticket routing, content moderation, legal document review, and HR screening.

**Why partial credit rewards?** Sparse rewards (0 or 1) make RL slow. Distance-based priority scoring and keyword-coverage reply scoring give the agent useful signal throughout the episode.

**Why 3 tasks with escalating complexity?** Task 1 tests classification. Task 2 tests generation. Task 3 tests reasoning under constraints — simulating what a senior support rep actually needs to know.

**Why SLA penalties?** Missing a legal/security escalation in the real world has consequences. The -0.15 penalty teaches the agent that false negatives on high-stakes emails are expensive.

---

## 📝 License

MIT License — see LICENSE file.
