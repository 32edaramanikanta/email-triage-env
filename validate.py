"""
Pre-submission validation script.
Run this before submitting to verify all checklist items pass.

Usage:
    python validate.py                          # validate local env only
    python validate.py --url http://localhost:7860   # also hit live server
"""

from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results = []


def check(name: str, passed: bool, detail: str = ""):
    icon = PASS if passed else FAIL
    msg = f"  {icon}  {name}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    results.append((name, passed))
    return passed


# ---------------------------------------------------------------------------
# 1. Models import and are typed
# ---------------------------------------------------------------------------
def validate_models():
    print("\n[1] Typed Pydantic Models")
    try:
        from models import Observation, Action, Reward, EpisodeState, ActionType, Priority, Label
        check("Observation model importable", True)
        check("Action model importable", True)
        check("Reward model importable", True)
        check("EpisodeState model importable", True)

        # Test Action validation
        from models import Action, ActionType, Priority, Label
        a = Action(action_type=ActionType.TRIAGE, priority=Priority.URGENT, label=Label.TECHNICAL)
        check("Action validation works", a.action_type == ActionType.TRIAGE)

        # Test draft_reply requires reply_text
        try:
            Action(action_type=ActionType.DRAFT_REPLY)
            check("draft_reply requires reply_text", False, "Should have raised ValueError")
        except Exception:
            check("draft_reply requires reply_text", True)

        # Test Reward range
        r = Reward(total=0.75, priority_score=0.8, label_score=0.7,
                   reply_quality_score=0.6, escalation_score=0.5, penalty=0.05)
        check("Reward model valid", 0.0 <= r.total <= 1.0)

    except Exception as e:
        check("Models import", False, str(e))


# ---------------------------------------------------------------------------
# 2. Environment API: reset / step / state
# ---------------------------------------------------------------------------
def validate_environment():
    print("\n[2] Environment API (reset / step / state)")
    try:
        from environment import EmailTriageEnv
        from models import Action, ActionType, Priority, Label

        for task_id in EmailTriageEnv.VALID_TASKS:
            env = EmailTriageEnv(task_id=task_id)

            # reset()
            obs = env.reset()
            check(f"reset() returns Observation [{task_id}]",
                  obs is not None and obs.email.id != "__done__")

            # state()
            state = env.state()
            check(f"state() returns EpisodeState [{task_id}]",
                  state is not None and state.step == 0)

            # step()
            action = Action(action_type=ActionType.TRIAGE,
                            priority=Priority.LOW, label=Label.OTHER)
            obs2, reward, done, info = env.step(action)
            check(f"step() returns (obs, reward, done, info) [{task_id}]",
                  obs2 is not None and reward is not None)
            check(f"reward in [0.0, 1.0] [{task_id}]",
                  0.0 <= reward.total <= 1.0,
                  f"got {reward.total}")

        # Test done → RuntimeError
        env2 = EmailTriageEnv("task1_basic_triage")
        env2.reset()
        env2._done = True
        try:
            env2.step(action)
            check("step() after done raises RuntimeError", False)
        except RuntimeError:
            check("step() after done raises RuntimeError", True)

    except Exception as e:
        check("Environment API", False, str(e))
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# 3. Graders: deterministic, all tasks, score in [0,1]
# ---------------------------------------------------------------------------
def validate_graders():
    print("\n[3] Graders (deterministic, 0.0–1.0)")
    try:
        from graders.graders import grade
        from models import Action, ActionType, Priority, Label
        from data.emails import TASK_REGISTRY

        for task_id, cfg in TASK_REGISTRY.items():
            for email in cfg["emails"]:
                action = Action(action_type=ActionType.TRIAGE,
                                priority=Priority.LOW, label=Label.OTHER)
                reward = grade(
                    task_id=task_id,
                    email_id=email.id,
                    action=action,
                    ground_truth=cfg["ground_truth"],
                    context=cfg.get("context", {}),
                    email_metadata=email.metadata,
                )
                in_range = 0.0 <= reward.total <= 1.0
                if not in_range:
                    check(f"Grader score in [0,1] [{task_id}/{email.id}]",
                          False, f"got {reward.total}")
                    return
            check(f"All grader scores in [0,1] [{task_id}]", True)

        # Determinism check: same input → same output twice
        from data.emails import TASK1_EMAILS, TASK1_GROUND_TRUTH
        e = TASK1_EMAILS[0]
        action = Action(action_type=ActionType.TRIAGE,
                        priority=Priority.URGENT, label=Label.TECHNICAL)
        r1 = grade("task1_basic_triage", e.id, action, TASK1_GROUND_TRUTH)
        r2 = grade("task1_basic_triage", e.id, action, TASK1_GROUND_TRUTH)
        check("Graders are deterministic", r1.total == r2.total,
              f"run1={r1.total}, run2={r2.total}")

        # Score variance check (graders don't always return same value)
        scores = set()
        for priority in [Priority.URGENT, Priority.LOW, Priority.SPAM]:
            a = Action(action_type=ActionType.TRIAGE, priority=priority, label=Label.OTHER)
            r = grade("task1_basic_triage", e.id, a, TASK1_GROUND_TRUTH)
            scores.add(r.total)
        check("Graders produce varying scores (not constant)", len(scores) > 1,
              f"unique scores: {scores}")

    except Exception as ex:
        check("Graders", False, str(ex))
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# 4. openenv.yaml exists and has required fields
# ---------------------------------------------------------------------------
def validate_yaml():
    print("\n[4] openenv.yaml")
    try:
        import yaml
        yaml_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
        check("openenv.yaml exists", os.path.exists(yaml_path))
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        required_keys = ["name", "version", "description", "tasks",
                         "observation_space", "action_space", "reward", "endpoints"]
        for k in required_keys:
            check(f"openenv.yaml has '{k}'", k in cfg)
        check("3+ tasks defined", len(cfg.get("tasks", [])) >= 3,
              f"found {len(cfg.get('tasks', []))}")
    except ImportError:
        check("openenv.yaml (yaml lib not available, skipping parse)", True,
              "Install pyyaml to fully validate")
    except Exception as e:
        check("openenv.yaml", False, str(e))


# ---------------------------------------------------------------------------
# 5. Required files exist
# ---------------------------------------------------------------------------
def validate_files():
    print("\n[5] Required Files")
    base = os.path.dirname(__file__)
    required = [
        "inference.py", "server.py", "environment.py", "models.py",
        "openenv.yaml", "Dockerfile", "requirements.txt", "README.md",
        "data/emails.py", "graders/graders.py", "tests/test_env.py",
    ]
    for f in required:
        path = os.path.join(base, f)
        check(f"File exists: {f}", os.path.exists(path))


# ---------------------------------------------------------------------------
# 6. Live server (optional)
# ---------------------------------------------------------------------------
def validate_live_server(url: str):
    print(f"\n[6] Live Server ({url})")
    try:
        import urllib.request
        import urllib.error

        # Health check
        try:
            with urllib.request.urlopen(f"{url}/", timeout=5) as r:
                check("GET / returns 200", r.status == 200)
        except Exception as e:
            check("GET / returns 200", False, str(e))
            return

        # Reset endpoint
        import json as _json
        data = _json.dumps({"task_id": "task1_basic_triage"}).encode()
        req = urllib.request.Request(
            f"{url}/reset",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                body = _json.loads(r.read())
                check("POST /reset returns Observation", "email" in body)
        except Exception as e:
            check("POST /reset returns Observation", False, str(e))

        # State endpoint
        try:
            with urllib.request.urlopen(f"{url}/state", timeout=5) as r:
                body = _json.loads(r.read())
                check("GET /state returns EpisodeState", "task_id" in body)
        except Exception as e:
            check("GET /state returns EpisodeState", False, str(e))

    except Exception as e:
        check("Live server", False, str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pre-submission validator")
    parser.add_argument("--url", default=None,
                        help="Live server URL to validate (e.g. http://localhost:7860)")
    args = parser.parse_args()

    print("=" * 60)
    print("Email Triage OpenEnv — Pre-Submission Validator")
    print("=" * 60)

    validate_files()
    validate_models()
    validate_environment()
    validate_graders()
    validate_yaml()

    if args.url:
        validate_live_server(args.url)

    # Summary
    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{total} checks passed", end="")
    if failed == 0:
        print(" 🎉 ALL CLEAR — ready to submit!")
    else:
        print(f" — {failed} failures, fix before submitting.")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
