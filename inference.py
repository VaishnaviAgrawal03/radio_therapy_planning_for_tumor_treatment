"""
Inference Script — RadiotherapyPlanningEnv
==========================================
Connects an LLM (via OpenAI-compatible API) to the RadiotherapyPlanningEnv
and runs it through all 3 tasks (prostate, head_neck, pediatric_brain).

Required env vars:
    API_BASE_URL  — LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME    — Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      — HuggingFace / API key

Usage:
    python inference.py
"""

import os
import textwrap
from typing import List, Optional

import numpy as np
import gymnasium as gym
from openai import OpenAI

import radiotherapy_env  # noqa: F401 — registers gym envs

# ── Configuration ─────────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TEMPERATURE = 0.3
MAX_TOKENS = 100

# Task configurations
TASKS = [
    {"key": "prostate",        "env_id": "RadiotherapyEnv-prostate-v1",       "max_steps": 50, "difficulty": "easy"},
    {"key": "head_neck",       "env_id": "RadiotherapyEnv-headneck-v1",       "max_steps": 60, "difficulty": "medium"},
    {"key": "pediatric_brain", "env_id": "RadiotherapyEnv-pediatricbrain-v1", "max_steps": 70, "difficulty": "hard"},
]

N_EPISODES = 3  # episodes per task (kept low for < 20 min runtime)
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI radiation oncologist planning cancer treatment.
    You control radiation beams on a 2D patient anatomy grid to maximize tumor
    dose while protecting organs-at-risk (OARs).

    Available actions (reply with ONLY the action number, nothing else):
      0 — Add a new beam at the next default angle
      1 — Rotate the last beam +10 degrees
      2 — Rotate the last beam -10 degrees
      3 — Increase the last beam's dose weight by 10%
      4 — Decrease the last beam's dose weight by 10%
      5 — Remove the last beam
      6 — Fine-tune all beams (small random adjustments)
      7 — Lock plan and finish (use only when satisfied)

    Strategy guidelines:
      - First, add 5-6 beams (action 0) to build coverage
      - Then adjust angles (actions 1,2) to avoid organs with high violations
      - Increase dose (action 3) if tumor coverage is low
      - Decrease dose (action 4) if organ violations are high
      - Fine-tune (action 6) for final optimization
      - Lock plan (action 7) when constraints look good or near the end

    IMPORTANT: Reply with ONLY a single digit (0-7). No explanation, no text.
""").strip()


# ── Logging (exact format required by hackathon) ─────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Observation formatting ───────────────────────────────────────────────────

def format_observation(obs: dict, info: dict, step: int, max_steps: int) -> str:
    """Convert numeric observation into readable text for the LLM."""
    # Beam info
    beams_desc = []
    for i in range(7):
        angle_norm, dose_weight, active = obs["beams"][i]
        if active > 0.5:
            angle_deg = angle_norm * 180.0
            beams_desc.append(f"  Beam {i+1}: angle={angle_deg:.0f}deg, dose_weight={dose_weight:.2f}")
    beams_text = "\n".join(beams_desc) if beams_desc else "  No beams placed yet"
    n_beams = len(beams_desc)

    # Constraints
    c = obs["constraints"]
    tumor_uncoverage = c[0]
    oar1_violation = c[1]
    oar2_violation = c[2]
    oar3_violation = c[3]

    # DVH summary from info if available
    dvh_text = ""
    if "dvh_summary" in info:
        dvh = info["dvh_summary"]
        tumor_cov = dvh.get("tumor_coverage", 0.0)
        tumor_d95 = dvh.get("tumor_d95", 0.0)
        dvh_text = f"  Tumor D95: {tumor_d95:.3f} (target: >=0.95)\n  Tumor coverage: {tumor_cov:.1%}"

    # Score
    current_score = info.get("score", 0.0)

    return textwrap.dedent(f"""
        Step {step}/{max_steps}  |  Beams active: {n_beams}/7  |  Current score: {current_score:.3f}

        Beams:
        {beams_text}

        Constraints (0.0 = perfect, higher = worse):
          Tumor uncoverage: {tumor_uncoverage:.3f}
          OAR 1 violation:  {oar1_violation:.3f}
          OAR 2 violation:  {oar2_violation:.3f}
          OAR 3 violation:  {oar3_violation:.3f}

        {dvh_text}

        Choose action (0-7):
    """).strip()


# ── LLM interaction ──────────────────────────────────────────────────────────

def get_llm_action(client: OpenAI, obs: dict, info: dict, step: int,
                   max_steps: int, history: List[str]) -> int:
    """Ask the LLM to choose an action given the current state."""
    user_prompt = format_observation(obs, info, step, max_steps)

    # Add recent history for context
    if history:
        recent = "\n".join(history[-5:])
        user_prompt = f"Recent actions:\n{recent}\n\n{user_prompt}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Parse action — find first digit 0-7 in response
        for char in text:
            if char.isdigit() and int(char) <= 7:
                return int(char)

        # Fallback: add beam early, fine-tune mid, lock late
        if step < max_steps * 0.6:
            return 0  # add beam
        elif step < max_steps * 0.85:
            return 6  # fine-tune
        else:
            return 7  # lock plan

    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        # Deterministic fallback strategy
        if step <= 6:
            return 0  # add beams
        elif step < max_steps - 2:
            return 6  # fine-tune
        else:
            return 7  # lock plan


# ── Episode runner ───────────────────────────────────────────────────────────

def run_episode(client: OpenAI, env_id: str, task_key: str, max_steps: int,
                seed: int) -> dict:
    """Run one episode: LLM plays the radiotherapy env."""
    env = gym.make(env_id)
    obs, info = env.reset(seed=seed)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_key, env=env_id, model=MODEL_NAME)

    try:
        for step in range(1, max_steps + 1):
            action = get_llm_action(client, obs, info, step, max_steps, history)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rewards.append(float(reward))
            steps_taken = step

            action_names = [
                "add_beam", "rotate_+10", "rotate_-10",
                "increase_dose", "decrease_dose", "remove_beam",
                "fine_tune", "lock_plan"
            ]
            action_str = action_names[action]

            log_step(step=step, action=action_str, reward=float(reward),
                     done=done, error=None)

            history.append(f"Step {step}: {action_str} -> reward {reward:.2f}")

            if done:
                break

        final_score = info.get("score", 0.0)
        success = final_score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        final_score = 0.0
        success = False

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=final_score,
                rewards=rewards)

    return {
        "task": task_key,
        "score": final_score,
        "steps": steps_taken,
        "rewards": rewards,
        "success": success,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("[ERROR] No API key found. Set HF_TOKEN or API_KEY environment variable.", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_results = {}

    for task_cfg in TASKS:
        task_key = task_cfg["key"]
        env_id = task_cfg["env_id"]
        max_steps = task_cfg["max_steps"]
        difficulty = task_cfg["difficulty"]

        print(f"\n{'='*60}", flush=True)
        print(f"  Task: {task_key} ({difficulty})", flush=True)
        print(f"  Running {N_EPISODES} episodes, max {max_steps} steps each", flush=True)
        print(f"{'='*60}\n", flush=True)

        task_scores = []
        for ep in range(N_EPISODES):
            print(f"\n--- Episode {ep+1}/{N_EPISODES} ---\n", flush=True)
            result = run_episode(client, env_id, task_key, max_steps, seed=42 + ep)
            task_scores.append(result["score"])

        mean_score = float(np.mean(task_scores))
        all_results[task_key] = {
            "difficulty": difficulty,
            "mean_score": mean_score,
            "scores": task_scores,
            "pass_rate": float(np.mean(np.array(task_scores) >= 0.6)),
        }

        print(f"\n  {task_key} mean score: {mean_score:.3f}", flush=True)

    # Final summary
    aggregate = float(np.mean([r["mean_score"] for r in all_results.values()]))
    print(f"\n{'='*60}", flush=True)
    print(f"  AGGREGATE SCORE: {aggregate:.3f}", flush=True)
    for key, res in all_results.items():
        print(f"    {key}: {res['mean_score']:.3f} (pass rate: {res['pass_rate']*100:.0f}%)", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
