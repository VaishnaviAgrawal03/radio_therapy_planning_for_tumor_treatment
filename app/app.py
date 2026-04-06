"""
HuggingFace Spaces — Interactive Demo
======================================
Features:
  - Trained PPO agent auto-play with heuristic fallback
  - Live reward/score chart per episode
  - Human vs Agent comparison mode
  - Episode log with color-coded score progress
"""

import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import radiotherapy_env
import gymnasium as gym

try:
    import gradio as gr
except ImportError:
    raise ImportError("pip install gradio")

# PPO import with graceful fallback
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("⚠ stable-baselines3 not found. PPO agent unavailable — using heuristic fallback.")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TASK_MAP = {
    "Prostate (Easy)":        ("prostate",         "RadiotherapyEnv-prostate-v1"),
    "Head & Neck (Medium)":   ("head_neck",         "RadiotherapyEnv-headneck-v1"),
    "Pediatric Brain (Hard)": ("pediatric_brain",   "RadiotherapyEnv-pediatricbrain-v1"),
}

# Ordered list of candidate model paths per task (tried in order, first match wins)
MODEL_CANDIDATES = {
    "prostate":        ["baseline/models/prostate_best/best_model",  "baseline/models/prostate_final"],
    "head_neck":       ["baseline/models/head_neck_best/best_model",  "baseline/models/head_neck_final"],
    "pediatric_brain": ["baseline/models/pediatric_brain_best/best_model", "baseline/models/pediatric_brain_final"],
}

ACTION_LABELS = {
    0: "Add beam",
    1: "Rotate last beam +10°",
    2: "Rotate last beam -10°",
    3: "Increase dose ↑",
    4: "Decrease dose ↓",
    5: "Remove last beam",
    6: "Fine-tune all beams",
    7: "Lock plan ✓",
}

# Trained PPO scores (from your evaluation results)
TRAINED_SCORES = {
    "prostate":        {"mean": 0.697, "pass_rate": 1.000, "std": 0.054},
    "head_neck":       {"mean": 0.750, "pass_rate": 0.967, "std": 0.059},
    "pediatric_brain": {"mean": 0.717, "pass_rate": 0.950, "std": 0.090},
}

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

session = {
    "env": None,
    "obs": None,
    "done": False,
    "step": 0,
    "total_reward": 0.0,
    "history": [],
    "reward_history": [],
    "score_history": [],
    "human_score": None,
    "agent_score": None,
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Load PPO model with smart path detection
# ─────────────────────────────────────────────────────────────────────────────

def load_model(task_key: str):
    """
    Try to load trained PPO model.
    Tries best_model first, then final, then returns None (heuristic fallback).
    """
    if not SB3_AVAILABLE:
        return None

    for path in MODEL_CANDIDATES.get(task_key, []):
        for candidate in [path + ".zip", path]:
            if os.path.exists(candidate):
                try:
                    model = PPO.load(candidate)  # SB3 handles .zip automatically
                    print(f"✓ Loaded PPO model: {candidate}")
                    return model
                except Exception as e:
                    print(f"⚠ Could not load {candidate}: {e}")

    print(f"⚠ No trained model found for '{task_key}'. Using heuristic fallback.")
    return None


def heuristic_action(obs: dict, step: int) -> int:
    """Smart heuristic fallback agent."""
    n_beams = int(np.sum(obs["beams"][:, 2]))
    constraints = obs["constraints"]
    tumor_uncov = float(constraints[0])
    max_oar_viol = float(constraints[1:].max()) if len(constraints) > 1 else 0.0

    if n_beams < 7 and step < 14:
        return 0
    elif step < 28:
        cycle = step % 4
        return [1, 2, 3, 6][cycle]
    elif step < 46:
        if tumor_uncov > 0.6:
            return 3
        elif max_oar_viol > 0.6:
            return 1 if (step % 2 == 0) else 2  # deterministic alternation
        return 6
    elif step < 56:
        if tumor_uncov > 0.4:
            return 3
        return 6
    else:
        return 7


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────

def make_env(task_name: str, render_mode: str = "rgb_array"):
    return gym.make(TASK_MAP[task_name][1], render_mode=render_mode)


def reset_env(task_name: str, seed: int = 42):
    """Reset environment — returns image, metrics, log, reward_chart."""
    if session["env"] is not None:
        session["env"].close()

    env = make_env(task_name)
    obs, info = env.reset(seed=int(seed))

    session.update({
        "env": env,
        "obs": obs,
        "done": False,
        "step": 0,
        "total_reward": 0.0,
        "history": [],
        "reward_history": [],
        "score_history": [],
    })

    frame = env.render()
    img = Image.fromarray(frame) if frame is not None else _blank_image()
    metrics = _format_metrics(obs, info, 0.0, 0, False)
    log = f"✓ Environment reset — Task: {task_name} | Seed: {seed}\nReady to plan! Click 'Watch Agent Plan' or use manual controls."

    # Empty reward chart on reset
    reward_chart = _make_reward_chart([], [])
    return img, metrics, log, reward_chart


def run_agent(task_name: str, seed: int = 42):
    """Run trained PPO agent (with heuristic fallback) for one full episode."""
    task_key = TASK_MAP[task_name][0]
    env_id   = TASK_MAP[task_name][1]

    env = gym.make(env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=int(seed))

    # Load model using correct task_key
    model = load_model(task_key)
    agent_type = "PPO (trained)" if model is not None else "Heuristic (no model found)"

    done = False
    step = 0
    reward_hist = []
    score_hist  = []
    log_lines   = [
        f"▶ Agent: {agent_type}",
        f"  Task:  {task_name}",
        f"  Seed:  {seed}",
        f"{'─'*55}",
    ]

    while not done and step < 70:
        if model is not None:
            raw_action, _ = model.predict(obs, deterministic=True)
            action = int(raw_action)
        else:
            action = heuristic_action(obs, step)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        reward_hist.append(float(reward))
        score_hist.append(float(info.get("score", 0.0)))

        log_lines.append(
            f"Step {step:2d} | {ACTION_LABELS.get(action,'?'):22s} | "
            f"r={reward:.3f} | score={info.get('score', 0.0):.3f}"
        )

    frame = env.render()
    env.close()

    img         = Image.fromarray(frame) if frame is not None else _blank_image()
    final_score = info.get("score", 0.0)
    verdict     = "✓ CLINICALLY ACCEPTABLE" if final_score >= 0.6 else "✗ Below threshold"

    # Store agent score for comparison
    session["agent_score"] = final_score

    log_lines.append(f"{'─'*55}")
    log_lines.append(f"FINAL SCORE: {final_score:.3f}  {verdict}")

    # Add known trained benchmark for context
    bench = TRAINED_SCORES.get(task_key, {})
    if bench:
        log_lines.append(
            f"Trained PPO benchmark: {bench['mean']:.3f} ± {bench['std']:.3f} "
            f"(pass rate {bench['pass_rate']*100:.0f}%)"
        )

    log = "\n".join(log_lines[-30:])
    metrics = _format_metrics(obs, info, reward, step, done)
    reward_chart = _make_reward_chart(reward_hist, score_hist)

    return img, metrics, log, reward_chart


def take_action(action_name: str):
    """Human manual action."""
    if session["env"] is None:
        empty = _make_reward_chart([], [])
        return _blank_image(), "No environment. Click 'Reset Environment' first.", "Reset first.", empty

    if session["done"]:
        empty = _make_reward_chart([], [])
        return _blank_image(), "Episode done. Reset to start again.", "Episode done.", empty

    action = next((k for k, v in ACTION_LABELS.items() if v == action_name), 0)

    obs, reward, terminated, truncated, info = session["env"].step(action)
    session["obs"] = obs
    session["step"] += 1
    session["total_reward"] += reward
    session["done"] = terminated or truncated
    session["reward_history"].append(float(reward))
    session["score_history"].append(float(info.get("score", 0.0)))

    frame = session["env"].render()
    img = Image.fromarray(frame) if frame is not None else _blank_image()
    metrics = _format_metrics(obs, info, reward, session["step"], session["done"])

    score = info.get("score", 0.0)
    score_bar = _score_bar(score)
    log_line = (
        f"Step {session['step']:2d} | {action_name:22s} | "
        f"r={reward:.3f} | score={score:.3f} {score_bar}"
    )
    session["history"].append(log_line)

    if session["done"]:
        final_score = info.get("score", 0.0)
        session["human_score"] = final_score
        verdict = "✓ ACCEPTED!" if final_score >= 0.6 else "✗ Below threshold"
        session["history"].append(f"{'─'*55}")
        session["history"].append(f"YOUR FINAL SCORE: {final_score:.3f}  {verdict}")

        # Comparison with agent if available
        if session["agent_score"] is not None:
            diff = final_score - session["agent_score"]
            winner = "YOU WIN! 🎉" if diff > 0 else f"Agent wins by {abs(diff):.3f}"
            session["history"].append(f"vs Agent score: {session['agent_score']:.3f} → {winner}")

    log = "\n".join(session["history"][-20:])
    reward_chart = _make_reward_chart(session["reward_history"], session["score_history"])
    return img, metrics, log, reward_chart


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_bar(score: float) -> str:
    """Mini ASCII progress bar for score."""
    filled = int(score * 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"|{bar}|"


def _make_reward_chart(reward_hist: list, score_hist: list):
    """
    Generate a matplotlib figure showing reward + score over steps.
    Returns a PIL Image.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), facecolor="#0d1117")

        steps = list(range(1, len(reward_hist) + 1))

        # Left: Reward per step
        ax1.set_facecolor("#0d1117")
        if reward_hist:
            ax1.plot(steps, reward_hist, color="#00d4aa", linewidth=2, marker="o",
                     markersize=3, label="Reward")
            ax1.fill_between(steps, reward_hist, alpha=0.15, color="#00d4aa")
            ax1.axhline(y=np.mean(reward_hist), color="#EF9F27", linestyle="--",
                        linewidth=1, alpha=0.7, label=f"Mean: {np.mean(reward_hist):.3f}")
        ax1.set_xlim(left=1)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlabel("Step", color="#888", fontsize=9)
        ax1.set_ylabel("Reward", color="#888", fontsize=9)
        ax1.set_title("Reward per step", color="white", fontsize=10, pad=8)
        ax1.tick_params(colors="#666")
        ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", edgecolor="#333")
        ax1.grid(True, alpha=0.1, color="#444")
        for sp in ax1.spines.values():
            sp.set_edgecolor("#333")

        # Right: Cumulative score progress
        ax2.set_facecolor("#0d1117")
        if score_hist:
            ax2.plot(steps, score_hist, color="#7F77DD", linewidth=2, marker="s",
                     markersize=3, label="Plan score")
            ax2.fill_between(steps, score_hist, alpha=0.15, color="#7F77DD")
            ax2.axhline(y=0.6, color="#E24B4A", linestyle="--", linewidth=1.5,
                        alpha=0.8, label="Clinical threshold (0.6)")
            final = score_hist[-1]
            color = "#1D9E75" if final >= 0.6 else "#E24B4A"
            ax2.scatter([steps[-1]], [final], color=color, s=80, zorder=5)
            ax2.annotate(f"{final:.3f}", (steps[-1], final),
                         textcoords="offset points", xytext=(6, 4),
                         fontsize=9, color=color)
        ax2.set_xlim(left=1)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel("Step", color="#888", fontsize=9)
        ax2.set_ylabel("Score", color="#888", fontsize=9)
        ax2.set_title("Plan score progress", color="white", fontsize=10, pad=8)
        ax2.tick_params(colors="#666")
        ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", edgecolor="#333")
        ax2.grid(True, alpha=0.1, color="#444")
        for sp in ax2.spines.values():
            sp.set_edgecolor("#333")

        plt.tight_layout(pad=1.2)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="#0d1117")
        buf.seek(0)
        from PIL import Image as PILImage
        img = PILImage.open(buf).convert("RGB")
        plt.close(fig)
        buf.close()
        return img

    except Exception as e:
        print(f"Chart error: {e}")
        return _blank_image(height=280, width=800)


def _format_metrics(obs, info, last_reward, step, done):
    dvh_summary = info.get("dvh_summary", {})
    score = info.get("score", 0.0)
    n_beams = info.get("n_beams", 0)

    score_bar = _score_bar(score)
    lines = [
        f"{'─'*38}",
        f"  Step:         {step}",
        f"  Beams placed: {n_beams} / 7",
        f"  Last reward:  {last_reward:.3f}",
        f"  Plan score:   {score:.3f} {'✓' if score >= 0.6 else '…'}",
        f"  {score_bar}",
        f"{'─'*38}",
    ]

    if dvh_summary:
        tc = dvh_summary.get("tumor_coverage", 0)
        lines.append(f"  Tumor coverage: {tc*100:.1f}%  (target ≥ 95%)")
        lines.append("")
        for key, val in dvh_summary.items():
            if "mean" in key and "oar" in key:
                oar_name = key.replace("oar_", "").replace("_mean", "").replace("_", " ").title()
                lines.append(f"  {oar_name:<18} mean: {val:.3f}")

    if done:
        lines.append(f"{'─'*38}")
        lines.append(f"  EPISODE COMPLETE")
        if score >= 0.6:
            lines.append(f"  ✓ CLINICALLY ACCEPTABLE PLAN")
        else:
            lines.append(f"  ✗ Plan quality below threshold")

    return "\n".join(lines)


def _blank_image(height: int = 400, width: int = 900):
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI — improved layout
# ─────────────────────────────────────────────────────────────────────────────

DESCRIPTION = """
# 🎯 RadiotherapyPlanningEnv — OpenEnv RL Environment

**An RL environment where an AI agent learns to plan cancer radiotherapy treatment.**

The agent places radiation beams to maximize tumor dose while protecting critical organs — a real clinical problem that takes human experts **2–4 hours per patient**.

*Built for the Meta × Scaler PyTorch OpenEnv Hackathon*
"""

# Benchmark summary for display
BENCHMARK_MD = """
### 📊 Trained PPO Benchmark Results

| Task | Mean Score | Pass Rate | Timesteps |
|------|-----------|-----------|-----------|
| Prostate (Easy) | **0.697** ± 0.054 | 100% | 200K |
| Head & Neck (Medium) | **0.750** ± 0.059 | 96.7% | 350K |
| Pediatric Brain (Hard) | **0.717** ± 0.090 | 95.0% | 1M |
| **Aggregate** | **0.721** | **97.2%** | — |

> Score ≥ 0.6 = clinically acceptable plan. Pediatric Brain is the hardest case in clinical radiotherapy.
"""

with gr.Blocks(title="RadiotherapyPlanningEnv") as demo:

    gr.Markdown(DESCRIPTION)

    # ── Controls row ──────────────────────────────────────────────────────────
    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=list(TASK_MAP.keys()),
            value="Prostate (Easy)",
            label="Task (difficulty)",
        )
        seed_slider = gr.Slider(0, 999, value=42, step=1, label="Random seed")

    with gr.Row():
        reset_btn = gr.Button("🔄 Reset Environment", variant="secondary")
        agent_btn = gr.Button("🤖 Watch PPO Agent Plan", variant="primary")

    # ── Main visualization ────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            viz_output = gr.Image(label="Dose Distribution + DVH", type="pil")
        with gr.Column(scale=1):
            metrics_output = gr.Textbox(
                label="Plan Metrics", lines=20, max_lines=22,
                value="Click 'Reset Environment' to start.",
            )

    # ── NEW: Reward curve chart ───────────────────────────────────────────────
    reward_chart_output = gr.Image(
        label="📈 Reward & Score Progress (updates after each episode)",
        type="pil",
    )

    # ── Manual control ────────────────────────────────────────────────────────
    gr.Markdown("### 🕹️ Manual Control — Play as the Agent")
    gr.Markdown("*Try to beat the PPO agent! Your score vs agent score will be shown at the end.*")

    with gr.Row():
        action_dropdown = gr.Dropdown(
            choices=list(ACTION_LABELS.values()),
            value="Add beam",
            label="Choose action",
        )
        step_btn = gr.Button("▶ Take Action", variant="primary")

    log_output = gr.Textbox(
        label="Episode Log",
        lines=14,
        max_lines=16,
        value="Reset the environment to start.",
    )

    # ── Benchmark + reference ─────────────────────────────────────────────────
    with gr.Accordion("📊 Benchmark Results & Scoring Reference", open=False):
        gr.Markdown(BENCHMARK_MD)
        gr.Markdown("""
        ### Action Reference
        | Action | Effect |
        |--------|--------|
        | Add beam | Place a new beam aimed at the tumor center |
        | Rotate last beam ±10° | Adjust angle of most recent beam |
        | Increase/Decrease dose | Change dose weight of last beam |
        | Fine-tune all beams | Small random improvement to all beams |
        | Lock plan ✓ | Finalize plan and terminate episode |

        ### Reward Formula
        ```
        reward = tumor_coverage × 0.50
               − oar_penalty    × 0.40   (priority-weighted)
               + plan_efficiency × 0.10
        ```
        ### Final Score Formula
        ```
        score = tumor_coverage × 0.55
              − oar_penalty    × 0.40
              + efficiency     × 0.05
        ```
        Score ≥ 0.6 = clinically acceptable treatment plan.
        """)

    # ── Wire up events ────────────────────────────────────────────────────────
    reset_btn.click(
        reset_env,
        inputs=[task_dropdown, seed_slider],
        outputs=[viz_output, metrics_output, log_output, reward_chart_output],
    )
    agent_btn.click(
        run_agent,
        inputs=[task_dropdown, seed_slider],
        outputs=[viz_output, metrics_output, log_output, reward_chart_output],
    )
    step_btn.click(
        take_action,
        inputs=[action_dropdown],
        outputs=[viz_output, metrics_output, log_output, reward_chart_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
