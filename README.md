---
title: RadiotherapyPlanningEnv
emoji: "☢️"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# RadiotherapyPlanningEnv — OpenEnv RL Environment

An RL environment where an AI agent learns to plan cancer radiotherapy treatment. The agent places radiation beams to maximize tumor dose while protecting surrounding organs-at-risk (OARs) — a real clinical problem that takes human experts **2-4 hours per patient**.

Built for the **Meta x Scaler PyTorch OpenEnv Hackathon**.

- **Live Demo**: [HuggingFace Space](https://huggingface.co/spaces/VaishnaviAgrawal/RadiotherapyPlanningEnv)
- **Repository**: [GitHub](https://github.com/VaishnaviAgrawal03/radio_therapy_planning_for_tumor_treatment)

---

## Clinical Motivation

~14 million cancer patients per year require radiotherapy. A radiation oncologist must decide:
- How many beams to use
- At what angles
- With what dose intensity

...while ensuring the tumor receives enough radiation and nearby healthy organs stay below safe limits. This environment simulates that decision-making process for RL agents.

---

## Tasks (3 Difficulty Levels)

| Task | Gym ID | Difficulty | OARs | Max Steps | Clinical Context |
|------|--------|-----------|------|-----------|-----------------|
| **Prostate** | `RadiotherapyEnv-prostate-v1` | Easy | 2 (rectum, bladder) | 50 | Clear geometry, well-separated organs |
| **Head & Neck** | `RadiotherapyEnv-headneck-v1` | Medium | 7 (spinal cord, brainstem, parotids, etc.) | 60 | Complex anatomy, many competing constraints |
| **Pediatric Brain** | `RadiotherapyEnv-pediatricbrain-v1` | Hard | 5 (brainstem 2-3mm away) | 70 | Near-zero margin for error, catastrophic brainstem penalty |

---

## Action Space — `Discrete(8)`

| Action | Description |
|--------|-------------|
| 0 | Add beam at next default angle |
| 1 | Rotate last beam +10 degrees |
| 2 | Rotate last beam -10 degrees |
| 3 | Increase last beam dose weight by 10% |
| 4 | Decrease last beam dose weight by 10% |
| 5 | Remove last beam |
| 6 | Fine-tune all beams (small random perturbation) |
| 7 | Lock plan (terminate episode) |

---

## Observation Space — `Dict`

| Key | Shape | Description |
|-----|-------|-------------|
| `dvh_tumor` | `Box(50,)` | Cumulative Dose-Volume Histogram for tumor |
| `dvh_oar` | `Box(3, 50)` | DVH for top 3 organs-at-risk |
| `beams` | `Box(7, 3)` | Per-beam: `[angle/180, dose_weight, is_active]` |
| `constraints` | `Box(4,)` | Normalized constraint violations `[tumor, oar1, oar2, oar3]` |
| `step_frac` | `Box(1,)` | Episode progress fraction `[0, 1]` |

All observations normalized to `[0, 1]` for neural network training stability.

---

## Reward Function

Dense per-step reward in `[0.0, 1.0]`:

```
reward = tumor_coverage x 0.55 - oar_penalty x 0.40 + plan_efficiency x 0.05
```

- **Tumor coverage (55%)**: D95 metric + fraction of tumor receiving >= 95% prescription dose
- **OAR penalty (40%)**: Priority-weighted organ violations (critical organs penalized 1.5x)
- **Plan efficiency (5%)**: Optimal beam count around 5-7

The reward function provides **meaningful partial progress signals** — reward increases when beams improve coverage and decreases when organs are violated. This is separate from the stricter `compute_score()` used for final grading.

---

## Physics Model

Gaussian pencil-beam dose calculation:

```
beam_dose = lateral_profile x depth_attenuation x dose_weight x BEAM_SCALE
```

- **Lateral profile**: Gaussian falloff from beam central axis
- **Depth attenuation**: Exponential decay through tissue (Beer-Lambert Law)
- **Beam superposition**: Total dose = sum of all beam contributions
- **Isocenter convergence**: All beams aimed at tumor center

Simplified from clinical Monte Carlo (milliseconds vs hours) but preserves the core trade-off: multiple beams overlap at tumor for high dose while surrounding organs receive minimal radiation.

---

## Quick Start

```python
import gymnasium as gym
import radiotherapy_env

# Easy task
env = gym.make("RadiotherapyEnv-prostate-v1", render_mode="rgb_array")
obs, info = env.reset(seed=42)

for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(f"Final score: {info['score']:.3f}")
env.close()
```

---

## Installation

```bash
# From source
git clone https://github.com/VaishnaviAgrawal03/radio_therapy_planning_for_tumor_treatment.git
cd radio_therapy_planning_for_tumor_treatment
pip install -e .

# With training support (PPO)
pip install -e ".[training]"

# With inference support (LLM agent)
pip install -e ".[inference]"

# With demo (Gradio)
pip install -e ".[demo]"
```

---

## Running the Inference Script

The inference script connects an LLM to the environment via OpenAI-compatible API:

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export API_KEY="your_api_key"
python inference.py
```

Output follows the required `[START]`, `[STEP]`, `[END]` format:

```
[START] task=prostate env=RadiotherapyEnv-prostate-v1 model=llama-3.3-70b-versatile
[STEP] step=1 action=add_beam reward=0.03 done=false error=null
[STEP] step=2 action=add_beam reward=0.07 done=false error=null
...
[END] success=true steps=50 score=0.635 rewards=0.03,0.07,...
```

---

## Baseline Results

**PPO Agent** (stable-baselines3, MultiInputPolicy):

| Task | Mean Score | Std | Pass Rate | Training Steps |
|------|-----------|-----|-----------|---------------|
| Prostate | 0.697 | 0.054 | 100% | 200K |
| Head & Neck | 0.750 | 0.059 | 96.7% | 350K |
| Pediatric Brain | 0.717 | 0.090 | 95.0% | 1M |
| **Aggregate** | **0.721** | — | — | — |

**LLM Agent** (Llama 3.3 70B via inference.py):

| Task | Mean Score | Episodes |
|------|-----------|----------|
| Prostate | 0.540 | 3 |
| Head & Neck | 0.560 | 3 |
| Pediatric Brain | 0.523 | 3 |
| **Aggregate** | **0.541** | — |

---

## Auto-Grader

```python
from radiotherapy_env.reward.grader import grade_all

def my_agent(obs, env):
    # Your agent logic here
    return env.action_space.sample()

results = grade_all(my_agent, n_episodes=20, seed=42)
print(f"Aggregate: {results['aggregate_score']:.3f}")
```

Each grader produces scores in `[0.0, 1.0]`, deterministic with seed, pass threshold at 0.60.

---

## Docker

```bash
docker build -t radiotherapy-env:latest .
docker run -p 7860:7860 radiotherapy-env:latest
```

---

## Tests

```bash
pytest tests/ -v
# 25 tests across: Gymnasium compliance, physics, reward, task difficulty
```

---

## Repository Structure

```
radiotherapy-env/
├── inference.py               # LLM inference script (hackathon requirement)
├── openenv.yaml               # OpenEnv spec metadata
├── Dockerfile                 # Container build
├── radiotherapy_env/          # Main package
│   ├── env.py                 # Core RadiotherapyEnv class
│   ├── physics/               # Dose calculation + patient models
│   ├── tasks/                 # 3 task definitions (prostate, head_neck, pediatric_brain)
│   ├── reward/                # Reward function + scoring + auto-grader
│   └── rendering/             # Dose heatmap + DVH visualization
├── baseline/                  # PPO training + evaluation + saved models
├── app/                       # Gradio interactive demo
└── tests/                     # Comprehensive test suite
```

---

## Author

**Vaishnavi Agrawal** — vagrawal_be22@thapar.edu
