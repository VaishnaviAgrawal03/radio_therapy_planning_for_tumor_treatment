# RadiotherapyPlanningEnv — Complete Project Context

> Comprehensive onboarding document. Covers everything from clinical motivation to deployment, judging criteria, and every file in the codebase.

---

## 1. What This Project Is

**RadiotherapyPlanningEnv** is a Gymnasium-compatible RL environment for cancer radiotherapy treatment planning, built for the **Meta x Scaler PyTorch OpenEnv Hackathon**.

An AI agent learns to place and optimize radiation beams to maximize tumor dose while protecting surrounding organs-at-risk (OARs) — a real clinical problem that takes human experts **2–4 hours per patient** (~14 million patients/year).

**This is NOT a clinical tool.** It's a **benchmark environment** — a simplified but physically grounded simulation for testing RL algorithms on a meaningful problem.

**Author:** Vaishnavi Agrawal (vagrawal_be22@thapar.edu)
**Version:** 1.0.0
**License:** MIT
**Python:** >=3.10

---

## 2. Hackathon Context & Judging

### What We Submit
- **GitHub repo:** https://github.com/VaishnaviAgrawal03/radio_therapy_planning_for_tumor_treatment
- **HuggingFace Space:** https://huggingface.co/spaces/VaishnaviAgrawal/RadiotherapyPlanningEnv
- **Docker image:** ghcr.io/vaishnaviagrawal03/radiotherapy-env:latest

### How Judging Works

**Phase 1: Automated Validation (pass/fail gate)**
- HF Space deploys and responds to `/reset` with 200
- `openenv validate` passes (checks pyproject.toml, uv.lock, server/app.py, openenv-core dep)
- Dockerfile builds
- `inference.py` runs and produces scores
- 3+ tasks with graders that produce scores in [0.0, 1.0]

**Phase 2: Agentic Evaluation (scored)**
- Judges re-run baseline agent
- Standard Open LLM agent (e.g., Nemotron 3 Super) run against the environment
- Score variance checked across episodes

**Phase 3: Human Review**
- Top submissions reviewed by Meta and HuggingFace engineers
- Evaluated for real-world utility, creativity, and exploit checks

### Scoring Breakdown

| Category | Weight | What judges look for |
|----------|--------|---------------------|
| **Real-world utility** | 30% | Clinical domain, fills a gap, immediate value for RL community |
| **Task & grader quality** | 25% | 3+ tasks, difficulty range, deterministic graders, hard task challenges frontier models |
| **Environment design** | 20% | Clean reset(), well-designed spaces, dense reward signal, sensible episodes |
| **Code quality & spec** | 15% | openenv validate passes, Docker builds, HF Space deploys, baseline reproduces |
| **Creativity & novelty** | 10% | Novel domain, interesting reward design, engaging mechanics |

### Pre-Submission Checklist

| Requirement | Status |
|------------|--------|
| HF Space deploys | Done |
| `/reset` returns 200 | Done |
| `openenv validate` passes | Done |
| Dockerfile builds | Done |
| `inference.py` runs and produces scores | Done (aggregate 0.541) |
| 3+ tasks with graders | Done (prostate, head_neck, pediatric_brain) |
| Scores/rewards in [0.0, 1.0] range | Done |
| Runs under 20 min on 2 vCPU / 8GB RAM | Done |

### Mandatory Environment Variables for Inference

```
API_BASE_URL   — LLM API endpoint
MODEL_NAME     — Model identifier
HF_TOKEN       — HuggingFace / API key
```

### Required Log Format

```
[START] task={task} env={env_id} model={model_name}
[STEP] step={n} action={action_str} reward={reward:.2f} done={bool} error={error_or_null}
[END] success={bool} steps={n} score={score:.3f} rewards={r1,r2,...}
```

---

## 3. Repository Structure

```
radiotherapy-env/
├── inference.py               # LLM inference script (HACKATHON REQUIREMENT)
├── openenv.yaml               # OpenEnv spec metadata
├── pyproject.toml             # Python package config (required by openenv validate)
├── setup.py                   # Legacy package config
├── uv.lock                    # Dependency lock file (required by openenv validate)
├── requirements.txt           # Pinned dependencies
├── Dockerfile                 # Container build (port 7860, ~800MB)
├── DEPLOY.sh                  # 11-step deployment guide
├── README.md                  # Project docs + HF Space YAML frontmatter
├── PROJECT_CONTEXT.md         # This file
│
├── radiotherapy_env/          # Main Python package
│   ├── __init__.py            # Gymnasium env registration (3 envs)
│   ├── env.py                 # Core RadiotherapyEnv class
│   ├── physics/
│   │   ├── dose_calculator.py # Pencil-beam dose model
│   │   ├── dvh.py             # Dose-Volume Histogram computation
│   │   └── phantom.py         # Patient anatomy + Beam dataclasses + generators
│   ├── tasks/
│   │   ├── __init__.py        # TASK_REGISTRY dict
│   │   ├── base_task.py       # Abstract base class
│   │   ├── prostate.py        # Easy (2 OARs)
│   │   ├── head_neck.py       # Medium (7 OARs, 0.95× difficulty scaling)
│   │   └── pediatric_brain.py # Hard (brainstem penalty tiers)
│   ├── reward/
│   │   ├── reward_fn.py       # compute_reward() + compute_score()
│   │   └── grader.py          # Auto-grading for leaderboard
│   └── rendering/
│       └── dose_heatmap.py    # RGB visualization (dose + DVH)
│
├── server/                    # OpenEnv HTTP server (required by openenv validate)
│   ├── __init__.py
│   ├── app.py                 # FastAPI endpoints: /reset, /step, /state, /health, /metadata, /schema
│   ├── models.py              # Pydantic: RadiotherapyAction, RadiotherapyObservation
│   └── radiotherapy_environment.py  # Gymnasium → HTTP bridge
│
├── baseline/
│   ├── train_ppo.py           # PPO training with stable-baselines3
│   ├── evaluate.py            # Agent evaluation + heuristic baseline
│   ├── results.json           # Aggregate benchmark results
│   ├── results_prostate.json
│   ├── results_head_neck.json
│   ├── results_pediatric_brain.json
│   ├── logs/                  # TensorBoard event files
│   └── models/                # Saved PPO model checkpoints
│       ├── prostate_best/best_model.zip
│       ├── head_neck_best/best_model.zip
│       └── pediatric_brain_best/best_model.zip
│
├── app/
│   └── app.py                 # Gradio interactive demo
│
└── tests/
    └── test_env.py            # 25 pytest tests
```

---

## 4. Gymnasium Environment

### Registration (`radiotherapy_env/__init__.py`)

Three environments registered via `gym.register()`:

| Gym ID | Task | Max Steps | Difficulty |
|--------|------|-----------|------------|
| `RadiotherapyEnv-prostate-v1` | prostate | 50 | Easy |
| `RadiotherapyEnv-headneck-v1` | head_neck | 60 | Medium |
| `RadiotherapyEnv-pediatricbrain-v1` | pediatric_brain | 70 | Hard |

### Core Environment (`radiotherapy_env/env.py`)

**Class:** `RadiotherapyEnv(gym.Env)`

**Class Constants:**
- `MAX_BEAMS = 7` — maximum radiation beams per plan
- `GRID_SIZE = 64` — patient phantom grid resolution
- `_LOCK_PLAN_ACTION = 7` — action index that terminates the episode

**Constructor:** `__init__(task="prostate", max_steps=50, render_mode=None)`

**Instance Fields:**
- `self.dose_calculator` — `DoseCalculator(grid_size=64)`
- `self.dvh_calculator` — `DVHCalculator(n_bins=50)`
- `self.patient` — current `PatientPhantom` (set in reset)
- `self.beams` — list of `Beam` objects
- `self.step_count` — steps taken
- `self.current_dose` — 64×64 dose grid
- `self._last_reward` — reward from previous step

### Action Space — `Discrete(8)`

| Action | Description | Implementation Detail |
|--------|-------------|----------------------|
| 0 | Add beam at next default angle | Angle: `len(beams) × (180/7) + noise(±5°)`, weight: 0.6 |
| 1 | Rotate last beam +10° | `(angle + 10) % 180` |
| 2 | Rotate last beam -10° | `(angle - 10) % 180` |
| 3 | Increase dose weight | `min(1.0, weight + 0.1)` |
| 4 | Decrease dose weight | `max(0.1, weight - 0.1)` |
| 5 | Remove last beam | `beams.pop()` |
| 6 | Fine-tune all beams | All beams: angle ±3°, weight ±0.05 |
| 7 | Lock plan (terminate) | No-op; termination handled in `step()` |

### Observation Space — `Dict`

| Key | Shape | Range | Description |
|-----|-------|-------|-------------|
| `dvh_tumor` | Box(50,) | [0, 1] | Cumulative DVH for tumor (50 bins) |
| `dvh_oar` | Box(3, 50) | [0, 1] | DVH for top 3 OARs |
| `beams` | Box(7, 3) | [0, 1] | `[angle/180, dose_weight, is_active]` per beam |
| `constraints` | Box(4,) | [0, 2] | `[tumor_uncoverage, oar1_violation, oar2_violation, oar3_violation]` |
| `step_frac` | Box(1,) | [0, 1] | `step_count / max_steps` |

### Key Methods

| Method | Returns | Purpose |
|--------|---------|---------|
| `reset(seed)` | `(obs, info)` | Sample new patient, initialize beams, compute dose |
| `step(action)` | `(obs, reward, terminated, truncated, info)` | Apply action, recompute dose, return reward |
| `state()` | `Dict` | Full state for OpenEnv spec (patient, beams, dose_grid, score) |
| `render()` | `np.ndarray (H,W,3)` | RGB visualization via `render_heatmap()` |
| `get_score()` | `float [0,1]` | Final plan quality via `compute_score()` |
| `get_dvh_summary()` | `Dict` | DVH metrics (D95, coverage, OAR means/maxes) |

### Episode Flow

```
reset() → sample patient → init beams=[] → compute dose=zeros → return obs
  ↓
step(action) → _apply_action() → recompute dose → compute reward → check termination → return obs
  ↓
Terminates when: action==7 (lock plan) OR step_count >= max_steps
```

### Constraint Violations (`_get_constraint_violations()`)

- `violations[0]` = 1.0 - tumor_coverage (0=perfect, 1=no coverage)
- `violations[1-3]` = max(0, (mean_oar_dose - limit) / limit) per OAR

---

## 5. Physics Layer

### Dose Calculator (`radiotherapy_env/physics/dose_calculator.py`)

**Class:** `DoseCalculator(grid_size=64, beam_width_sigma=4.0, attenuation_mu=0.012, prescription_dose=1.0)`

**Class Constant:** `BEAM_SCALE = 0.40` — calibrated so 7 converging beams ≈ 1.0 Gy at isocenter

**Pre-computed:** Coordinate meshgrids `_cx`, `_cy` cached in `__init__`

#### `compute(patient, beams) → np.ndarray (64×64)`

Superposition of single-beam contributions. All beams converge at `patient.tumor_center` (isocenter), not grid center.

#### `_compute_single_beam(beam, body_mask, isocenter) → np.ndarray (64×64)`

Gaussian pencil-beam model:

```
beam_dose = profile × attenuation × dose_weight × BEAM_SCALE × body_mask
```

1. **Coordinate transform:** Shift grid to isocenter-centered system
2. **Lateral distance:** Perpendicular distance from beam axis → `lateral = -cx×sin(a) + cy×cos(a)`
3. **Gaussian profile:** `exp(-0.5 × (lateral / sigma)²)` — beam width controlled by sigma=4.0
4. **Depth attenuation:** `exp(-mu × depth_norm × grid_size)` — Beer-Lambert Law
5. **Scale:** Multiply by `dose_weight × BEAM_SCALE`
6. **Body mask:** Zero dose outside patient body

**Physics trade-off:** Fast (~ms/compute) vs accurate (no scatter, no tissue-dependent attenuation, 2D only). Suitable for RL training, not clinical use.

#### `get_dvh_summary(dose, patient) → Dict`

Returns: `tumor_d95`, `tumor_dmean`, `tumor_dmax`, `tumor_coverage`, and per-OAR `oar_<name>_mean`, `oar_<name>_max`, `oar_<name>_limit`, `oar_<name>_violation`.

Always emits all four tumor keys even when tumor mask is empty (returns 0.0).

### DVH Calculator (`radiotherapy_env/physics/dvh.py`)

**Class:** `DVHCalculator(n_bins=50, max_dose_factor=2.0)`

**`compute(dose, mask, reference_dose) → np.ndarray (50,)`**

Cumulative DVH: for each of 50 bins from 0 to 2.0, computes fraction of voxels ≥ threshold.

- Reference dose = prescription (tumor) or limit (OAR) for normalization
- Returns float32 array shape (50,)

### Patient & Beam Models (`radiotherapy_env/physics/phantom.py`)

**Module constant:** `_GRID_SIZE = 64`

**`Beam` dataclass:** `angle: float [0, 180)`, `dose_weight: float [0.1, 1.0]`, `to_dict()`

**`OAR` dataclass:** `name: str`, `mask: np.ndarray`, `limit: float`, `priority: int` (1=critical, 2=important, 3=moderate)

**`PatientPhantom` dataclass:** `case_id`, `grid_size`, `tumor_mask`, `oars: List[OAR]`, `prescription_dose`, `body_mask`, `tumor_center: Tuple[float, float]`, `tumor_radius: float`

**Mask utilities:** `_make_circular_mask()`, `_make_elliptical_mask()` (supports rotation), `_make_rect_mask()`

### Patient Generators

| Generator | Task | Tumor Shape | OARs | Dose Limits |
|-----------|------|-------------|------|-------------|
| `ProstatePatientGenerator` | Easy | Circle r∈[5,8] | Rectum (ellipse, limit=0.40, P1), Bladder (circle, limit=0.50, P2) | Generous |
| `HeadNeckPatientGenerator` | Medium | Rotatable ellipse | 7 OARs: Spinal cord (P1, 0.45), Brainstem (P1, 0.45), 2×Parotids (P2, 0.26), Mandible (P3, 0.60), Larynx (P2, 0.40), Esophagus (P2, 0.34) | Tight |
| `PediatricBrainPatientGenerator` | Hard | Circle 2-3 voxels from brainstem | 5 OARs: Brainstem (P1, 0.30), Optic chiasm (P1, 0.25), 2×Cochlea (P2, 0.20), Whole brain (P3, 0.60) | Very tight |

**Key difficulty mechanic (Pediatric Brain):** Tumor is positioned 2-3 voxels from brainstem using a random angle offset. The near-zero margin makes conformal beam placement extremely challenging.

**`case_id` format:** `"prostate_NNNN"`, `"head_neck_NNNN"`, `"pediatric_brain_NNNN"`

---

## 6. Task System

### Abstract Base (`radiotherapy_env/tasks/base_task.py`)

```python
class BaseTask(ABC):
    @abstractmethod
    def sample_patient(self, rng: np.random.Generator) -> PatientPhantom: ...
    
    @abstractmethod
    def reward(self, dose, patient, beams) -> float: ...
```

### Task Implementations

| Task | Class | Patient Generator | Reward Modification |
|------|-------|-------------------|---------------------|
| Prostate | `ProstateTask` | `ProstatePatientGenerator` | None — direct `compute_reward()` |
| Head & Neck | `HeadNeckTask` | `HeadNeckPatientGenerator` | `base_reward × 0.95` (5% difficulty scaling) |
| Pediatric Brain | `PediatricBrainTask` | `PediatricBrainPatientGenerator` | Graduated brainstem penalty |

### Pediatric Brain Brainstem Penalty

Finds "Brainstem" OAR and applies multipliers based on mean dose violation:

| Condition | Multiplier | Severity |
|-----------|-----------|----------|
| `bs_mean > 1.5 × limit` | 0.30 | Severe |
| `bs_mean > 1.2 × limit` | 0.55 | Moderate |
| `bs_mean > 1.0 × limit` | 0.75 | Mild |
| compliant | 1.0 | No penalty |

### Task Registry (`radiotherapy_env/tasks/__init__.py`)

```python
TASK_REGISTRY = {
    "prostate":        ProstateTask,
    "head_neck":       HeadNeckTask,
    "pediatric_brain": PediatricBrainTask,
}
```

---

## 7. Reward & Scoring System

### Training Reward (`compute_reward()`)

Dense per-step reward, range [0.0, 1.0]:

```
reward = tumor_coverage × 0.55 − oar_penalty × 0.40 + plan_efficiency × 0.05
```

**Tumor coverage (0.55):**
```
0.5 × min(1.0, D95 / prescription_dose) + 0.5 × coverage_at_95%
```
- D95 = `np.percentile(tumor_dose, 5)` — dose to coldest 5% of tumor
- coverage_at_95% = fraction of tumor voxels with dose ≥ 0.95 × prescription

**OAR penalty (0.40):**
- Critical organs (P1): steep ramp, full penalty at 10% above limit
  - `mean_violation = min(1.0, max(0, mean_dose - limit) / (0.1 × limit))`
  - `max_violation = min(1.0, max(0, max_dose - 1.05×limit) / (0.1 × limit))`
  - Combined: `0.5 × mean + 0.5 × max`
- Non-critical (P2, P3): linear, full penalty at 50% above limit
  - `violation = min(1.0, max(0, mean_dose - limit) / (0.5 × limit))`
- Weighted by priority: `PRIORITY_WEIGHTS = {1: 1.5, 2: 1.0, 3: 0.5}`
- Normalized by total weight, capped at 1.0

**Plan efficiency (0.05):**
```
max(0.0, 1.0 - abs(n_beams - 6) / 7.0)
```
Peak at 6 beams.

### Final Score (`compute_score()`)

Stricter clinical evaluation, range [0.0, 1.0]:

```
score = tumor_score × 0.55 + oar_score × 0.40 + efficiency_score × 0.05
```

**Key difference from training reward:**
- Critical OARs use **binary pass/fail** (not gradient):
  - `mean_ok = mean_dose <= limit` → True/False
  - `max_ok = max_dose <= 1.05×limit` → True/False
  - Score = `0.5 × float(mean_ok) + 0.5 × float(max_ok)` → either 0.0, 0.5, or 1.0
- Non-critical OARs: same linear gradient as training
- OAR scores averaged (not penalty-subtracted)

**Why two functions?** Training reward gives gradient signal for learning ("getting warmer"). Score gives strict pass/fail for evaluation ("did you pass?"). An agent that scores well on compute_reward() should also score well on compute_score() because they use identical metrics and weights.

### Auto-Grader (`radiotherapy_env/reward/grader.py`)

```python
grade_task(env_id, agent_fn, n_episodes=20, seed=42) → Dict
grade_all(agent_fn, n_episodes=20, seed=42) → Dict
```

- Each episode uses `seed + ep` for diversity
- Pass rate threshold: score ≥ 0.60
- Returns: mean, std, min, max, pass_rate, scores list
- `grade_all()` averages across three tasks

---

## 8. Rendering (`radiotherapy_env/rendering/dose_heatmap.py`)

**`render_heatmap(dose, patient, beams, reward, step, size=512) → np.ndarray (H,W,3)`**

Two-panel matplotlib figure (dark theme #0d1117):

**Left panel — Dose heatmap:**
- Custom 7-color colormap: dark blue → cyan → yellow → red → white
- Contours: body (white), tumor (green #00ff88), OARs by priority (red/orange/blue)
- Beam arrows converging at tumor isocenter; opacity ∝ dose_weight

**Right panel — DVH curves:**
- Tumor: green; each OAR: colored by priority
- Prescription reference line + OAR dose limit lines

**Fallback:** `_simple_render()` — minimal RGB without matplotlib (blue=dose, green=tumor, red=OARs)

Imports are inside `try/except ImportError`; `_MATPLOTLIB_AVAILABLE` flag controls runtime fallback.

---

## 9. Inference Script (`inference.py`)

**This is the MAIN file the judges run.**

### Configuration (env vars)

```python
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TEMPERATURE = 0.3
MAX_TOKENS = 100
N_EPISODES = 3  # per task
SUCCESS_SCORE_THRESHOLD = 0.1
```

### System Prompt

Instructs LLM to:
- Add 5-6 beams first (action 0)
- Adjust angles to avoid organs (actions 1, 2)
- Increase/decrease dose based on constraints (actions 3, 4)
- Fine-tune for final optimization (action 6)
- Lock plan when satisfied (action 7)
- Reply with ONLY a single digit (0-7)

### How It Works

1. **format_observation()** converts numeric obs to readable text:
   - Beam angles and weights
   - Constraint violations (0=perfect, higher=worse)
   - DVH summary (D95, coverage)
   - Current score
   - Recent action history (last 5)

2. **get_llm_action()** sends observation to LLM, parses first digit 0-7 from response

3. **Fallback strategy** when LLM fails:
   - Step ≤ 6: action 0 (add beams)
   - Step < max_steps - 2: action 6 (fine-tune)
   - Otherwise: action 7 (lock plan)

4. **run_episode()** runs one full episode with [START]/[STEP]/[END] logging

5. **main()** runs all 3 tasks × 3 episodes each, prints aggregate score

### Tested Results (Llama 3.3 70B via Groq)

| Task | Mean Score | Episodes |
|------|-----------|----------|
| Prostate | 0.540 | 3 |
| Head & Neck | 0.560 | 3 |
| Pediatric Brain | 0.523 | 3 |
| **Aggregate** | **0.541** | 9 |

---

## 10. OpenEnv HTTP Server (`server/`)

Required by `openenv validate` for multi-mode deployment.

### Endpoints (`server/app.py`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Returns `{"status": "healthy"}` |
| `/metadata` | GET | Returns name, description, version, tasks list |
| `/schema` | GET | Returns Pydantic JSON schemas for action/observation/state |
| `/reset` | POST | Accepts `{"task": "prostate"}`, returns observation + reward + done |
| `/step` | POST | Accepts `{"action": 0}`, returns observation + reward + done |
| `/state` | GET | Returns full internal env state |

### Pydantic Models (`server/models.py`)

**RadiotherapyAction:** `action: int` (0-7, validated with ge=0, le=7)

**RadiotherapyObservation:** dvh_tumor, dvh_oar, beams, constraints, step_frac, score, n_beams, task

### Environment Wrapper (`server/radiotherapy_environment.py`)

Bridges Gymnasium to HTTP: converts numpy arrays to JSON-serializable lists, maps task names to env IDs, manages env lifecycle (close previous on reset).

### Entry Point

```python
# In pyproject.toml:
[project.scripts]
server = "server.app:main"

# Run:
uv run server
# or:
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## 11. PPO Baseline Training (`baseline/train_ppo.py`)

### Hyperparameters

```python
PPO("MultiInputPolicy", vec_env,
    learning_rate=3e-4,
    n_steps=512,           # steps before update
    batch_size=64,
    n_epochs=10,           # SGD passes per batch
    gamma=0.99,            # discount factor
    gae_lambda=0.95,       # advantage estimation
    clip_range=0.2,        # PPO clip
    ent_coef=0.01,         # exploration bonus
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
)
```

### Training Configuration

- 4 parallel vectorized envs for faster data collection
- EvalCallback: every ~10K steps, evaluate on 10 episodes, save best model
- CheckpointCallback: every ~20K steps, save intermediate model

### Training Budget

| Task | Timesteps | CPU Time | Patients Seen |
|------|-----------|----------|---------------|
| Prostate | 200K | ~15 min | ~5,000 |
| Head & Neck | 350K | ~25 min | ~7,000 |
| Pediatric Brain | 1M | ~50 min | ~15,000 |

### PPO Benchmark Results

| Task | Mean Score | Std | Pass Rate |
|------|-----------|-----|-----------|
| Prostate | 0.697 | ±0.054 | 100% |
| Head & Neck | 0.750 | ±0.059 | 96.7% |
| Pediatric Brain | 0.717 | ±0.090 | 95.0% |
| **Aggregate** | **0.721** | — | — |

---

## 12. Baseline Evaluation (`baseline/evaluate.py`)

### Agents

**`random_agent(obs, env)`** — random action (baseline floor, ~0.15 score)

**`smart_heuristic_agent(obs, env)`** — rule-based:
- Steps 0-30: Add beams if < 6 active
- Steps 30-40: Rotate/increase-dose/fine-tune based on constraint violations
- Steps 40+: Lock plan

**PPO agent** — loads `baseline/models/{task}_best/best_model.zip`, deterministic prediction

---

## 13. Gradio Demo (`app/app.py`)

**Framework:** Gradio, port 7860, targeting HuggingFace Spaces

### Features
- Task selector (3 difficulty levels)
- Seed slider (0–999) for reproducibility
- "Watch PPO Agent Plan" — runs trained agent (or heuristic fallback)
- Manual play with 8 action buttons
- Side-by-side human vs. agent score comparison
- Dual-axis reward/score chart (teal + purple, dark theme)
- Episode log with color-coded actions

### Model Loading (`load_model`)

Iterates `MODEL_CANDIDATES[task_key]` (ordered list of paths), tries each with/without `.zip`, returns first success or `None` (heuristic fallback).

### Heuristic Fallback (`heuristic_action`)

Fully deterministic (no `np.random`); uses `step % 2` for rotation direction.

### Session State

```python
session = {
    "env": env, "obs": dict, "done": bool, "step": int,
    "total_reward": float, "history": list,
    "reward_history": list, "score_history": list,
    "human_score": float | None, "agent_score": float | None,
}
```

---

## 14. Tests (`tests/test_env.py`)

**Run:** `pytest tests/ -v` — 25 tests, all must pass

| Class | Tests | Coverage |
|-------|-------|----------|
| `TestGymnasiumCompliance` | 8+ | Env registration, obs validity, seed reproducibility, state(), truncation, termination |
| `TestPhysics` | 5 | Zero-beam dose, positive dose, grid shape, body mask enforcement, multi-beam coverage |
| `TestReward` | 4 | Zero-beam reward, reward growth, score range, per-step computation |
| `TestTaskDifficulty` | 4+ | OAR counts (2/7/5), brainstem proximity (<15 voxels), all tasks runnable |

---

## 15. Deployment

### Docker (`Dockerfile`)

- Base: `python:3.10-slim`
- System deps: `libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender-dev`
- Env vars: `PYTHONUNBUFFERED=1`, `MPLBACKEND=Agg`
- Port: 7860
- Healthcheck: creates env, resets, closes
- Default CMD: `python app/app.py`

### HuggingFace Space

- **URL:** https://huggingface.co/spaces/VaishnaviAgrawal/RadiotherapyPlanningEnv
- **SDK:** Docker (Blank template)
- **Hardware:** CPU Basic (free)
- **README frontmatter:** YAML with `sdk: docker`
- **Git remote:** `hf` → `https://huggingface.co/spaces/VaishnaviAgrawal/RadiotherapyPlanningEnv`
- **Auth:** Token embedded in remote URL
- **LFS:** `.zip` files tracked with Git LFS (required by HF)

### GitHub

- **URL:** https://github.com/VaishnaviAgrawal03/radio_therapy_planning_for_tumor_treatment
- **Git remote:** `origin` → `git@github2:VaishnaviAgrawal03/...` (uses second SSH key)
- **SSH config:** `~/.ssh/config` maps `github2` host to `id_ed25519_github2` key

### OpenEnv Validation

`openenv validate` checks:
1. `pyproject.toml` exists
2. `uv.lock` exists
3. `[project.scripts]` has `server` entry point
4. `openenv-core>=0.2.0` in dependencies
5. `server/app.py` exists with `def main()` and `if __name__ == "__main__": main()`

---

## 16. Data Flow

```
Task Selection (string key)
        │
        ▼
TASK_REGISTRY[task] → TaskClass()
        │
        ▼
task.sample_patient(rng) → PatientPhantom
        │                   (tumor_mask, OARs, prescription_dose, body_mask, tumor_center)
        ▼
DoseCalculator.compute(patient, beams)
        │ Gaussian pencil-beam superposition at isocenter
        ▼
dose: np.ndarray (64×64, float32)
        │
        ├──► DVHCalculator.compute(dose, mask, ref) → dvh: np.ndarray (50,)
        │    [per structure: tumor + each OAR]
        │
        ├──► compute_reward(dose, patient, beams) → float  [training, gradient signal]
        │
        ├──► compute_score(dose, patient, beams) → float   [grading, binary for critical OARs]
        │
        ├──► _get_obs() → Dict observation [dvh_tumor, dvh_oar, beams, constraints, step_frac]
        │
        └──► render_heatmap(dose, patient, beams) → uint8 RGB [two-panel: dose + DVH]
```

---

## 17. Dependencies

| Category | Packages | Purpose |
|----------|----------|---------|
| Core | gymnasium, numpy | RL framework, numerics |
| Visualization | matplotlib, Pillow, scikit-image | Dose heatmap, contours |
| OpenEnv | openenv-core, fastapi, uvicorn, pydantic | Server endpoints, validation |
| Config | pyyaml | YAML parsing |
| Training (optional) | stable-baselines3, torch | PPO agent |
| Inference (optional) | openai | LLM API client |
| Demo (optional) | gradio | Web UI |
| Dev (optional) | pytest, pytest-cov | Testing |

---

## 18. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 64×64 grid | Balances fidelity with RL training speed |
| Max 7 beams | Clinically realistic IMRT constraint |
| Gaussian pencil-beam model | Fast (~ms) vs Monte Carlo (minutes). Preserves core trade-offs |
| Discrete(8) action space | Clinically meaningful, small enough for exploration |
| Dense per-step rewards | Enables gradient-based RL learning (not sparse end-of-episode) |
| Normalized observations [0,1] | Improves neural network training stability |
| Three task tiers | Graduated difficulty for curriculum learning research |
| Separate reward vs score | Training needs smooth gradients; evaluation needs strict clinical criteria |
| Priority-weighted OAR penalties | Reflects clinical reality (spinal cord > salivary glands) |
| Beams converge at tumor center | Realistic isocenter targeting, not grid center |
| DVH as observation | Compact, rotation-invariant, clinically standard representation |
| Beam angles [0, 180) not [0, 360) | Avoids redundancy (opposite angles equivalent) |
| Gradio + HF Spaces | Accessible demo without local setup |
| OpenEnv spec compliance | Standardized interface for benchmark comparability |
| FastAPI server endpoints | Required by openenv validate for multi-mode deployment |

---

## 19. Refactoring Changelog (v1.0.0)

### Bugs Fixed
- B1: `app.py` `load_model` now loads correct candidate (was always loading `path`)
- B2: `reward_fn.py` `compute_score` early-exit guard aligned with `compute_reward`
- B3: `grader.py` removed dead `episode_score = 0.0` assignment
- B4: `dose_calculator.py` `get_dvh_summary` always emits all four tumor keys

### Dead Code Removed
- D1-D2: Removed unused `import numpy` from prostate.py and head_neck.py
- D3: Removed stale comment from dose_calculator.py
- D4: Cleaned module docstring and dev comments from app.py
- D5: Moved `import radiotherapy_env` to module level in grader.py

### Code Quality
- Q1: `from .physics.phantom import Beam` moved to module-level in env.py
- Q2: `assert task in TASK_REGISTRY` → `if … raise ValueError`
- Q3: Added `_LOCK_PLAN_ACTION = 7` class constant
- Q4: Matplotlib imports moved to module-level try/except
- Q5: `BEAM_SCALE = 0.40` promoted to class constant
- Q6: Three duplicate `GRID = 64` → single `_GRID_SIZE = 64`
- Q7: Removed `sys.path.insert` hacks from tests and baseline scripts
- Q8: `heuristic_action` made deterministic (no `np.random.choice`)

### Naming
- N1: `case_id` prefixes aligned to task registry keys
- N2: Variable shadowing in `compute_score` fixed: `individual_score` + `mean_oar_score`
- N3: `self.calculator` → `self.dose_calculator`; `self.dvh_calc` → `self.dvh_calculator`
- N4: `MODEL_PATHS` + `MODEL_PATHS_FALLBACK` → single `MODEL_CANDIDATES`

### Type Annotations
- T1: `patient: "PatientPhantom"` and `beams: "List[Beam]"` via TYPE_CHECKING guard
- T2: `isocenter: Optional[Tuple[float, float]]`; direct `patient.tumor_center`
- T3: `tumor_center: Tuple[float, float]` (was plain `tuple`)

### Minor Fixes
- M1: Pediatric brain timesteps corrected to 1M (was 500K)
- M2: `_blank_image_small()` merged into `_blank_image(height, width)`

---

## 20. Configuration Quick Reference

| Setting | Value |
|---------|-------|
| Grid resolution | 64×64 |
| Max beams | 7 |
| Beam width sigma | 4.0 |
| Attenuation mu | 0.012 |
| Beam scale | 0.40 |
| DVH bins | 50 |
| DVH max dose factor | 2.0 |
| Reward weights | tumor=0.55, OAR=0.40, efficiency=0.05 |
| Priority weights | P1=1.5, P2=1.0, P3=0.5 |
| Pass threshold | score ≥ 0.60 |
| Optimal beam count | 6 |
| PPO learning rate | 3e-4 |
| PPO gamma | 0.99 |
| PPO clip range | 0.2 |
| PPO network | 256×256 (policy + value) |
| LLM temperature | 0.3 |
| LLM max tokens | 100 |
| Docker port | 7860 (Gradio), 8000 (FastAPI) |

---

## 21. Environment IDs Quick Reference

```python
# Easy (2 OARs, 50 steps)
env = gym.make("RadiotherapyEnv-prostate-v1")

# Medium (7 OARs, 60 steps)
env = gym.make("RadiotherapyEnv-headneck-v1")

# Hard (brainstem-adjacent, 70 steps)
env = gym.make("RadiotherapyEnv-pediatricbrain-v1")
```
