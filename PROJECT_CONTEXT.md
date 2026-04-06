# RadiotherapyPlanningEnv — Project Context

> Auto-generated project context file. Covers architecture, modules, data flow, and design decisions.

---

## 1. Project Overview

**RadiotherapyPlanningEnv** is a production-ready OpenEnv-compatible reinforcement learning environment that simulates radiotherapy treatment planning. An RL agent learns to place and optimize radiation beams to maximize tumor dose coverage while protecting surrounding organs-at-risk (OARs).

**Clinical Motivation**: ~14 million cancer patients/year require radiotherapy. Manual treatment planning takes 2–4 hours per patient. This environment enables RL-based automation of beam placement and dose optimization.

**Version**: 1.0.0  
**License**: MIT  
**Python requirement**: >=3.9  
**OpenEnv spec**: https://github.com/openenv/openenv

---

## 2. Repository Structure

```
radiotherapy-env/
├── radiotherapy_env/          # Main Python package (RL environment)
│   ├── __init__.py            # Gymnasium env registration
│   ├── env.py                 # Core RadiotherapyEnv class
│   ├── physics/               # Dose calculation and patient models
│   │   ├── dose_calculator.py # Pencil-beam dose model
│   │   ├── dvh.py             # Dose-Volume Histogram computation
│   │   └── phantom.py         # Patient anatomy + Beam dataclasses
│   ├── tasks/                 # Task definitions per cancer type
│   │   ├── base_task.py       # Abstract base class
│   │   ├── prostate.py        # Easy (2 OARs)
│   │   ├── head_neck.py       # Medium (7 OARs)
│   │   ├── pediatric_brain.py # Hard (brainstem-adjacent)
│   │   └── __init__.py        # TASK_REGISTRY dict
│   ├── reward/                # Reward and scoring
│   │   ├── reward_fn.py       # compute_reward() + compute_score()
│   │   └── grader.py          # Auto-grading for leaderboard
│   └── rendering/
│       └── dose_heatmap.py    # RGB visualization (dose + DVH)
├── baseline/                  # Training + evaluation scripts
│   ├── train_ppo.py           # PPO training with stable-baselines3
│   ├── evaluate.py            # Agent evaluation + heuristic baseline
│   ├── results.json           # Aggregate benchmark results
│   ├── results_prostate.json
│   ├── results_head_neck.json
│   ├── results_pediatric_brain.json
│   ├── logs/                  # TensorBoard event files
│   └── models/                # Saved PPO model checkpoints
│       ├── prostate_best/
│       ├── head_neck_best/
│       └── pediatric_brain_best/
├── app/
│   └── app.py                 # Gradio interactive demo (HuggingFace Spaces)
├── tests/
│   └── test_env.py            # Comprehensive pytest suite
├── app.py                     # Entry-point wrapper → app/app.py
├── openenv.yaml               # OpenEnv spec metadata
├── setup.py                   # Package build config (pip install)
├── requirements.txt           # Pinned dependencies
├── Dockerfile                 # Container build (port 7860)
├── DEPLOY.sh                  # 11-step deployment guide
└── README.md                  # Project documentation
```

---

## 3. Gymnasium Environment

### Registration (`radiotherapy_env/__init__.py`)

Three environments registered via `gym.register()`:

| Gym ID | Task | Max Steps | Difficulty |
|--------|------|-----------|------------|
| `RadiotherapyEnv-prostate-v1` | prostate | 50 | Easy |
| `RadiotherapyEnv-headneck-v1` | head_neck | 60 | Medium |
| `RadiotherapyEnv-pediatricbrain-v1` | pediatric_brain | 70 | Hard |

**Usage**:
```python
import gymnasium as gym
import radiotherapy_env  # triggers registration

env = gym.make("RadiotherapyEnv-prostate-v1")
obs, info = env.reset()
```

### Core Environment (`radiotherapy_env/env.py`)

**Class**: `RadiotherapyEnv(gym.Env)`

**Constructor**: `__init__(task="prostate", max_steps=50, render_mode=None)`

**Class Constants**:
- `MAX_BEAMS = 7`
- `GRID_SIZE = 64`
- `_LOCK_PLAN_ACTION = 7` — action index that terminates the episode

**Action Space**: `Discrete(8)`

| Action | Description |
|--------|-------------|
| 0 | Add beam at next default angle |
| 1 | Rotate last beam +10° |
| 2 | Rotate last beam -10° |
| 3 | Increase dose weight |
| 4 | Decrease dose weight |
| 5 | Remove last beam |
| 6 | Fine-tune all beams |
| 7 | Lock plan (terminate episode) — matched via `_LOCK_PLAN_ACTION` |

**Observation Space**: `Dict`

| Key | Shape | Description |
|-----|-------|-------------|
| `dvh_tumor` | Box(50,) | Cumulative DVH for tumor |
| `dvh_oar` | Box(3, 50) | DVH for top 3 OARs |
| `beams` | Box(7, 3) | [normalized_angle, dose_weight, is_active] per beam |
| `constraints` | Box(4,) | Normalized constraint violations [tumor, oar1, oar2, oar3] |
| `step_frac` | Box(1,) | Episode progress fraction |

**Instance Fields**:
- `self.dose_calculator` — `DoseCalculator` instance
- `self.dvh_calculator` — `DVHCalculator` instance

**Key Methods**:

- `reset(seed)` — Samples new patient from task generator, initializes beams
- `step(action)` — Applies action, recomputes dose, returns (obs, reward, terminated, truncated, info)
- `state()` — Returns full state dict (OpenEnv spec compliance)
- `render()` — Returns RGB numpy array via `render_heatmap()`
- `get_score()` — Returns final plan quality score [0.0, 1.0]
- `get_dvh_summary()` — Returns DVH metrics dict

**Episode Flow**:
1. `reset()` → sample patient → initialize beams → compute dose → return obs
2. Per step: `_apply_action()` → `DoseCalculator.compute()` → `DVHCalculator.compute()` → reward → obs
3. Termination: `action == _LOCK_PLAN_ACTION` (lock plan) OR `step >= max_steps`

**Design Notes**:
- 64×64 grid resolution (speed/fidelity balance)
- Max 7 beams (clinically realistic)
- All observations normalized to [0, 1]
- Dense per-step rewards
- Beams initialized at evenly-spaced angles with small noise
- Task validation raises `ValueError` (not `assert`) so it works under Python `-O` flag

---

## 4. Physics Layer

### Patient & Beam Models (`radiotherapy_env/physics/phantom.py`)

**Module constant**: `_GRID_SIZE = 64` — shared by all patient generators (no per-class duplication)

**`Beam` (dataclass)**:
- `angle: float` — beam direction [0, 180°)
- `dose_weight: float` — relative beam strength [0.1, 1.0]
- `to_dict()` — serialization

**`OAR` (dataclass)**:
- `name: str`, `mask: np.ndarray`, `limit: float`, `priority: int`
- Priority: 1=critical, 2=important, 3=moderate

**`PatientPhantom` (dataclass)**:
- `case_id`, `grid_size`, `tumor_mask`, `oars`, `prescription_dose`, `body_mask`
- `tumor_center: Tuple[float, float]` — isocenter (x, y) for beam convergence
- `tumor_radius: float`

**`case_id` format** (aligned with task registry keys):
- Prostate: `"prostate_<NNNN>"`
- Head & neck: `"head_neck_<NNNN>"`
- Pediatric brain: `"pediatric_brain_<NNNN>"`

**Patient Generators** (one per task, all accept `rng: np.random.Generator`):

| Generator | Task | Tumor Shape | OARs | Constraints |
|-----------|------|-------------|------|-------------|
| `ProstatePatientGenerator` | Easy | Spherical | Rectum (ellipse), Bladder (circle) | 0.40, 0.50 Gy |
| `HeadNeckPatientGenerator` | Medium | Rotatable ellipse | Spinal cord, Brainstem, 2×Parotids, Mandible, Larynx, Esophagus | 0.25–0.60 Gy |
| `PediatricBrainPatientGenerator` | Hard | Circle 2-3mm from brainstem | Brainstem, Optic chiasm, 2×Cochlea, Whole brain | 0.20–0.60 Gy |

**Mask generation utilities**: `_make_circular_mask()`, `_make_elliptical_mask()`, `_make_rect_mask()`

### Dose Calculator (`radiotherapy_env/physics/dose_calculator.py`)

**Class**: `DoseCalculator(grid_size=64, beam_width_sigma=4.0, attenuation_mu=0.012, prescription_dose=1.0)`

**Class Constant**: `BEAM_SCALE = 0.40` — calibrated so 7 converging beams ≈ 1.0 Gy at isocenter

**`compute(patient, beams) → np.ndarray`** — Superposition of single-beam contributions

**`_compute_single_beam(beam, body_mask, isocenter: Optional[Tuple[float, float]])`** — Gaussian pencil-beam model:
1. Coordinate transform to isocenter-centered system (uses `patient.tumor_center` directly)
2. Lateral distance from beam axis → Gaussian profile
3. Depth along beam axis → exponential attenuation
4. Multiply by `dose_weight × BEAM_SCALE`
5. Apply `body_mask` (zero outside patient)

**`get_dvh_summary(dose, patient) → dict`** — DVH metrics (D95, Dmean, Dmax, coverage, OAR violations).
Always returns all four tumor keys (`tumor_d95`, `tumor_dmean`, `tumor_dmax`, `tumor_coverage`) even when the tumor mask is empty.

**Design Notes**:
- Pre-computed meshgrids `_cx`, `_cy` cached in `__init__`
- No scatter modeling (speed trade-off acceptable for RL)
- `BEAM_SCALE` is a class constant — tune it there, not inline
- Dose normalized to prescription dose (scale invariance across tasks)

### DVH Calculator (`radiotherapy_env/physics/dvh.py`)

**Class**: `DVHCalculator(n_bins=50, max_dose_factor=2.0)`

**`compute(dose, mask, reference_dose) → np.ndarray`** — Cumulative DVH:
- Extracts voxel doses inside mask
- Normalizes by reference dose
- For each of 50 bins: computes fraction of voxels ≥ threshold
- Returns float32 array shape (50,)

**Design Notes**:
- Cumulative (not differential) DVH — better for constraint checking
- Fixed 50 bins as compact RL observation
- Reference dose handles both prescription (tumor) and limit (OAR) contexts

---

## 5. Task System

### Abstract Base (`radiotherapy_env/tasks/base_task.py`)

```python
class BaseTask(ABC):
    @abstractmethod
    def sample_patient(self, rng: np.random.Generator) -> PatientPhantom: ...
    
    @abstractmethod
    def reward(self, dose, patient, beams) -> float: ...
```

### Task Implementations

| File | Class | Reward Modification |
|------|-------|---------------------|
| `prostate.py` | `ProstateTask` | None — direct `compute_reward()` passthrough |
| `head_neck.py` | `HeadNeckTask` | `base_reward × 0.95` (difficulty scaling) |
| `pediatric_brain.py` | `PediatricBrainTask` | Graduated brainstem penalty (0.30×, 0.55×, 0.75×, 1.0×) |

**Pediatric brain penalty tiers**:
- `brainstem_mean > 1.5 × limit` → multiplier 0.30 (severe)
- `brainstem_mean > 1.2 × limit` → multiplier 0.55 (moderate)
- `brainstem_mean > 1.0 × limit` → multiplier 0.75 (mild)
- compliant → multiplier 1.0 (no penalty)

### Task Registry (`radiotherapy_env/tasks/__init__.py`)

```python
TASK_REGISTRY = {
    "prostate":        ProstateTask,
    "head_neck":       HeadNeckTask,
    "pediatric_brain": PediatricBrainTask,
}
```

---

## 6. Reward & Scoring System

### Training Reward (`radiotherapy_env/reward/reward_fn.py` — `compute_reward()`)

Dense per-step reward, range [0.0, 1.0]:

```
reward = tumor_coverage × 0.50 − oar_penalty × 0.40 + plan_efficiency × 0.10
```

**Tumor coverage** (weight 0.50):
- `0.8 × fraction_tumor_at_95pct_Rx + 0.2 × min(1, mean_dose / Rx_dose)`

**OAR penalty** (weight 0.40):
- Per OAR: mean violation + max violation (critical organs only, priority=1)
- Weighted by priority: critical=1.5, important=1.0, moderate=0.5
- Normalized by total weight, capped at 2.0× limit

**Plan efficiency** (weight 0.10):
- Optimal beam count: 5–7 (peak at 6)
- `0.7 × beam_efficiency + 0.3 × weight_efficiency`

### Final Score (`compute_score()`)

Stricter clinical evaluation, range [0.0, 1.0]:

```
score = tumor_score × 0.55 + oar_score × 0.40 + efficiency_score × 0.05
```

- Tumor: D95 metric + coverage at 95% Rx
- OAR: Per-OAR score stored in `individual_score`; mean across OARs stored in `mean_oar_score` (no variable shadowing)
- Efficiency: same beam count formula

**Priority weights**: `PRIORITY_WEIGHTS = {1: 1.5, 2: 1.0, 3: 0.5}`

### Auto-Grader (`radiotherapy_env/reward/grader.py`)

```python
grade_task(env_id, agent_fn, n_episodes=20, seed=42) -> dict
grade_all(agent_fn, n_episodes=20, seed=42) -> dict
```

- `import radiotherapy_env` is at module level (triggers gym registration on import)
- Each episode uses seed+ep for diverse evaluation
- Pass rate threshold: score ≥ 0.60
- Returns: mean, std, min, max, pass_rate, scores list
- `grade_all()` aggregates three tasks arithmetically

---

## 7. Rendering (`radiotherapy_env/rendering/dose_heatmap.py`)

**`render_heatmap(dose: np.ndarray, patient: "PatientPhantom", beams: "List[Beam]", reward, step, size=512) → np.ndarray`**

Returns uint8 RGB array (H, W, 3). Two-panel matplotlib figure.

**Imports**: `matplotlib`, `PIL.Image`, and related modules are imported at module level inside a `try/except ImportError`. The `_MATPLOTLIB_AVAILABLE` flag controls runtime fallback. `PatientPhantom` and `Beam` are imported under `TYPE_CHECKING` only (no circular import risk).

**Left panel — Dose heatmap**:
- Custom colormap: dark blue → cyan → yellow → red → white
- Contours: body (white), tumor (green `#00ff88`), OARs by priority (red/orange/blue)
- Beam arrows converging at tumor isocenter; opacity ∝ dose_weight
- Dark theme background `#0d1117`

**Right panel — DVH curves**:
- Tumor: green; each OAR: colored by priority
- Prescription reference line + OAR dose limit lines
- Cumulative dose-volume on both axes

**Fallback**: `_simple_render()` — minimal RGB without matplotlib:
- Blue channel = dose, green channel = tumor, red channel = OARs

---

## 8. Baseline Training & Evaluation

### PPO Training (`baseline/train_ppo.py`)

**Algorithm**: PPO via stable-baselines3, `MultiInputPolicy`

**Hyperparameters**:
- learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10
- gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01
- Network: policy [256, 256], value [256, 256]
- 4 parallel vectorized envs during training

**Training budget** (used in `--all` mode):
| Task | Timesteps | CPU time |
|------|-----------|----------|
| Prostate | 200K | ~15 min |
| Head & Neck | 350K | ~25 min |
| Pediatric Brain | 1M | ~50 min |

**Saved artifacts**:
- `baseline/models/{task}_best/best_model.zip`
- `baseline/models/{task}_checkpoints/` (periodic)
- `baseline/logs/{task}/` (TensorBoard)

### Evaluation (`baseline/evaluate.py`)

**Agents**:
- `random_agent(obs, env)` — random action baseline
- `smart_heuristic_agent(obs, env)` — stage-based rule system:
  - Steps 0–30: Add beams if < 6 active
  - Steps 30–40: Rotate/increase-dose/fine-tune based on constraint violations
  - Steps 40+: Lock plan

### Benchmark Results (`baseline/results.json`)

| Task | Difficulty | Mean Score | Std | Pass Rate | Timesteps |
|------|------------|------------|-----|-----------|-----------|
| Prostate | Easy | 0.697 | ±0.054 | 100% | 200K |
| Head & Neck | Medium | 0.750 | ±0.059 | 96.7% | 350K |
| Pediatric Brain | Hard | 0.717 | ±0.090 | 95.0% | 1M |
| **Aggregate** | — | **0.721** | — | — | — |

---

## 9. Interactive Demo (`app/app.py`)

**Framework**: Gradio, targeting HuggingFace Spaces

**Features**:
- Task selector (3 difficulty levels)
- Seed slider (0–999) for reproducibility
- Agent auto-play (trained PPO or heuristic fallback)
- Human manual play (8 action buttons)
- Side-by-side human vs. agent score comparison
- Dual-axis reward/score chart (teal + purple, dark theme)
- Episode log with color-coded actions

**Model loading**: `load_model(task_key)` — iterates `MODEL_CANDIDATES[task_key]` (ordered list of paths per task), tries each with and without `.zip` extension, returns first successful load or `None` (heuristic fallback). Bug fixed: previously always loaded `path` instead of the matched `candidate`.

**Model path config**: `MODEL_CANDIDATES` — single dict mapping each task key to an ordered list of candidate paths (replaces the former `MODEL_PATHS` + `MODEL_PATHS_FALLBACK` pair).

**Heuristic fallback**: `heuristic_action(obs, step)` — fully deterministic (no `np.random`); uses `step % 2` alternation where rotation direction was previously random.

**Session state**: Dict tracking env, obs, step count, reward/score history, human and agent scores

---

## 10. Tests (`tests/test_env.py`)

Four test classes:

| Class | Tests | Coverage |
|-------|-------|----------|
| `TestGymnasiumCompliance` | 8 | Env registration, obs validity, seed reproducibility, state(), truncation, termination |
| `TestPhysics` | 5 | Zero-beam dose, positive dose, grid shape, body mask enforcement, multi-beam coverage |
| `TestReward` | 4 | Zero-beam reward, reward growth, score range, per-step computation |
| `TestTaskDifficulty` | 4 | OAR counts (2/7/5), brainstem proximity (<15 voxels), task runnability |

**Run** (requires `pip install -e .[dev]` first — no `sys.path` hacks needed):
```bash
pytest tests/ -v
```

---

## 11. Deployment

### Docker (`Dockerfile`)

- Base: `python:3.10-slim`
- Port: 7860 (Gradio default)
- Env: `MPLBACKEND=Agg` (headless matplotlib)
- Healthcheck: creates env, resets, closes
- Default CMD: `python app/app.py`

### DEPLOY.sh (11-step guide)

1. Python version check
2. venv setup
3. `pip install -e .[training,demo,dev]`
4. Inline sanity check (env create + random run)
5. `pytest tests/ -v`
6. Train all PPO agents
7. Evaluate + save results.json
8. Launch Gradio demo
9. Docker build + healthcheck
10. Push to HuggingFace Spaces
11. Push to GitHub Container Registry

### openenv.yaml

OpenEnv spec metadata file covering: name/version/description, task definitions, action/observation spaces, reward structure, baseline results, installation instructions, and quick start code.

---

## 12. Data Flow

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
        │ Gaussian pencil-beam superposition
        ▼
dose: np.ndarray (64×64, float32)
        │
        ├──► DVHCalculator.compute(dose, mask, ref) → dvh: np.ndarray (50,)
        │    [per structure: tumor + each OAR]
        │
        ├──► compute_reward(dose, patient, beams) → float  [training]
        │
        ├──► compute_score(dose, patient, beams) → float   [grading]
        │
        ├──► _get_obs() → Dict observation
        │
        └──► render_heatmap(dose, patient, beams) → uint8 RGB
```

---

## 13. Dependencies

**Core** (always required):
- `gymnasium>=0.29.0` — RL framework
- `numpy>=1.24.0` — numerical computing
- `matplotlib>=3.7.0` — visualization
- `Pillow>=9.0.0` — image processing
- `scikit-image>=0.20.0` — contour extraction
- `pyyaml>=6.0` — YAML parsing

**Training** (`pip install .[training]`):
- `stable-baselines3>=2.0.0`
- `torch>=2.0.0`

**Demo** (`pip install .[demo]`):
- `gradio>=4.0.0`

**Dev** (`pip install .[dev]`):
- `pytest>=7.0.0`
- `pytest-cov>=4.0.0`

---

## 14. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 64×64 grid | Balances spatial fidelity with RL training speed |
| Max 7 beams | Clinically realistic IMRT constraint |
| Gaussian pencil-beam model | Fast approximation (no Monte Carlo) suitable for thousands of RL episodes |
| Discrete(8) action space | Clinically meaningful actions, small enough for efficient exploration |
| Dense rewards | Per-step signal accelerates RL convergence |
| Normalized observations [0,1] | Improves neural network training stability |
| Three task tiers | Graduated difficulty mirrors real clinical complexity progression |
| Separate compute_reward / compute_score | Training uses dense partial reward; evaluation uses strict clinical criteria |
| Priority-weighted OAR penalties | Reflects clinical priority (spinal cord more critical than salivary glands) |
| Gradio + HuggingFace deployment | Accessible demo without local setup |
| OpenEnv spec compliance | Standardized interface for benchmark comparability |

---

## 15. Environment IDs Quick Reference

```python
# Easy (2 OARs, 50 steps)
env = gym.make("RadiotherapyEnv-prostate-v1")

# Medium (7 OARs, 60 steps)
env = gym.make("RadiotherapyEnv-headneck-v1")

# Hard (brainstem-adjacent, 70 steps)
env = gym.make("RadiotherapyEnv-pediatricbrain-v1")
```

---

## 16. Refactoring Changelog (v1.0.0 → clean state)

Applied after initial submission to bring the codebase to onboarding-ready quality. All 25 tests pass.

### Bugs Fixed
| ID | File | Description |
|----|------|-------------|
| B1 | `app/app.py` | `load_model` now calls `PPO.load(candidate)` — previously always loaded `path` regardless of which candidate existed |
| B2 | `reward/reward_fn.py` | `compute_score` early-exit guard aligned with `compute_reward` (removed spurious `or dose is None`) |
| B3 | `reward/grader.py` | Removed dead `episode_score = 0.0` initialization immediately overwritten on the next line |
| B4 | `physics/dose_calculator.py` | `get_dvh_summary` now always emits all four tumor keys (`tumor_d95`, `tumor_dmean`, `tumor_dmax`, `tumor_coverage`) even when the tumor mask is empty |

### Dead Code Removed
| ID | File | Description |
|----|------|-------------|
| D1 | `tasks/prostate.py` | Removed unused `import numpy as np` |
| D2 | `tasks/head_neck.py` | Removed unused `import numpy as np` |
| D3 | `physics/dose_calculator.py` | Removed stale `# Keep raw accumulated dose — raw pass` comment |
| D4 | `app/app.py` | Cleaned module docstring; removed inline `# FIX 1/2/3` development comments |
| D5 | `reward/grader.py` | Moved `import radiotherapy_env` from inside `grade_task()` to module level |

### Code Quality
| ID | File | Description |
|----|------|-------------|
| Q1 | `env.py` | `from .physics.phantom import Beam` moved to module-level imports |
| Q2 | `env.py` | `assert task in TASK_REGISTRY` replaced with `if … raise ValueError` |
| Q3 | `env.py` | Added `_LOCK_PLAN_ACTION = 7` class constant; `step()` uses it instead of bare `7` |
| Q4 | `rendering/dose_heatmap.py` | `matplotlib.use("Agg")` and related imports moved to module-level `try/except` block |
| Q5 | `physics/dose_calculator.py` | `BEAM_SCALE = 0.40` promoted to class constant |
| Q6 | `physics/phantom.py` | Three duplicate `GRID = 64` class attributes replaced with module-level `_GRID_SIZE = 64` |
| Q7 | `tests/test_env.py`, `baseline/train_ppo.py`, `baseline/evaluate.py` | Removed `sys.path.insert` hacks (package must be installed with `pip install -e .`) |
| Q8 | `app/app.py` | `heuristic_action` no longer uses `np.random.choice`; uses deterministic `step % 2` alternation |

### Naming
| ID | File | Description |
|----|------|-------------|
| N1 | `physics/phantom.py` | `case_id` prefixes aligned to task registry keys: `headneck_` → `head_neck_`, `pedibrain_` → `pediatric_brain_` |
| N2 | `reward/reward_fn.py` | Variable shadowing in `compute_score` fixed: loop variable → `individual_score`, aggregate → `mean_oar_score` |
| N3 | `env.py` | `self.calculator` → `self.dose_calculator`; `self.dvh_calc` → `self.dvh_calculator` |
| N4 | `app/app.py` | `MODEL_PATHS` + `MODEL_PATHS_FALLBACK` merged into `MODEL_CANDIDATES` (ordered list per task) |

### Type Annotations
| ID | File | Description |
|----|------|-------------|
| T1 | `rendering/dose_heatmap.py` | `patient: "PatientPhantom"` and `beams: "List[Beam]"` via `TYPE_CHECKING` guard |
| T2 | `physics/dose_calculator.py` | `isocenter: Optional[Tuple[float, float]]`; direct `patient.tumor_center` replaces `getattr` fallback |
| T3 | `physics/phantom.py` | `tumor_center: Tuple[float, float]` (was plain `tuple`) |

### Minor Fixes
| ID | File | Description |
|----|------|-------------|
| M1 | `baseline/train_ppo.py` | `pediatric_brain` timesteps in `--all` mode corrected from `500_000` to `1_000_000` (matches docs and results) |
| M2 | `app/app.py` | `_blank_image_small()` merged into `_blank_image(height=400, width=900)` with optional size parameters |
