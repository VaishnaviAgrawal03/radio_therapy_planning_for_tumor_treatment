"""
RadiotherapyPlanningEnv-v1
==========================
A Gymnasium-compatible RL environment for radiation therapy treatment planning.
An agent learns to place and optimize radiation beams to maximize tumor dose
while protecting surrounding organs-at-risk (OARs).

Observation Space:
    Dict:
        dvh_tumor  : Box(50,)   - Dose-Volume Histogram for tumor (normalized)
        dvh_oar    : Box(3,50)  - DVH for up to 3 OARs
        beams      : Box(7,3)   - Current beams [angle_norm, dose_norm, active]
        constraints: Box(4,)    - Normalized constraint violations per structure
        step_frac  : Box(1,)    - Fraction of max steps used

Action Space:
    Discrete(8):
        0 - Add beam at next default angle
        1 - Rotate last beam +10 degrees
        2 - Rotate last beam -10 degrees
        3 - Increase last beam dose by 10%
        4 - Decrease last beam dose by 10%
        5 - Remove last beam
        6 - Fine-tune all beams (small random perturbation)
        7 - Lock plan (terminate episode)

Reward:
    Per-step partial reward:
        tumor_coverage * 0.50
      - oar_penalty      * 0.40
      + plan_efficiency  * 0.10
    Range: [0.0, 1.0]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

from .physics.phantom import PatientPhantom, Beam
from .physics.dose_calculator import DoseCalculator
from .physics.dvh import DVHCalculator
from .reward.reward_fn import compute_reward, compute_score
from .tasks import TASK_REGISTRY
from .rendering.dose_heatmap import render_heatmap


class RadiotherapyEnv(gym.Env):
    """
    Radiotherapy Treatment Planning Environment.

    The agent acts as an automated treatment planner, iteratively placing
    and adjusting radiation beams to produce a clinically acceptable plan.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    MAX_BEAMS = 7
    GRID_SIZE = 64          # patient phantom grid resolution
    _LOCK_PLAN_ACTION = 7   # action index that terminates the episode

    def __init__(
        self,
        task: str = "prostate",
        max_steps: int = 50,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            task: One of 'prostate' (easy), 'head_neck' (medium), 'pediatric_brain' (hard)
            max_steps: Maximum steps per episode
            render_mode: 'human' or 'rgb_array'
        """
        super().__init__()

        if task not in TASK_REGISTRY:
            raise ValueError(f"Task '{task}' not found. Choose from {list(TASK_REGISTRY.keys())}")
        self.task_name = task
        self.task = TASK_REGISTRY[task]()
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.dose_calculator = DoseCalculator(grid_size=self.GRID_SIZE)
        self.dvh_calculator = DVHCalculator(n_bins=50)

        # ── Action Space ──────────────────────────────────────────────────────
        self.action_space = spaces.Discrete(8)

        # ── Observation Space ─────────────────────────────────────────────────
        self.observation_space = spaces.Dict({
            "dvh_tumor":   spaces.Box(0.0, 1.0, shape=(50,),   dtype=np.float32),
            "dvh_oar":     spaces.Box(0.0, 1.0, shape=(3, 50), dtype=np.float32),
            "beams":       spaces.Box(0.0, 1.0, shape=(self.MAX_BEAMS, 3), dtype=np.float32),
            "constraints": spaces.Box(0.0, 2.0, shape=(4,),    dtype=np.float32),
            "step_frac":   spaces.Box(0.0, 1.0, shape=(1,),    dtype=np.float32),
        })

        # Internal state
        self.patient: Optional[PatientPhantom] = None
        self.beams: list = []
        self.step_count: int = 0
        self.current_dose: Optional[np.ndarray] = None
        self._last_reward: float = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Core Gymnasium API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        """Reset environment with a new patient case."""
        super().reset(seed=seed)

        self.patient = self.task.sample_patient(self.np_random)
        self.beams = []
        self.step_count = 0
        self.current_dose = np.zeros(
            (self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32
        )
        self._last_reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Apply action, recompute dose, return (obs, reward, terminated, truncated, info)."""
        assert self.patient is not None, "Call reset() before step()"
        assert self.action_space.contains(action), f"Invalid action: {action}"

        terminated = False
        truncated = False

        # Apply the chosen action
        self._apply_action(int(action))

        # Recompute dose distribution
        self.current_dose = self.dose_calculator.compute(self.patient, self.beams)

        # Compute reward
        reward = compute_reward(self.current_dose, self.patient, self.beams)
        self._last_reward = reward

        self.step_count += 1

        # Check termination
        if action == self._LOCK_PLAN_ACTION:
            terminated = True
        elif self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def state(self) -> Dict[str, Any]:
        """
        Return full environment state (OpenEnv spec requirement).
        Useful for visualization and debugging.
        """
        assert self.patient is not None, "Call reset() first"
        return {
            "task": self.task_name,
            "patient": self.patient.to_dict(),
            "beams": [b.to_dict() for b in self.beams],
            "dose_grid": self.current_dose.tolist() if self.current_dose is not None else None,
            "step_count": self.step_count,
            "last_reward": self._last_reward,
            "score": self.get_score(),
        }

    def render(self) -> Optional[np.ndarray]:
        """Render current dose distribution as an image."""
        if self.render_mode is None:
            return None
        if self.patient is None:
            return None

        frame = render_heatmap(
            dose=self.current_dose,
            patient=self.patient,
            beams=self.beams,
            reward=self._last_reward,
            step=self.step_count,
        )

        if self.render_mode == "human":
            try:
                import matplotlib.pyplot as plt
                plt.imshow(frame)
                plt.axis("off")
                plt.tight_layout()
                plt.pause(0.001)
                plt.clf()
            except ImportError:
                pass
        return frame

    def close(self):
        pass

    # ──────────────────────────────────────────────────────────────────────────
    # Public helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_score(self) -> float:
        """
        Compute final plan quality score [0.0, 1.0].
        This is what the auto-grader uses.
        """
        if self.patient is None or self.current_dose is None:
            return 0.0
        return compute_score(self.current_dose, self.patient, self.beams)

    def get_dvh_summary(self) -> Dict[str, float]:
        """Return key DVH metrics for the current plan."""
        if self.patient is None or self.current_dose is None:
            return {}
        return self.dose_calculator.get_dvh_summary(self.current_dose, self.patient)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_action(self, action: int):
        """Mutate beam list based on discrete action."""
        if action == 0:  # Add beam
            if len(self.beams) < self.MAX_BEAMS:
                # Spread beams evenly — start with 180/max_beams increments
                base_angle = len(self.beams) * (180.0 / self.MAX_BEAMS)
                # Add small noise for diversity
                angle = base_angle + self.np_random.uniform(-5, 5)
                self.beams.append(Beam(angle=angle % 180, dose_weight=0.6))

        elif action == 1:  # Rotate last beam +10°
            if self.beams:
                self.beams[-1].angle = (self.beams[-1].angle + 10) % 180

        elif action == 2:  # Rotate last beam -10°
            if self.beams:
                self.beams[-1].angle = (self.beams[-1].angle - 10) % 180

        elif action == 3:  # Increase dose
            if self.beams:
                self.beams[-1].dose_weight = min(1.0, self.beams[-1].dose_weight + 0.1)

        elif action == 4:  # Decrease dose
            if self.beams:
                self.beams[-1].dose_weight = max(0.1, self.beams[-1].dose_weight - 0.1)

        elif action == 5:  # Remove last beam
            if self.beams:
                self.beams.pop()

        elif action == 6:  # Fine-tune — small perturbation to all beams
            for beam in self.beams:
                beam.angle = (beam.angle + self.np_random.uniform(-3, 3)) % 180
                beam.dose_weight = np.clip(
                    beam.dose_weight + self.np_random.uniform(-0.05, 0.05), 0.1, 1.0
                )

        elif action == 7:  # Lock plan — no-op (termination handled in step())
            pass

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Build observation dictionary."""
        # DVH for tumor
        dvh_tumor = np.zeros(50, dtype=np.float32)
        # DVH for OARs (pad to 3 OARs)
        dvh_oar = np.zeros((3, 50), dtype=np.float32)

        if self.patient is not None and self.current_dose is not None:
            dvh_tumor = self.dvh_calculator.compute(
                self.current_dose, self.patient.tumor_mask, self.patient.prescription_dose
            )
            for i, oar in enumerate(self.patient.oars[:3]):
                dvh_oar[i] = self.dvh_calculator.compute(
                    self.current_dose, oar.mask, oar.limit
                )

        # Beam configuration [angle_norm, dose_weight, active]
        beams_arr = np.zeros((self.MAX_BEAMS, 3), dtype=np.float32)
        for i, beam in enumerate(self.beams[:self.MAX_BEAMS]):
            beams_arr[i] = [beam.angle / 180.0, beam.dose_weight, 1.0]

        # Constraint violations (normalized)
        constraints = np.zeros(4, dtype=np.float32)
        if self.patient is not None and self.current_dose is not None:
            constraints = self._get_constraint_violations()

        step_frac = np.array([self.step_count / self.max_steps], dtype=np.float32)

        return {
            "dvh_tumor":   dvh_tumor,
            "dvh_oar":     dvh_oar,
            "beams":       beams_arr,
            "constraints": constraints,
            "step_frac":   step_frac,
        }

    def _get_constraint_violations(self) -> np.ndarray:
        """Compute normalized constraint violations for each OAR + tumor."""
        violations = np.zeros(4, dtype=np.float32)
        if self.current_dose is None or self.patient is None:
            return violations

        # Tumor: how far from 95% coverage target
        tumor_dose = self.current_dose[self.patient.tumor_mask]
        if len(tumor_dose) > 0:
            coverage = np.mean(tumor_dose >= 0.95 * self.patient.prescription_dose)
            violations[0] = float(1.0 - coverage)  # 0 = perfect, 1 = no coverage

        # OARs: how much they exceed their dose limits
        for i, oar in enumerate(self.patient.oars[:3]):
            oar_dose = self.current_dose[oar.mask]
            if len(oar_dose) > 0:
                mean_dose = np.mean(oar_dose)
                violations[i + 1] = float(max(0, mean_dose - oar.limit) / oar.limit)

        return violations

    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary info dict."""
        info = {
            "step_count": self.step_count,
            "n_beams": len(self.beams),
            "task": self.task_name,
        }
        if self.patient is not None and self.current_dose is not None:
            info["dvh_summary"] = self.dose_calculator.get_dvh_summary(
                self.current_dose, self.patient
            )
            info["score"] = self.get_score()
        return info
