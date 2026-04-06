"""Task 2: Head & Neck (Medium)"""
from .base_task import BaseTask
from ..physics.phantom import HeadNeckPatientGenerator
from ..reward.reward_fn import compute_reward


class HeadNeckTask(BaseTask):
    """Medium task — head & neck cancer with 7 OARs."""
    def __init__(self):
        self._gen = HeadNeckPatientGenerator()

    def sample_patient(self, rng):
        return self._gen.generate(rng)

    def reward(self, dose, patient, beams):
        # Extra penalty multiplier for head & neck complexity
        base = compute_reward(dose, patient, beams)
        # Slight difficulty boost: slightly harder reward shaping
        return float(base * 0.95)
