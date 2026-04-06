"""Task 1: Prostate (Easy)"""
from .base_task import BaseTask
from ..physics.phantom import PatientPhantom, ProstatePatientGenerator
from ..reward.reward_fn import compute_reward


class ProstateTask(BaseTask):
    """Easy task — prostate cancer with 2 OARs."""
    def __init__(self):
        self._gen = ProstatePatientGenerator()

    def sample_patient(self, rng):
        return self._gen.generate(rng)

    def reward(self, dose, patient, beams):
        return compute_reward(dose, patient, beams)
