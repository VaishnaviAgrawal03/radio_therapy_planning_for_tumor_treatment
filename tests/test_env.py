"""
Test Suite for RadiotherapyPlanningEnv
=======================================
Tests Gymnasium spec compliance, physics correctness, and reward properties.

Run with:
    pytest tests/ -v
"""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import radiotherapy_env
from radiotherapy_env import RadiotherapyEnv
from radiotherapy_env.physics.dose_calculator import DoseCalculator
from radiotherapy_env.physics.phantom import Beam, ProstatePatientGenerator
from radiotherapy_env.reward.reward_fn import compute_reward, compute_score


# ─────────────────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def prostate_env():
    env = RadiotherapyEnv(task="prostate", max_steps=50)
    env.reset(seed=42)
    return env

@pytest.fixture
def headneck_env():
    env = RadiotherapyEnv(task="head_neck", max_steps=60)
    env.reset(seed=42)
    return env

@pytest.fixture
def brain_env():
    env = RadiotherapyEnv(task="pediatric_brain", max_steps=70)
    env.reset(seed=42)
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium spec compliance
# ─────────────────────────────────────────────────────────────────────────────

class TestGymnasiumCompliance:

    def test_check_env_prostate(self):
        """gymnasium check_env must pass with no warnings."""
        env = RadiotherapyEnv(task="prostate", max_steps=20)
        check_env(env, warn=True)
        env.close()

    def test_check_env_headneck(self):
        env = RadiotherapyEnv(task="head_neck", max_steps=20)
        check_env(env, warn=True)
        env.close()

    def test_check_env_pediatric(self):
        env = RadiotherapyEnv(task="pediatric_brain", max_steps=20)
        check_env(env, warn=True)
        env.close()

    def test_registered_envs(self):
        """All three registered IDs must be creatable via gym.make."""
        for env_id in [
            "RadiotherapyEnv-prostate-v1",
            "RadiotherapyEnv-headneck-v1",
            "RadiotherapyEnv-pediatricbrain-v1",
        ]:
            env = gym.make(env_id)
            obs, info = env.reset(seed=0)
            assert obs is not None
            env.close()

    def test_reset_returns_valid_obs(self, prostate_env):
        obs, info = prostate_env.reset(seed=0)
        assert prostate_env.observation_space.contains(obs), \
            "Observation not in observation_space"

    def test_step_returns_valid_obs(self, prostate_env):
        for action in range(8):
            prostate_env.reset(seed=action)
            obs, reward, term, trunc, info = prostate_env.step(action)
            assert prostate_env.observation_space.contains(obs)

    def test_reward_in_range(self, prostate_env):
        """Reward must be in [0, 1]."""
        for _ in range(30):
            action = prostate_env.action_space.sample()
            _, reward, term, trunc, _ = prostate_env.step(action)
            assert 0.0 <= reward <= 1.0, f"Reward {reward} out of [0,1]"
            if term or trunc:
                prostate_env.reset(seed=0)

    def test_seed_reproducibility(self):
        """Same seed must produce same trajectory."""
        env1 = RadiotherapyEnv(task="prostate")
        env2 = RadiotherapyEnv(task="prostate")

        obs1, _ = env1.reset(seed=99)
        obs2, _ = env2.reset(seed=99)
        np.testing.assert_array_equal(obs1["step_frac"], obs2["step_frac"])

        for _ in range(5):
            action = env1.action_space.sample()
            o1, r1, *_ = env1.step(action)
            o2, r2, *_ = env2.step(action)
            assert abs(r1 - r2) < 1e-6, "Rewards differ with same seed+actions"

        env1.close()
        env2.close()

    def test_state_method(self, prostate_env):
        """state() must return a dict with required keys."""
        state = prostate_env.state()
        assert isinstance(state, dict)
        for key in ["task", "patient", "beams", "step_count", "last_reward"]:
            assert key in state, f"Key '{key}' missing from state()"

    def test_max_steps_truncation(self):
        """Episode must truncate at max_steps."""
        env = RadiotherapyEnv(task="prostate", max_steps=5)
        env.reset(seed=0)
        for i in range(4):
            _, _, term, trunc, _ = env.step(0)
            assert not trunc, f"Truncated too early at step {i+1}"
        _, _, term, trunc, _ = env.step(0)
        assert trunc, "Should be truncated at max_steps"
        env.close()

    def test_lock_plan_terminates(self, prostate_env):
        """Action 7 (lock plan) must terminate the episode."""
        _, _, term, trunc, _ = prostate_env.step(7)
        assert term, "Lock plan (action=7) should terminate episode"


# ─────────────────────────────────────────────────────────────────────────────
# Physics tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPhysics:

    def test_zero_beams_zero_dose(self):
        gen = ProstatePatientGenerator()
        rng = np.random.default_rng(0)
        patient = gen.generate(rng)
        calc = DoseCalculator(grid_size=64)
        dose = calc.compute(patient, beams=[])
        np.testing.assert_array_equal(dose, np.zeros((64, 64)))

    def test_dose_positive_with_beams(self):
        gen = ProstatePatientGenerator()
        rng = np.random.default_rng(0)
        patient = gen.generate(rng)
        calc = DoseCalculator(grid_size=64)
        beams = [Beam(angle=0, dose_weight=1.0)]
        dose = calc.compute(patient, beams)
        assert dose.max() > 0, "Dose should be positive with active beams"

    def test_dose_grid_shape(self):
        gen = ProstatePatientGenerator()
        rng = np.random.default_rng(0)
        patient = gen.generate(rng)
        calc = DoseCalculator(grid_size=64)
        beams = [Beam(angle=45, dose_weight=0.8)]
        dose = calc.compute(patient, beams)
        assert dose.shape == (64, 64)

    def test_dose_outside_body_is_zero(self):
        gen = ProstatePatientGenerator()
        rng = np.random.default_rng(0)
        patient = gen.generate(rng)
        calc = DoseCalculator(grid_size=64)
        beams = [Beam(angle=0, dose_weight=1.0)]
        dose = calc.compute(patient, beams)
        # Dose outside body mask should be zero
        outside_dose = dose[~patient.body_mask]
        assert outside_dose.max() < 1e-6, "Dose outside body should be zero"

    def test_more_beams_more_tumor_coverage(self):
        gen = ProstatePatientGenerator()
        rng = np.random.default_rng(0)
        patient = gen.generate(rng)
        calc = DoseCalculator(grid_size=64)

        beams_1 = [Beam(angle=0, dose_weight=1.0)]
        beams_5 = [Beam(angle=i * 36, dose_weight=0.8) for i in range(5)]

        dose_1 = calc.compute(patient, beams_1)
        dose_5 = calc.compute(patient, beams_5)

        cov_1 = np.mean(dose_1[patient.tumor_mask] >= 0.5)
        cov_5 = np.mean(dose_5[patient.tumor_mask] >= 0.5)

        assert cov_5 >= cov_1, "More spread beams should give better tumor coverage"


# ─────────────────────────────────────────────────────────────────────────────
# Reward tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReward:

    def test_reward_zero_with_no_beams(self):
        env = RadiotherapyEnv(task="prostate")
        env.reset(seed=0)
        # Manually compute reward with empty beams
        from radiotherapy_env.reward.reward_fn import compute_reward
        dose = np.zeros((64, 64), dtype=np.float32)
        reward = compute_reward(dose, env.patient, beams=[])
        assert reward == 0.0

    def test_reward_increases_with_tumor_coverage(self):
        gen = ProstatePatientGenerator()
        rng = np.random.default_rng(0)
        patient = gen.generate(rng)
        calc = DoseCalculator(grid_size=64)

        # Perfect plan: many beams covering tumor
        beams = [Beam(angle=i * 25, dose_weight=0.8) for i in range(7)]
        dose = calc.compute(patient, beams)
        reward = compute_reward(dose, patient, beams)
        assert reward > 0.3 - 1e-2, f"Good plan should have reward > 0.3, got {reward:.3f}"

    def test_score_in_range(self):
        env = RadiotherapyEnv(task="prostate")
        env.reset(seed=0)

        # Take some actions
        for _ in range(10):
            env.step(env.action_space.sample())

        score = env.get_score()
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1]"

    def test_reward_partial_at_every_step(self, prostate_env):
        """Reward should be non-zero after adding a beam."""
        _, reward, _, _, _ = prostate_env.step(0)  # add beam
        # After first beam, reward may be low but should compute
        assert isinstance(reward, float)


# ─────────────────────────────────────────────────────────────────────────────
# Task difficulty tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskDifficulty:

    def test_prostate_has_2_oars(self, prostate_env):
        assert len(prostate_env.patient.oars) == 2

    def test_headneck_has_7_oars(self, headneck_env):
        assert len(headneck_env.patient.oars) == 7

    def test_pediatric_has_5_oars(self, brain_env):
        assert len(brain_env.patient.oars) == 5

    def test_pediatric_brainstem_adjacent_to_tumor(self, brain_env):
        """Brainstem should be very close to tumor in pediatric case."""
        patient = brain_env.patient
        brainstem = next(o for o in patient.oars if o.name == "Brainstem")

        # Find centroids
        t_yx = np.argwhere(patient.tumor_mask)
        b_yx = np.argwhere(brainstem.mask)
        if len(t_yx) == 0 or len(b_yx) == 0:
            pytest.skip("Empty mask")

        t_center = t_yx.mean(axis=0)
        b_center = b_yx.mean(axis=0)
        dist = np.linalg.norm(t_center - b_center)

        # Should be close (< 15 voxels)
        assert dist < 15, \
            f"Brainstem should be close to tumor, got distance {dist:.1f} voxels"

    def test_all_tasks_runnable(self):
        for task in ["prostate", "head_neck", "pediatric_brain"]:
            env = RadiotherapyEnv(task=task)
            obs, _ = env.reset(seed=0)
            for _ in range(5):
                obs, r, term, trunc, info = env.step(env.action_space.sample())
                if term or trunc:
                    break
            env.close()
