"""
OpenEnv server environment wrapper for RadiotherapyPlanningEnv.
Bridges the Gymnasium env to the OpenEnv HTTP server protocol.
"""

import numpy as np
import gymnasium as gym
import radiotherapy_env  # noqa: F401

from typing import Any


class RadiotherapyEnvironment:
    """OpenEnv-compatible environment wrapper."""

    def __init__(self):
        self.env = None
        self.task = "prostate"
        self._last_obs = None
        self._last_info = None

    def reset(self, task: str = "prostate") -> dict[str, Any]:
        task_to_env = {
            "prostate": "RadiotherapyEnv-prostate-v1",
            "head_neck": "RadiotherapyEnv-headneck-v1",
            "pediatric_brain": "RadiotherapyEnv-pediatricbrain-v1",
        }
        self.task = task
        env_id = task_to_env.get(task, "RadiotherapyEnv-prostate-v1")

        if self.env is not None:
            self.env.close()
        self.env = gym.make(env_id)
        obs, info = self.env.reset()
        self._last_obs = obs
        self._last_info = info

        return {
            "observation": self._format_obs(obs, info),
            "reward": 0.0,
            "done": False,
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        action_value = int(action.get("action", 0))
        obs, reward, terminated, truncated, info = self.env.step(action_value)
        done = terminated or truncated
        self._last_obs = obs
        self._last_info = info

        return {
            "observation": self._format_obs(obs, info),
            "reward": float(reward),
            "done": done,
        }

    def state(self) -> dict[str, Any]:
        if self.env is None:
            return {}
        return self.env.state()

    def _format_obs(self, obs: dict, info: dict) -> dict[str, Any]:
        """Convert numpy arrays to lists for JSON serialization."""
        return {
            "dvh_tumor": obs["dvh_tumor"].tolist(),
            "dvh_oar": obs["dvh_oar"].tolist(),
            "beams": obs["beams"].tolist(),
            "constraints": obs["constraints"].tolist(),
            "step_frac": obs["step_frac"].tolist(),
            "score": info.get("score", 0.0),
            "n_beams": info.get("n_beams", 0),
            "task": self.task,
        }

    def close(self):
        if self.env is not None:
            self.env.close()
