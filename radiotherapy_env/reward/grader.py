"""
Auto-Grader
===========
Evaluates a trained agent on all 3 tasks and produces standardized scores.
Used by the OpenEnv leaderboard system.
"""

import numpy as np
from typing import Dict, List
import gymnasium as gym
import radiotherapy_env  # noqa: F401 — registers all gym environments


def grade_task(env_id: str, agent_fn, n_episodes: int = 20, seed: int = 42) -> Dict:
    """
    Grade an agent on a specific task.

    Args:
        env_id:      One of the registered environment IDs
        agent_fn:    Callable: (obs, env) -> action
        n_episodes:  Number of evaluation episodes
        seed:        Random seed for reproducibility

    Returns:
        Dict with mean_score, std_score, min_score, max_score, pass_rate
    """
    env = gym.make(env_id)
    scores = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            action = agent_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        episode_score = info.get("score", 0.0)
        scores.append(episode_score)

    env.close()
    scores = np.array(scores)

    return {
        "env_id": env_id,
        "n_episodes": n_episodes,
        "mean_score":  float(np.mean(scores)),
        "std_score":   float(np.std(scores)),
        "min_score":   float(np.min(scores)),
        "max_score":   float(np.max(scores)),
        "pass_rate":   float(np.mean(scores >= 0.6)),  # clinical pass threshold
        "scores":      scores.tolist(),
    }


def grade_all(agent_fn, n_episodes: int = 20, seed: int = 42) -> Dict:
    """
    Grade an agent on all 3 tasks.

    Returns:
        Full grading report with per-task and aggregate scores.
    """
    tasks = [
        ("RadiotherapyEnv-prostate-v1",      "easy"),
        ("RadiotherapyEnv-headneck-v1",      "medium"),
        ("RadiotherapyEnv-pediatricbrain-v1","hard"),
    ]

    results = {}
    total_score = 0.0

    for env_id, difficulty in tasks:
        print(f"  Grading {difficulty} task ({env_id})...")
        result = grade_task(env_id, agent_fn, n_episodes=n_episodes, seed=seed)
        result["difficulty"] = difficulty
        results[difficulty] = result
        total_score += result["mean_score"]
        print(f"    Score: {result['mean_score']:.3f} ± {result['std_score']:.3f}  "
              f"Pass rate: {result['pass_rate']*100:.1f}%")

    aggregate = total_score / len(tasks)

    return {
        "aggregate_score": aggregate,
        "tasks": results,
        "summary": {
            "easy":   results["easy"]["mean_score"],
            "medium": results["medium"]["mean_score"],
            "hard":   results["hard"]["mean_score"],
            "total":  aggregate,
        }
    }
