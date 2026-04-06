"""
Evaluate Baseline Agent
=======================
Runs the pre-trained PPO agent on all 3 tasks and prints reproducible scores.
Used to verify that the environment is solvable and scores are consistent.

Usage:
    python baseline/evaluate.py
    python baseline/evaluate.py --task prostate --model baseline/models/prostate_best/best_model
"""

import os
import sys
import argparse
import json
import numpy as np
import gymnasium as gym
import radiotherapy_env


def random_agent(obs, env):
    """Random baseline agent."""
    return env.action_space.sample()


def smart_heuristic_agent(obs, env):
    """
    Simple heuristic agent for baseline comparison.
    Strategy:
      - First 7 steps: add beams (action 0)
      - Next steps: rotate or fine-tune
      - Final step: lock plan
    """
    step = int(obs["step_frac"][0] * env.spec.max_episode_steps)
    n_beams = int(np.sum(obs["beams"][:, 2]))  # active beams

    if n_beams < 6 and step < 30:
        return 0  # add beam
    elif step < 40:
        # Rotate based on constraint violations
        violations = obs["constraints"]
        if violations[0] > 0.3:  # tumor under-covered
            return 3  # increase dose
        elif violations[1:].max() > 0.5:  # OAR violated
            return 1  # rotate
        else:
            return 6  # fine-tune
    else:
        return 7  # lock plan


def evaluate(task: str, model_path: str = None, n_episodes: int = 30, seed: int = 42):
    """
    Evaluate an agent on a specific task.
    If model_path is None, uses the heuristic baseline.
    """
    task_to_env = {
        "prostate":        "RadiotherapyEnv-prostate-v1",
        "head_neck":       "RadiotherapyEnv-headneck-v1",
        "pediatric_brain": "RadiotherapyEnv-pediatricbrain-v1",
    }
    env_id = task_to_env[task]
    env = gym.make(env_id)

    # Load trained model if provided
    if model_path and os.path.exists(model_path + ".zip"):
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            agent_type = "PPO (trained)"
            def agent_fn(obs, _env):
                action, _ = model.predict(obs, deterministic=True)
                return int(action)
        except ImportError:
            agent_type = "Heuristic (SB3 not available)"
            agent_fn = smart_heuristic_agent
    else:
        agent_type = "Heuristic baseline"
        agent_fn = smart_heuristic_agent

    scores = []
    rewards_per_ep = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action = agent_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        scores.append(info.get("score", 0.0))
        rewards_per_ep.append(ep_reward)

    env.close()

    scores = np.array(scores)
    result = {
        "task":       task,
        "env_id":     env_id,
        "agent_type": agent_type,
        "n_episodes": n_episodes,
        "seed":       seed,
        "mean_score": float(np.mean(scores)),
        "std_score":  float(np.std(scores)),
        "min_score":  float(np.min(scores)),
        "max_score":  float(np.max(scores)),
        "pass_rate":  float(np.mean(scores >= 0.6)),
        "mean_reward_per_ep": float(np.mean(rewards_per_ep)),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None,
                        choices=["prostate", "head_neck", "pediatric_brain"])
    parser.add_argument("--model", default=None,
                        help="Path to trained model (without .zip)")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = ([args.task] if args.task
             else ["prostate", "head_neck", "pediatric_brain"])

    print("\n" + "="*65)
    print("  RadiotherapyPlanningEnv — Baseline Evaluation Report")
    print("="*65)

    all_results = {}
    for task in tasks:
        model_path = args.model or f"baseline/models/{task}_best/best_model"
        result = evaluate(task, model_path, args.episodes, args.seed)
        all_results[task] = result

        diff = {"prostate": "Easy", "head_neck": "Medium", "pediatric_brain": "Hard"}
        print(f"\n  [{diff[task].upper()}] {task.replace('_', ' ').title()}")
        print(f"  Agent:       {result['agent_type']}")
        print(f"  Mean score:  {result['mean_score']:.3f} ± {result['std_score']:.3f}")
        print(f"  Range:       [{result['min_score']:.3f}, {result['max_score']:.3f}]")
        print(f"  Pass rate:   {result['pass_rate']*100:.1f}%  (threshold: score >= 0.6)")

    print("\n" + "─"*65)
    total = np.mean([r["mean_score"] for r in all_results.values()])
    print(f"  Aggregate score (mean across tasks): {total:.3f}")
    print("─"*65 + "\n")

    # Save results
    with open("baseline/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results saved to baseline/results.json")


if __name__ == "__main__":
    main()
