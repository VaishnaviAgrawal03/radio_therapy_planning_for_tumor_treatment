"""
Baseline PPO Training Script
=============================
Trains a PPO agent on all three RadiotherapyEnv difficulty levels.
Uses stable-baselines3 with a MultiInputPolicy for Dict observations.

Usage:
    python baseline/train_ppo.py --task prostate --timesteps 200000
    python baseline/train_ppo.py --task head_neck --timesteps 300000
    python baseline/train_ppo.py --task pediatric_brain --timesteps 500000
    python baseline/train_ppo.py --all  # train all three sequentially
"""

import os
import sys
import argparse
import json
import time

import numpy as np
import gymnasium as gym
import radiotherapy_env  # registers all envs


def make_env(env_id: str, seed: int = 0, render_mode=None):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env
    return _init


def train(task: str, timesteps: int, save_dir: str = "baseline/models"):
    """Train a PPO agent on the given task."""
    task_to_env = {
        "prostate":       "RadiotherapyEnv-prostate-v1",
        "head_neck":      "RadiotherapyEnv-headneck-v1",
        "pediatric_brain":"RadiotherapyEnv-pediatricbrain-v1",
    }

    assert task in task_to_env, f"Unknown task: {task}"
    env_id = task_to_env[task]

    print(f"\n{'='*60}")
    print(f"  Training PPO on: {env_id}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"{'='*60}\n")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import VecMonitor
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    except ImportError:
        print("ERROR: stable-baselines3 not installed.")
        print("Run: pip install stable-baselines3")
        return None

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"baseline/logs/{task}", exist_ok=True)

    # Vectorized environments for faster training
    n_envs = 4
    vec_env = make_vec_env(env_id, n_envs=n_envs, seed=42)
    eval_env = make_vec_env(env_id, n_envs=1, seed=100)

    # PPO with MultiInputPolicy handles Dict observations natively
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=f"baseline/logs/{task}",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/{task}_best",
        log_path=f"baseline/logs/{task}",
        eval_freq=max(1000, timesteps // 20),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(5000, timesteps // 10),
        save_path=f"{save_dir}/{task}_checkpoints",
        name_prefix=f"ppo_{task}",
    )

    # Train
    t0 = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    train_time = time.time() - t0

    # Save final model
    final_path = f"{save_dir}/{task}_final"
    model.save(final_path)
    print(f"\nModel saved to: {final_path}.zip")
    print(f"Training time: {train_time/60:.1f} minutes")

    # Evaluate final model
    print(f"\nEvaluating final model...")
    scores = evaluate_model(model, env_id, n_episodes=20, seed=200)

    result = {
        "task": task,
        "env_id": env_id,
        "timesteps": timesteps,
        "train_time_sec": round(train_time, 1),
        "mean_score": round(float(np.mean(scores)), 4),
        "std_score":  round(float(np.std(scores)),  4),
        "min_score":  round(float(np.min(scores)),  4),
        "max_score":  round(float(np.max(scores)),  4),
        "pass_rate":  round(float(np.mean(np.array(scores) >= 0.6)), 3),
        "scores": [round(s, 4) for s in scores],
    }

    # Save results
    results_path = f"baseline/results_{task}.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {results_path}")

    print(f"\n{'─'*40}")
    print(f"  Task:      {task}")
    print(f"  Mean score: {result['mean_score']:.3f} ± {result['std_score']:.3f}")
    print(f"  Pass rate:  {result['pass_rate']*100:.1f}%")
    print(f"{'─'*40}")

    vec_env.close()
    eval_env.close()
    return result


def evaluate_model(model, env_id: str, n_episodes: int = 20, seed: int = 0):
    """Evaluate a trained model and return per-episode scores."""
    env = gym.make(env_id)
    scores = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        scores.append(info.get("score", 0.0))

    env.close()
    return scores


def main():
    parser = argparse.ArgumentParser(description="Train PPO on RadiotherapyEnv")
    parser.add_argument("--task", type=str, default="prostate",
                        choices=["prostate", "head_neck", "pediatric_brain"],
                        help="Task to train on")
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Total training timesteps")
    parser.add_argument("--all", action="store_true",
                        help="Train all three tasks sequentially")
    args = parser.parse_args()

    if args.all:
        tasks = [
            ("prostate",       200_000),
            ("head_neck",      350_000),
            ("pediatric_brain",1_000_000),
        ]
        all_results = {}
        for task, ts in tasks:
            result = train(task, ts)
            if result:
                all_results[task] = result

        # Combined results
        with open("baseline/results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\n✓ All tasks trained. Results saved to baseline/results.json")
    else:
        train(args.task, args.timesteps)


if __name__ == "__main__":
    main()
