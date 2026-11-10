"""
PyForgeAI Showcase Demo
-----------------------
Demonstrates training game AI agents using both:
1. Neuroevolution (GeneticTrainer with Elitism, Adaptive Mutation & Diversity Injection)
2. Reinforcement Learning (QLearningTrainer with Double-DQN & Experience Replay)
on the built-in TargetSeekEnvironment, plus model serialization and live adapter inference.
"""

from __future__ import annotations

import os
import sys
import tempfile

# Ensure local `src` is importable when running without `pip install -e .`
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pyforgeai import (
    AIFactory,
    EarlyStoppingCallback,
    HistoryCallback,
    TargetSeekEnvironment,
    load_model,
    save_model,
)


def run_showcase() -> None:
    print("=" * 68)
    print(" PyForgeAI v1.0.0 — Neuroevolution & Deep Q-Learning Showcase")
    print("=" * 68)

    env = TargetSeekEnvironment(max_steps=60, seed=42)
    factory = AIFactory(
        {
            "type_model": "Pytorch",
            "input_size": env.state_size,
            "output_size": env.action_size,
            "hidden_sizes": [16, 16],
        }
    )

    # ------------------------------------------------------------------
    # 1. Neuroevolution (Genetic Algorithm)
    # ------------------------------------------------------------------
    print("\n[1/3] Training with GeneticTrainer (Adaptive Mutation + Elitism)...")
    ga_history = HistoryCallback()
    ga_trainer = factory.create_GeneticTrainer(
        generations=12,
        population_size=14,
        num_trials=2,
        elitism=2,
        adaptive_mutation=True,
        diversity_interval=5,
        callbacks=[ga_history, EarlyStoppingCallback(patience=6)],
        verbose=True,
    )
    ga_result = ga_trainer.train_with_result(env)
    print(
        f"-> Genetic Training Finished: Best Fitness = {ga_result.best_score:.2f} "
        f"({ga_result.iterations_completed} generations)"
    )

    # ------------------------------------------------------------------
    # 2. Deep Q-Learning (Double DQN)
    # ------------------------------------------------------------------
    print("\n[2/3] Training with QLearningTrainer (Double-DQN + ReplayMemory)...")
    dqn_trainer = factory.create_QLearningTrainer(
        episodes=30,
        lr=2e-3,
        batch_size=16,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.95,
        double_dqn=True,
        verbose=True,
    )
    dqn_result = dqn_trainer.train_environment_with_result(env)
    print(
        f"-> DQN Training Finished: Best Episode Reward = {dqn_result.best_score:.2f} "
        f"({dqn_result.iterations_completed} episodes)"
    )

    # ------------------------------------------------------------------
    # 3. Serialization & Runtime Controller Inference
    # ------------------------------------------------------------------
    print("\n[3/3] Testing Model Serialization & Runtime AIController Inference...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "champion_agent.pth")
        save_model(ga_result.best_model, ckpt_path)
        restored_model = load_model(ckpt_path, "Pytorch")

        controller = factory.create_Controller(state_provider=env.reset)
        adapter = controller.set_runtime_model(restored_model)

        obs = env.reset()
        action = controller.select_action(obs, deterministic=True)
        probs = adapter.predict_probabilities(obs)
        activations = adapter.get_activations()

        print(f"-> Saved & Restored Model Architecture: hidden_sizes={restored_model.hidden_sizes}")
        print(f"-> Sample State:        {obs.round(3)}")
        print(f"-> Action Probabilities:{probs.round(3)} -> Chosen Action: {action}")
        if activations is not None:
            print(f"-> Normalized Layer-1 Activations (first 6 neurons): {activations.flatten()[:6].round(3)}")

    print("\nAll showcase steps completed successfully!")


if __name__ == "__main__":
    run_showcase()
