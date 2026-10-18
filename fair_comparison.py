
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from environment_check import create_fixed_match_environment
from agents import (
    action_to_vector,
    random_legal_action,
    select_opponent_action,
)
from dqn_agent import DQNAgent


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
MODEL_PATH = PROJECT_DIR / "models" / "dqn_opponent_modeling.pt"

SEEDS = [42, 123, 2026]
MATCHES_PER_CONFIGURATION = 30
MAX_STEPS = 100_000

CONFIGURATIONS = {
    "all_random": {
        "Agent2": "random",
        "Agent3": "random",
        "Agent4": "random",
    },
    "mixed": {
        "Agent2": "random",
        "Agent3": "rule_based",
        "Agent4": "rule_based",
    },
    "all_rule_based": {
        "Agent2": "rule_based",
        "Agent3": "rule_based",
        "Agent4": "rule_based",
    },
}


def load_dqn(seed):
    """Load the trained DQN and disable exploration."""
    agent = DQNAgent(seed=seed)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    agent.policy_net.load_state_dict(checkpoint["policy_net"])
    agent.target_net.load_state_dict(checkpoint["target_net"])
    agent.epsilon = 0.0
    agent.policy_net.eval()

    return agent


def run_evaluation(agent_type, config_name, policies, seed):
    """Evaluate one agent against one opponent setup and seed."""
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    env, observation, _ = create_fixed_match_environment(
        number_of_matches=MATCHES_PER_CONFIGURATION + 1,
        seed=seed,
    )

    game = env.unwrapped
    agent = load_dqn(seed) if agent_type == "DQN" else None

    results = []
    steps = 0
    original_start = game.start_match_handle_cards

    def capture_before_next_match():
        """Capture scores before the next match resets player states."""
        player = next(
            p for p in game.players if p.name == "Agent1"
        )

        position = int(player.finishing_position)
        score = int(player.current_score)

        if position not in (0, 1, 2, 3) or score != 3 - position:
            raise RuntimeError(
                "Invalid match result captured before player reset: "
                f"position={position}, score={score}"
            )

        results.append({
            "agent_type": agent_type,
            "configuration": config_name,
            "seed": seed,
            "match": len(results) + 1,
            "agent1_position": position,
            "agent1_match_score": score,
            "win": int(position == 0),
            "steps": steps,
        })

        original_start()

    game.start_match_handle_cards = capture_before_next_match

    try:
        while (
            len(results) < MATCHES_PER_CONFIGURATION
            and steps < MAX_STEPS
        ):
            player_name = game.players[game.currentPlayer].name

            if player_name == "Agent1":
                if agent_type == "DQN":
                    action_index = agent.select_action(observation)
                else:
                    action_index = random_legal_action(observation, rng)
            else:
                action_index = select_opponent_action(
                    observation,
                    policies[player_name],
                    rng,
                )

            observation, reward, terminated, truncated, info = env.step(
                action_to_vector(action_index)
            )
            steps += 1

        if len(results) != MATCHES_PER_CONFIGURATION:
            raise RuntimeError(
                f"{agent_type}, {config_name}, seed {seed}: "
                f"captured {len(results)} of "
                f"{MATCHES_PER_CONFIGURATION} matches."
            )

        return results

    finally:
        game.start_match_handle_cards = original_start
        env.close()


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for seed in SEEDS:
        for config_name, policies in CONFIGURATIONS.items():
            for agent_type in ["RandomBaseline", "DQN"]:
                print(
                    f"\nSeed {seed} | {config_name} | {agent_type}"
                )

                results = run_evaluation(
                    agent_type,
                    config_name,
                    policies,
                    seed,
                )
                all_results.extend(results)

                wins = sum(row["win"] for row in results)
                average_score = np.mean([
                    row["agent1_match_score"] for row in results
                ])
                average_position = np.mean([
                    row["agent1_position"] for row in results
                ])

                print(
                    f"Matches: {len(results)} | "
                    f"Wins: {wins} | "
                    f"Win rate: {wins / len(results):.3f} | "
                    f"Average score: {average_score:.3f} | "
                    f"Average position: {average_position:.3f}"
                )

    raw_path = RESULTS_DIR / "fair_comparison_raw.csv"
    summary_path = RESULTS_DIR / "fair_comparison_summary.csv"

    df = pd.DataFrame(all_results)
    df.to_csv(raw_path, index=False)

    # First average within each seed, then summarize across seeds.
    per_seed = (
        df.groupby(["agent_type", "configuration", "seed"])
        .agg(
            matches=("win", "count"),
            wins=("win", "sum"),
            win_rate=("win", "mean"),
            average_score=("agent1_match_score", "mean"),
            average_position=("agent1_position", "mean"),
        )
        .reset_index()
    )

    summary = (
        per_seed.groupby(["agent_type", "configuration"])
        .agg(
            total_matches=("matches", "sum"),
            mean_seed_win_rate=("win_rate", "mean"),
            sd_seed_win_rate=("win_rate", "std"),
            mean_seed_score=("average_score", "mean"),
            mean_seed_position=("average_position", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(summary_path, index=False)

    print("\nFinal fair comparison (mean across seeds):")
    print(summary.round(3).to_string(index=False))
    print("\nRaw results:", raw_path)
    print("Summary:", summary_path)
    print("Total matches:", len(df))


if __name__ == "__main__":
    main()
