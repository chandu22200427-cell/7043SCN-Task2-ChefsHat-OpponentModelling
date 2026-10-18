
import csv
import random
from pathlib import Path

import numpy as np
import torch

from environment_check import create_fixed_match_environment
from agents import action_to_vector, select_opponent_action
from dqn_agent import DQNAgent


SEED = 123
MATCHES_PER_CONFIGURATION = 30

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODELS_DIR / "dqn_opponent_modeling.pt"
OUTPUT_PATH = RESULTS_DIR / "dqn_evaluation.csv"

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


def load_greedy_agent():
    """Load the trained network and disable exploration."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run train_dqn.py first."
        )

    agent = DQNAgent(seed=SEED)
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    agent.policy_net.load_state_dict(checkpoint["policy_net"])
    agent.target_net.load_state_dict(checkpoint["target_net"])
    agent.epsilon = 0.0
    agent.policy_net.eval()
    agent.target_net.eval()

    return agent


def evaluate_configuration(config_name, opponent_policies):
    """Evaluate a greedy DQN against one opponent configuration."""
    random.seed(SEED)
    np.random.seed(SEED)

    agent = load_greedy_agent()

    env, observation, _ = create_fixed_match_environment(
        number_of_matches=MATCHES_PER_CONFIGURATION + 1,
        seed=SEED,
    )
    game = env.unwrapped

    completed_results = []
    total_steps = 0
    max_steps = 100_000

    original_start_match = game.start_match_handle_cards

    def capture_before_next_match():
        """Capture the completed match before player state resets."""
        player = next(
            p for p in game.players if p.name == "Agent1"
        )

        position = int(player.finishing_position)
        score = int(player.current_score)

        if position in (0, 1, 2, 3) and score == 3 - position:
            completed_results.append(
                {
                    "configuration": config_name,
                    "match": len(completed_results) + 1,
                    "seed": SEED,
                    "agent1_position": position,
                    "agent1_match_score": score,
                    "steps": total_steps,
                }
            )

        original_start_match()

    game.start_match_handle_cards = capture_before_next_match

    try:
        while (
            len(completed_results) < MATCHES_PER_CONFIGURATION
            and total_steps < max_steps
        ):
            player_name = game.players[game.currentPlayer].name

            if player_name == "Agent1":
                action_index = agent.select_action(observation)
            else:
                action_index = select_opponent_action(
                    observation,
                    opponent_policies[player_name],
                )

            observation, reward, terminated, truncated, info = env.step(
                action_to_vector(action_index)
            )
            total_steps += 1

        if len(completed_results) != MATCHES_PER_CONFIGURATION:
            raise RuntimeError(
                f"{config_name}: captured {len(completed_results)} "
                f"of {MATCHES_PER_CONFIGURATION} matches."
            )

        return completed_results

    finally:
        game.start_match_handle_cards = original_start_match
        env.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for config_name, policies in CONFIGURATIONS.items():
        print(f"\n{'=' * 50}")
        print("Evaluating:", config_name)
        print("=" * 50)

        results = evaluate_configuration(config_name, policies)
        all_results.extend(results)

        positions = [row["agent1_position"] for row in results]
        scores = [row["agent1_match_score"] for row in results]

        wins = sum(position == 0 for position in positions)

        print("Matches:", len(results))
        print(f"Wins: {wins}/{len(results)}")
        print("Win rate:", round(wins / len(results), 3))
        print("Average score:", round(float(np.mean(scores)), 3))
        print(
            "Average finishing position:",
            round(float(np.mean(positions)), 3),
        )

    with OUTPUT_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "configuration",
                "match",
                "seed",
                "agent1_position",
                "agent1_match_score",
                "steps",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    print("\nEvaluation saved to:", OUTPUT_PATH)
    print("Total evaluated matches:", len(all_results))


if __name__ == "__main__":
    main()
