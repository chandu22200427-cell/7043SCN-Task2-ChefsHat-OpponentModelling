
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from environment_check import create_fixed_match_environment
from agents import action_to_vector, random_legal_action, select_opponent_action
from dqn_agent import DQNAgent
from opponent_model import OpponentModel


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

ORIGINAL_MODEL = PROJECT_DIR / "models" / "dqn_opponent_modeling.pt"
AWARE_MODEL = PROJECT_DIR / "models" / "dqn_opponent_aware.pt"

SEEDS = [42, 123, 2026]
MATCHES_PER_CONFIGURATION = 30
MAX_STEPS = 100_000

OPPONENT_NAMES = ("Agent2", "Agent3", "Agent4")
BASE_STATE_SIZE = 228
STATE_SIZE = 234
ACTION_SIZE = 200

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


def augment_observation(observation, opponent_model):
    """Append pass/play estimates for each of the three opponents."""
    observation = np.asarray(observation, dtype=np.float32)

    if observation.shape != (BASE_STATE_SIZE,):
        raise ValueError(
            f"Expected base state shape ({BASE_STATE_SIZE},), "
            f"received {observation.shape}"
        )

    features = []
    for name in OPPONENT_NAMES:
        features.extend(opponent_model.get_state_features(name))

    augmented = np.concatenate(
        [observation, np.asarray(features, dtype=np.float32)]
    )

    if augmented.shape != (STATE_SIZE,):
        raise ValueError(f"Unexpected augmented state shape: {augmented.shape}")

    return augmented


def load_agent(model_path, state_size, seed):
    """Load a trained checkpoint and disable exploration."""
    agent = DQNAgent(
        state_size=state_size,
        action_size=ACTION_SIZE,
        seed=seed,
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("state_size", state_size) != state_size:
        raise ValueError(
            f"{model_path.name} expects state size "
            f"{checkpoint.get('state_size')}, not {state_size}"
        )

    agent.policy_net.load_state_dict(checkpoint["policy_net"])
    agent.target_net.load_state_dict(checkpoint["target_net"])
    agent.epsilon = 0.0
    agent.policy_net.eval()

    return agent


def run_evaluation(agent_type, configuration, policies, seed):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    env, observation, _ = create_fixed_match_environment(
        number_of_matches=MATCHES_PER_CONFIGURATION + 1,
        seed=seed,
    )

    game = env.unwrapped
    original_start = game.start_match_handle_cards

    if agent_type == "RandomBaseline":
        agent = None
    elif agent_type == "DQN":
        agent = load_agent(ORIGINAL_MODEL, BASE_STATE_SIZE, seed)
    elif agent_type == "OpponentAwareDQN":
        agent = load_agent(AWARE_MODEL, STATE_SIZE, seed)
    else:
        env.close()
        raise ValueError(f"Unknown agent type: {agent_type}")

    opponent_model = OpponentModel(OPPONENT_NAMES)
    results = []
    steps = 0

    def capture_before_next_match():
        """Record the completed match before player scores reset."""
        player = next(p for p in game.players if p.name == "Agent1")
        position = int(player.finishing_position)
        score = int(player.current_score)

        if position not in (0, 1, 2, 3) or score != 3 - position:
            raise RuntimeError(
                "Invalid match result: "
                f"position={position}, score={score}"
            )

        results.append({
            "agent_type": agent_type,
            "configuration": configuration,
            "seed": seed,
            "match": len(results) + 1,
            "agent1_position": position,
            "agent1_match_score": score,
            "win": int(position == 0),
            "steps": steps,
        })

        opponent_model.reset()
        original_start()

    game.start_match_handle_cards = capture_before_next_match

    try:
        while (
            len(results) < MATCHES_PER_CONFIGURATION
            and steps < MAX_STEPS
        ):
            player_name = game.players[game.currentPlayer].name

            if agent_type == "OpponentAwareDQN":
                state = augment_observation(observation, opponent_model)
            else:
                state = observation

            if player_name == "Agent1":
                if agent_type == "RandomBaseline":
                    action_index = random_legal_action(observation, rng)
                else:
                    action_index = agent.select_action(state)
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

            # Learn from the observed opponent action for future decisions.
            if (
                agent_type == "OpponentAwareDQN"
                and player_name in OPPONENT_NAMES
                and info.get("Action_Valid", True)
            ):
                opponent_model.update(player_name, action_index)

        if len(results) != MATCHES_PER_CONFIGURATION:
            raise RuntimeError(
                f"{agent_type}, {configuration}, seed {seed}: "
                f"captured {len(results)} of "
                f"{MATCHES_PER_CONFIGURATION} matches after {steps} steps."
            )

        return results

    finally:
        game.start_match_handle_cards = original_start
        env.close()


def main():
    for path in (ORIGINAL_MODEL, AWARE_MODEL):
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    agent_types = ["RandomBaseline", "DQN", "OpponentAwareDQN"]

    for seed in SEEDS:
        for configuration, policies in CONFIGURATIONS.items():
            for agent_type in agent_types:
                print(
                    f"\nSeed {seed} | {configuration} | {agent_type}",
                    flush=True,
                )

                results = run_evaluation(
                    agent_type, configuration, policies, seed
                )
                all_results.extend(results)

                wins = sum(row["win"] for row in results)
                scores = [row["agent1_match_score"] for row in results]
                positions = [row["agent1_position"] for row in results]

                print(
                    f"Matches: {len(results)} | "
                    f"Wins: {wins} | "
                    f"Win rate: {wins / len(results):.3f} | "
                    f"Average score: {np.mean(scores):.3f} | "
                    f"Average position: {np.mean(positions):.3f}",
                    flush=True,
                )

    raw_path = RESULTS_DIR / "opponent_aware_evaluation_raw.csv"
    summary_path = RESULTS_DIR / "opponent_aware_evaluation_summary.csv"

    df = pd.DataFrame(all_results)
    df.to_csv(raw_path, index=False)

    # Calculate metrics within each seed first.
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

    print("\nFinal evaluation summary:")
    print(summary.round(3).to_string(index=False))
    print(f"\nRaw results: {raw_path}")
    print(f"Summary: {summary_path}")
    print(f"Total matches: {len(df)}")


if __name__ == "__main__":
    main()
