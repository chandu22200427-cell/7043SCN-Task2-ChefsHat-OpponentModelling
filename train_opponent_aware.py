
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from agents import action_to_vector, select_opponent_action
from dqn_agent import DQNAgent
from environment_check import create_fixed_match_environment
from opponent_model import OpponentModel


SEED = 42
TOTAL_MATCHES = 100
MAX_STEPS = 100_000
TARGET_UPDATE_MATCHES = 10

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = RESULTS_DIR / "opponent_aware_training_log.csv"
SUMMARY_PATH = RESULTS_DIR / "opponent_aware_training_summary.csv"
CURVE_PATH = RESULTS_DIR / "opponent_aware_training_curve.png"
MODEL_PATH = MODELS_DIR / "dqn_opponent_aware.pt"

BASE_STATE_SIZE = 228
OPPONENT_NAMES = ("Agent2", "Agent3", "Agent4")
STATE_SIZE = BASE_STATE_SIZE + 2 * len(OPPONENT_NAMES)
ACTION_SIZE = 200

CONFIGURATIONS = [
    ("all_random", {
        "Agent2": "random",
        "Agent3": "random",
        "Agent4": "random",
    }),
    ("mixed", {
        "Agent2": "random",
        "Agent3": "rule_based",
        "Agent4": "rule_based",
    }),
    ("all_rule_based", {
        "Agent2": "rule_based",
        "Agent3": "rule_based",
        "Agent4": "rule_based",
    }),
]


def augment_observation(observation, opponent_model):
    """Append two behaviour features for each opponent."""
    base = np.asarray(observation, dtype=np.float32)

    if base.shape != (BASE_STATE_SIZE,):
        raise ValueError(
            f"Expected observation shape ({BASE_STATE_SIZE},), "
            f"got {base.shape}"
        )

    features = []
    for name in OPPONENT_NAMES:
        features.extend(opponent_model.get_state_features(name))

    state = np.concatenate([
        base,
        np.asarray(features, dtype=np.float32),
    ])

    if state.shape != (STATE_SIZE,):
        raise RuntimeError(f"Unexpected augmented state shape: {state.shape}")

    return state


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        seed=SEED,
    )
    opponent_model = OpponentModel(OPPONENT_NAMES)

    env, observation, _ = create_fixed_match_environment(
        number_of_matches=TOTAL_MATCHES + 1,
        seed=SEED,
    )
    game = env.unwrapped

    completed_matches = 0
    total_steps = 0
    records = []
    losses = []

    pending_state = None
    pending_action = None
    pending_duration = 0

    original_start_match = game.start_match_handle_cards

    def finish_previous_match():
        """Capture the previous match result before player state resets."""
        nonlocal completed_matches
        nonlocal pending_state, pending_action, pending_duration
        nonlocal opponent_model

        if completed_matches >= TOTAL_MATCHES:
            original_start_match()
            return

        player = next(p for p in game.players if p.name == "Agent1")
        position = int(player.finishing_position)
        score = int(player.current_score)

        if position not in (0, 1, 2, 3):
            raise RuntimeError(
                f"Invalid finishing position: {position}"
            )
        if score != 3 - position:
            raise RuntimeError(
                f"Inconsistent result: position={position}, score={score}"
            )

        if pending_state is not None:
            agent.remember(
                state=pending_state,
                action=pending_action,
                reward=score / 3.0,
                next_state=pending_state,
                done=True,
                discount=0.0,
            )
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        completed_matches += 1
        config_name, _ = CONFIGURATIONS[
            (completed_matches - 1) % len(CONFIGURATIONS)
        ]

        records.append({
            "episode": completed_matches,
            "configuration": config_name,
            "agent1_position": position,
            "agent1_match_score": score,
            "win": int(position == 0),
            "epsilon": agent.epsilon,
            "steps": total_steps,
            "latest_loss": losses[-1] if losses else float("nan"),
        })

        agent.decay_epsilon()
        if completed_matches % TARGET_UPDATE_MATCHES == 0:
            agent.update_target_network()

        print(
            f"Match {completed_matches:3d}/{TOTAL_MATCHES} | "
            f"{config_name:14s} | Score: {score} | "
            f"Position: {position} | Epsilon: {agent.epsilon:.3f}"
        )

        pending_state = None
        pending_action = None
        pending_duration = 0

        # Do not carry opponent observations into the next match.
        opponent_model.reset()
        original_start_match()

    game.start_match_handle_cards = finish_previous_match

    try:
        while completed_matches < TOTAL_MATCHES and total_steps < MAX_STEPS:
            player_name = game.players[int(game.currentPlayer)].name
            config_index = completed_matches % len(CONFIGURATIONS)
            _, opponent_policies = CONFIGURATIONS[config_index]

            if player_name == "Agent1":
                current_state = augment_observation(
                    observation, opponent_model
                )

                if pending_state is not None:
                    agent.remember(
                        state=pending_state,
                        action=pending_action,
                        reward=0.0,
                        next_state=current_state,
                        done=False,
                        discount=agent.gamma ** pending_duration,
                    )
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(loss)

                action_index = agent.select_action(current_state)
                pending_state = current_state.copy()
                pending_action = action_index
                pending_duration = 0
                model_for_action = None

            else:
                action_index = select_opponent_action(
                    observation,
                    opponent_policies[player_name],
                )
                # Keep the current match's model reference in case this
                # environment step triggers a match reset.
                model_for_action = opponent_model

            observation, reward, terminated, truncated, info = env.step(
                action_to_vector(action_index)
            )
            total_steps += 1

            if player_name != "Agent1" and info.get("Action_Valid", True):
                model_for_action.update(player_name, action_index)

            if pending_state is not None:
                pending_duration += 1

        if completed_matches < TOTAL_MATCHES:
            raise RuntimeError(
                f"Stopped after {total_steps} steps with "
                f"{completed_matches}/{TOTAL_MATCHES} matches captured. "
                "Check MAX_STEPS and environment termination."
            )

        if len(records) != TOTAL_MATCHES:
            raise RuntimeError(
                f"Expected {TOTAL_MATCHES} results, captured {len(records)}."
            )

        with LOG_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        df = pd.DataFrame(records)
        summary = (
            df.groupby("configuration", sort=False)
            .agg(
                matches=("win", "count"),
                wins=("win", "sum"),
                win_rate=("win", "mean"),
                average_score=("agent1_match_score", "mean"),
                average_position=("agent1_position", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(SUMMARY_PATH, index=False)

        df["rolling_win_rate"] = df["win"].rolling(
            window=20, min_periods=1
        ).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(
            df["episode"],
            df["rolling_win_rate"],
            label="Opponent-aware DQN rolling win rate",
        )
        plt.xlabel("Completed training matches")
        plt.ylabel("Rolling win rate (last 20 matches)")
        plt.title("Opponent-Aware DQN Training Performance")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(CURVE_PATH, dpi=300)
        plt.close()

        agent.save(MODEL_PATH)

        print("\nOpponent-aware training complete")
        print("Matches captured:", len(records))
        print("Environment steps:", total_steps)
        print("Overall win rate:", round(float(df["win"].mean()), 3))
        print("\nOpponent configuration summary:")
        print(summary.to_string(index=False))
        print("\nSaved files:")
        print(LOG_PATH)
        print(SUMMARY_PATH)
        print(CURVE_PATH)
        print(MODEL_PATH)

    finally:
        game.start_match_handle_cards = original_start_match
        env.close()


if __name__ == "__main__":
    main()
