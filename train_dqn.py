
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from environment_check import create_fixed_match_environment
from agents import action_to_vector, select_opponent_action
from dqn_agent import DQNAgent


SEED = 42
TOTAL_MATCHES = 100
MAX_STEPS = 100_000
TARGET_UPDATE_MATCHES = 10

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = RESULTS_DIR / "dqn_training_log.csv"
SUMMARY_PATH = RESULTS_DIR / "dqn_training_summary.csv"
CURVE_PATH = RESULTS_DIR / "dqn_training_curve.png"
MODEL_PATH = MODELS_DIR / "dqn_opponent_modeling.pt"


CONFIGURATIONS = [
    (
        "all_random",
        {
            "Agent2": "random",
            "Agent3": "random",
            "Agent4": "random",
        },
    ),
    (
        "mixed",
        {
            "Agent2": "random",
            "Agent3": "rule_based",
            "Agent4": "rule_based",
        },
    ),
    (
        "all_rule_based",
        {
            "Agent2": "rule_based",
            "Agent3": "rule_based",
            "Agent4": "rule_based",
        },
    ),
]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    agent = DQNAgent(seed=SEED)

    # Chef's Hat Gym increments its match counter when a match starts.
    # One additional match allows the final requested result to be
    # captured before the environment resets the players.
    env, observation, _ = create_fixed_match_environment(
        number_of_matches=TOTAL_MATCHES + 1,
        seed=SEED,
    )

    game = env.unwrapped

    completed_matches = 0
    total_steps = 0
    records = []
    losses = []

    # Tracks Agent1's most recent decision within the current match.
    pending_state = None
    pending_action = None
    pending_duration = 0

    original_start_match = game.start_match_handle_cards

    def finish_previous_match():
        """Record the previous match before player state is reset."""

        nonlocal completed_matches
        nonlocal pending_state, pending_action, pending_duration

        # Normally the callback is invoked when a new match starts,
        # after the previous match has ended.
        if completed_matches >= TOTAL_MATCHES:
            original_start_match()
            return

        player = next(
            p for p in game.players
            if p.name == "Agent1"
        )

        position = int(player.finishing_position)
        score = int(player.current_score)

        if position not in (0, 1, 2, 3):
            raise RuntimeError(
                "Invalid finishing position captured before reset: "
                f"{position}"
            )

        if score != 3 - position:
            raise RuntimeError(
                f"Inconsistent match result: position={position}, "
                f"score={score}"
            )

        # Assign the terminal reward to Agent1's last decision.
        # Use the same state as next_state because this transition
        # is terminal and must not bootstrap.
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

        record = {
            "episode": completed_matches,
            "configuration": config_name,
            "agent1_position": position,
            "agent1_match_score": score,
            "win": int(position == 0),
            "epsilon": agent.epsilon,
            "steps": total_steps,
            "latest_loss": (
                losses[-1] if losses else float("nan")
            ),
        }

        records.append(record)

        # Reduce exploration after the completed match.
        agent.decay_epsilon()

        if completed_matches % TARGET_UPDATE_MATCHES == 0:
            agent.update_target_network()

        print(
            f"Match {completed_matches:3d}/{TOTAL_MATCHES} | "
            f"{config_name:14s} | "
            f"Score: {score} | "
            f"Position: {position} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

        # Never carry a pending action into another match.
        pending_state = None
        pending_action = None
        pending_duration = 0

        # Allow the original environment method to deal the next match.
        original_start_match()

    game.start_match_handle_cards = finish_previous_match

    try:
        while (
            completed_matches < TOTAL_MATCHES
            and total_steps < MAX_STEPS
        ):
            player_name = game.players[
                game.currentPlayer
            ].name

            config_index = (
                completed_matches % len(CONFIGURATIONS)
            )

            _, opponent_policies = CONFIGURATIONS[
                config_index
            ]

            if player_name == "Agent1":

                # Complete the previous Agent1 transition.
                # Its next state is the observation at Agent1's
                # next decision, not the state after an opponent's
                # individual action.
                if pending_state is not None:
                    transition_discount = (
                        agent.gamma ** pending_duration
                    )

                    agent.remember(
                        state=pending_state,
                        action=pending_action,
                        reward=0.0,
                        next_state=np.asarray(
                            observation,
                            dtype=np.float32,
                        ),
                        done=False,
                        discount=transition_discount,
                    )

                    loss = agent.train_step()

                    if loss is not None:
                        losses.append(loss)

                action_index = agent.select_action(
                    observation
                )

                pending_state = np.asarray(
                    observation,
                    dtype=np.float32,
                ).copy()

                pending_action = action_index
                pending_duration = 0

            else:
                action_index = select_opponent_action(
                    observation,
                    opponent_policies[player_name],
                )

            observation, reward, terminated, truncated, info = (
                env.step(action_to_vector(action_index))
            )

            total_steps += 1

            # Count environment steps since Agent1's pending action.
            # If a match ends, the callback clears the pending action
            # before control returns here.
            if pending_state is not None:
                pending_duration += 1

        if completed_matches < TOTAL_MATCHES:
            raise RuntimeError(
                f"Training stopped after {total_steps} steps, "
                f"with only {completed_matches}/{TOTAL_MATCHES} "
                "matches captured. Check MAX_STEPS and the "
                "environment's match termination behaviour."
            )

        if len(records) != TOTAL_MATCHES:
            raise RuntimeError(
                f"Expected {TOTAL_MATCHES} results but captured "
                f"{len(records)}."
            )

        # Save match-level training history.
        with LOG_PATH.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(records[0].keys()),
            )
            writer.writeheader()
            writer.writerows(records)

        df = pd.DataFrame(records)

        summary = (
            df.groupby("configuration", sort=False)
            .agg(
                matches=("win", "count"),
                wins=("win", "sum"),
                win_rate=("win", "mean"),
                average_score=(
                    "agent1_match_score", "mean"
                ),
                average_position=(
                    "agent1_position", "mean"
                ),
            )
            .reset_index()
        )

        summary.to_csv(
            SUMMARY_PATH,
            index=False,
        )

        # Rolling win rate over the latest 20 completed matches.
        df["rolling_win_rate"] = (
            df["win"].rolling(
                window=20,
                min_periods=1,
            ).mean()
        )

        plt.figure(figsize=(10, 5))

        plt.plot(
            df["episode"],
            df["rolling_win_rate"],
            label="DQN rolling win rate",
        )

        plt.xlabel("Completed training matches")
        plt.ylabel("Rolling win rate (last 20 matches)")
        plt.title("DQN Training Performance")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            CURVE_PATH,
            dpi=300,
        )
        plt.close()

        agent.save(MODEL_PATH)

        print("\nTraining complete")
        print("Matches captured:", len(records))
        print("Environment steps:", total_steps)
        print(
            "Overall win rate:",
            round(float(df["win"].mean()), 3),
        )

        print("\nOpponent summary:")
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
