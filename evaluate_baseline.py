
import csv
from pathlib import Path

import numpy as np

from environment_test import create_environment
from agents import random_legal_action, action_to_vector


def run_one_game(seed, max_steps=5000):
    env, observation, _ = create_environment(seed=seed)
    rng = np.random.default_rng(seed)

    try:
        done = False
        steps = 0

        while not done and steps < max_steps:
            action_index = random_legal_action(observation, rng)

            observation, reward, terminated, truncated, info = env.step(
                action_to_vector(action_index)
            )

            done = terminated or truncated
            steps += 1

        game = env.unwrapped

        results = {
            "seed": seed,
            "steps": steps,
            "completed": done,
        }

        for player in game.players:
            results[player.name + "_position"] = player.finishing_position
            results[player.name + "_score"] = player.current_score

        return results

    finally:
        env.close()


if __name__ == "__main__":
    number_of_matches = 20
    results = []

    for seed in range(number_of_matches):
        result = run_one_game(seed)
        results.append(result)
        print(
            f"Match {seed + 1}/{number_of_matches}: "
            f"steps={result['steps']}, completed={result['completed']}"
        )

    output_path = (
        Path(__file__).resolve().parent
        / "results"
        / "random_baseline.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    completed_results = [r for r in results if r["completed"]]

    print("\n--- Random baseline summary ---")
    print("Completed matches:", len(completed_results), "/", number_of_matches)

    for player in ["Agent1", "Agent2", "Agent3", "Agent4"]:
        scores = [r[player + "_score"] for r in completed_results]
        positions = [r[player + "_position"] for r in completed_results]

        win_rate = np.mean(np.array(positions) == 0) if positions else 0.0
        average_score = np.mean(scores) if scores else 0.0
        average_position = np.mean(positions) if positions else 0.0

        print(
            f"{player}: win_rate={win_rate:.1%}, "
            f"average_score={average_score:.2f}, "
            f"average_position={average_position:.2f}"
        )

    print("\nCSV saved to:", output_path)
