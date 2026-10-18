
import csv
import random
from pathlib import Path

import numpy as np

from environment_check import create_fixed_match_environment
from agents import (
    random_legal_action,
    action_to_vector,
    select_opponent_action,
)


def run_experiment(opponent_policies, seed, number_of_matches=20):
    """Evaluate Agent1 against different opponent policies."""

    random.seed(seed)
    np.random.seed(seed)

    env, observation, _ = create_fixed_match_environment(
        number_of_matches=number_of_matches + 1,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    game = env.unwrapped
    steps = 0
    max_steps = 20000
    records = []
    recorded_matches = set()

    def record_result(match_number, position, match_score, total_score, step_number):
        """Record a result only when its values are valid."""

        if match_number in recorded_matches:
            return

        if position not in (0, 1, 2, 3):
            return

        if match_score not in (0, 1, 2, 3):
            return

        if match_score != 3 - position:
            print(
                f"WARNING: Inconsistent result for match {match_number}: "
                f"position={position}, score={match_score}"
            )
            return

        records.append({
            "match": match_number,
            "seed": seed,
            "agent1_position": position,
            "agent1_match_score": match_score,
            "agent1_total_score": total_score,
            "steps_so_far": step_number,
        })
        recorded_matches.add(match_number)

    def capture_completed_match(match_number, step_number):
        """Capture player state before the next match resets it."""

        agent = next(
            player for player in game.players
            if player.name == "Agent1"
        )

        record_result(
            match_number=match_number,
            position=int(agent.finishing_position),
            match_score=int(agent.current_score),
            total_score=int(agent.acumulated_score),
            step_number=step_number,
        )

    original_start_match = game.start_match_handle_cards

    def capture_before_next_match():
        # The previous match has ended; player state is still available.
        completed_match_number = int(game.matches)
        capture_completed_match(completed_match_number, steps + 1)
        original_start_match()

    game.start_match_handle_cards = capture_before_next_match

    try:
        while steps < max_steps and not game.isGameOver():
            current_player = game.currentPlayer
            player_name = game.players[current_player].name

            if player_name == "Agent1":
                action_index = random_legal_action(observation, rng)
            else:
                policy = opponent_policies[player_name]
                action_index = select_opponent_action(
                    observation, policy, rng
                )

            observation, reward, terminated, truncated, info = env.step(
                action_to_vector(action_index)
            )
            steps += 1

            # The final match does not start another match. Try the
            # final player state first, then the returned score data.
            if terminated and game.isGameOver():
                final_match = int(game.matches)

                capture_completed_match(final_match, steps)

                if final_match not in recorded_matches:
                    player_names = info.get("Player_Names", [])
                    match_scores = info.get("Match_Score", [])
                    game_scores = info.get("Game_Score", [])

                    if (
                        "Agent1" in player_names
                        and len(match_scores) == len(player_names)
                        and len(game_scores) == len(player_names)
                    ):
                        agent_index = player_names.index("Agent1")
                        score = int(match_scores[agent_index])

                        if score in (0, 1, 2, 3):
                            record_result(
                                match_number=final_match,
                                position=3 - score,
                                match_score=score,
                                total_score=int(game_scores[agent_index]),
                                step_number=steps,
                            )

        records.sort(key=lambda record: record["match"])

        print(
            f"\nRequested matches: {number_of_matches}; "
            f"valid results captured: {len(records)}"
        )

        if len(records) != number_of_matches:
            missing = sorted(
                set(range(1, number_of_matches + 1))
                - recorded_matches
            )
            print("Missing match results:", missing)
            print("Environment match counter:", game.matches)
            print("Experiment over:", game.isGameOver())
        else:
            print(f"Successfully recorded all {number_of_matches} matches.")

        return records

    finally:
        game.start_match_handle_cards = original_start_match
        env.close()


if __name__ == "__main__":
    number_of_matches = 20
    seed = 42

    configurations = {
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

    all_records = []

    for config_name, policies in configurations.items():
        print(f"\n{'=' * 50}")
        print("Configuration:", config_name)
        print("=" * 50)

        records = run_experiment(
            opponent_policies=policies,
            seed=seed,
            number_of_matches=number_of_matches,
        )

        for record in records:
            record["configuration"] = config_name
            all_records.append(record)

        if not records:
            print("No valid match results were recorded.")
            continue

        positions = [r["agent1_position"] for r in records]
        scores = [r["agent1_match_score"] for r in records]
        wins = sum(position == 0 for position in positions)

        print("Matches recorded:", len(records))
        print(f"Agent1 wins: {wins}/{len(records)}")
        print("Agent1 win rate:", round(wins / len(records), 3))
        print("Average match score:", round(float(np.mean(scores)), 3))
        print(
            "Average finishing position:",
            round(float(np.mean(positions)), 3),
        )

    output_path = (
        Path(__file__).resolve().parent
        / "results"
        / "opponent_comparison.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "configuration",
        "match",
        "seed",
        "agent1_position",
        "agent1_match_score",
        "agent1_total_score",
        "steps_so_far",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print("\nResults saved to:", output_path)
    print("Total match records saved:", len(all_records))
