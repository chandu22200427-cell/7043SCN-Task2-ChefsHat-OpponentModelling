
import numpy as np

from environment_test import create_environment
from agents import random_legal_action, action_to_vector

env, observation, _ = create_environment(seed=42)
rng = np.random.default_rng(42)

try:
    game = env.unwrapped
    steps = 0

    while steps < 2000:
        action_index = random_legal_action(observation, rng)

        observation, reward, terminated, truncated, info = env.step(
            action_to_vector(action_index)
        )
        steps += 1

        if terminated:
            print("Match ended after steps:", steps)
            print("\nInformation keys:")
            print(list(info.keys()))

            print("\nMatch score from info:")
            print(info.get("Match_Score"))

            print("\nAccumulated score from info:")
            print(info.get("Game_Score"))

            print("\nFinished players from info:")
            print(info.get("Finished_Players"))

            print("\nPlayer state immediately after step:")
            for player in game.players:
                print(
                    player.name,
                    "position =", player.finishing_position,
                    "current score =", player.current_score,
                    "total score =", player.acumulated_score,
                )
            break

finally:
    env.close()
