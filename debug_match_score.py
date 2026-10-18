
from environment_check import create_fixed_match_environment
from agents import random_legal_action, action_to_vector
import numpy as np
import random

random.seed(42)
np.random.seed(42)

env, observation, _ = create_fixed_match_environment(
    number_of_matches=2,
    seed=42,
)

game = env.unwrapped
rng = np.random.default_rng(42)

try:
    for step in range(20000):
        current_player = game.currentPlayer
        player_name = game.players[current_player].name

        action_index = random_legal_action(observation, rng)

        observation, reward, terminated, truncated, info = env.step(
            action_to_vector(action_index)
        )

        if terminated:
            print("\nMATCH TERMINATED")
            print("Step:", step + 1)
            print("Info Match_Score:", info.get("Match_Score"))
            print("Info Game_Score:", info.get("Game_Score"))
            print("Info Player_Names:", info.get("Player_Names"))

            print("\nPlayer states:")
            for player in game.players:
                print(
                    player.name,
                    "| position:", player.finishing_position,
                    "| current score:", player.current_score,
                    "| accumulated score:", player.acumulated_score,
                )
            break
    else:
        print("No completed match within 20,000 steps.")

finally:
    env.close()
