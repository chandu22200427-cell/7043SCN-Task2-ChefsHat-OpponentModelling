
import numpy as np

from environment_test import create_environment
from agents import random_legal_action, action_to_vector


env, observation, _ = create_environment(seed=42)
rng = np.random.default_rng(42)

try:
    done = False
    steps = 0

    while not done and steps < 5000:
        action_index = random_legal_action(observation, rng)

        observation, reward, terminated, truncated, info = env.step(
            action_to_vector(action_index)
        )

        done = terminated or truncated
        steps += 1

    game = env.unwrapped

    print("Steps:", steps)
    print("Match completed:", done)
    print("Final scores:", game.get_current_score())
    print("\nFinishing positions:")

    for player in game.players:
        print(
            player.name,
            "| position:", player.finishing_position,
            "| finished:", player.finished,
            "| score:", player.current_score,
        )

finally:
    env.close()
