
import numpy as np

from environment_check import create_fixed_match_environment
from agents import random_legal_action, action_to_vector


env, observation, _ = create_fixed_match_environment(
    number_of_matches=5,
    seed=42,
)

rng = np.random.default_rng(42)
steps = 0
max_steps = 5000

try:
    game = env.unwrapped

    while steps < max_steps and not game.isGameOver():
        action_index = random_legal_action(observation, rng)

        observation, reward, terminated, truncated, info = env.step(
            action_to_vector(action_index)
        )

        steps += 1

        if terminated or truncated:
            print(
                f"Match ended: {game.matches}; "
                f"Experiment over: {game.isGameOver()}"
            )

        # Do not call reset() here.
        # Chef's Hat starts the next match automatically.

    print("\nSteps taken:", steps)
    print("Match counter:", game.matches)
    print("Experiment over:", game.isGameOver())
    print("Accumulated scores:", game.get_acumulated_score())

finally:
    env.close()
