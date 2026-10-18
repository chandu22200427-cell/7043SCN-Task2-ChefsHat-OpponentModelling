
from environment_test import create_environment
from agents import random_legal_action, action_to_vector

env, observation, _ = create_environment(seed=42)

try:
    game = env.unwrapped

    print("Initial accumulated scores:")
    print(game.get_acumulated_score())

    print("\nInitial performance scores:")
    print(game.get_acumulated_performance_score())

    rng_seed = 42
    import numpy as np
    rng = np.random.default_rng(rng_seed)

    steps = 0
    max_steps = 1000

    while steps < max_steps:
        action_index = random_legal_action(observation, rng)

        observation, reward, terminated, truncated, info = env.step(
            action_to_vector(action_index)
        )

        steps += 1

        if terminated or truncated:
            print("\nMatch completed after", steps, "steps.")
            print("Accumulated scores:")
            print(game.get_acumulated_score())
            print("Accumulated performance scores:")
            print(game.get_acumulated_performance_score())
            break

finally:
    env.close()
