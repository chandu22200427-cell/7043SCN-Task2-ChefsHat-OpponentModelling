
import numpy as np

from environment_test import create_environment
from agents import rule_based_action, action_to_vector

env, observation, _ = create_environment(seed=42)

try:
    action_index = rule_based_action(observation)
    action_vector = action_to_vector(action_index)

    # Check that the selected action is legal.
    legal = observation[28:228]
    assert legal[action_index] == 1, "Selected action is not legal!"

    next_observation, reward, terminated, truncated, info = env.step(
        action_vector
    )

    print("Rule-based action test passed!")
    print("Selected action index:", action_index)
    print("Action valid according to environment:", info.get("Action_Valid"))
    print("Next observation shape:", next_observation.shape)
    print("Reward:", reward)

finally:
    env.close()
