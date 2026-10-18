
import numpy as np


def random_legal_action(observation, rng=None):
    """Choose uniformly from the legal actions in an observation."""
    if rng is None:
        rng = np.random.default_rng()

    action_mask = observation[28:228]
    legal_actions = np.flatnonzero(action_mask == 1)

    if len(legal_actions) == 0:
        raise ValueError("No legal actions are available.")

    return int(rng.choice(legal_actions))


def action_to_vector(action_index):
    """Convert an action index into the environment's 200-value vector."""
    if not 0 <= action_index < 200:
        raise ValueError("Action index must be between 0 and 199.")

    action = np.zeros(200, dtype=np.float32)
    action[action_index] = 1.0

    return action

def rule_based_action(observation):
    """Prefer legal card-playing actions; pass only when necessary."""
    action_mask = observation[28:228]
    legal_actions = np.flatnonzero(action_mask == 1)

    if len(legal_actions) == 0:
        raise ValueError("No legal actions are available.")

    pass_action = 199
    playable_actions = legal_actions[legal_actions != pass_action]

    if len(playable_actions) > 0:
        return int(playable_actions[0])

    return pass_action

def select_opponent_action(observation, policy, rng=None):
    """Select an action for a baseline opponent."""
    if policy == "random":
        return random_legal_action(observation, rng)

    if policy == "rule_based":
        return rule_based_action(observation)

    raise ValueError(
        f"Unknown opponent policy: {policy}"
    )
