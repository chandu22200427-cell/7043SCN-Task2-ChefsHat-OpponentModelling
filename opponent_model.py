
class OpponentModel:
    """Track observed pass/play tendencies for each opponent."""

    PASS_ACTION = 199

    def __init__(self, opponent_names):
        self.opponent_names = tuple(opponent_names)
        self.reset()

    def reset(self):
        """Reset opponent statistics at the start of a new match."""
        self.stats = {
            name: {
                "observed_actions": 0,
                "passes": 0,
                "plays": 0,
            }
            for name in self.opponent_names
        }

    def update(self, opponent_name, action_index):
        """Record an opponent's submitted action."""
        if opponent_name not in self.stats:
            raise ValueError(f"Unknown opponent: {opponent_name}")

        profile = self.stats[opponent_name]
        profile["observed_actions"] += 1

        if int(action_index) == self.PASS_ACTION:
            profile["passes"] += 1
        else:
            profile["plays"] += 1

    def get_profile(self, opponent_name):
        """Return smoothed pass/play estimates."""
        profile = self.stats[opponent_name]
        total = profile["observed_actions"]

        return {
            "observed_actions": total,
            "pass_rate": (profile["passes"] + 1) / (total + 2),
            "play_rate": (profile["plays"] + 1) / (total + 2),
        }

    def get_state_features(self, opponent_name):
        profile = self.get_profile(opponent_name)
        return [profile["pass_rate"], profile["play_rate"]]
