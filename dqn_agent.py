
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Estimate Q-values for the 200 possible actions."""

    def __init__(self, state_size=228, action_size=200):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:
    """DQN agent with replay memory and legal-action masking."""

    def __init__(
        self,
        state_size=228,
        action_size=200,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        replay_capacity=20000,
        seed=42,
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # CPU is used for compatibility with your current installation.
        self.device = torch.device("cpu")

        self.policy_net = QNetwork(
            state_size, action_size
        ).to(self.device)

        self.target_net = QNetwork(
            state_size, action_size
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
        )

        self.loss_function = nn.SmoothL1Loss()

        # Each entry contains:
        # state, action, reward, next_state, done, discount
        self.memory = deque(maxlen=replay_capacity)

    def select_action(self, observation):
        """Select an epsilon-greedy action from legal actions."""

        observation = np.asarray(
            observation, dtype=np.float32
        )

        if observation.shape != (self.state_size,):
            raise ValueError(
                f"Expected observation shape "
                f"({self.state_size},), got {observation.shape}"
            )

        legal_actions = np.flatnonzero(
            observation[28:228] == 1
        )

        if len(legal_actions) == 0:
            raise ValueError("No legal actions are available.")

        # Exploration: randomly choose a legal action.
        if random.random() < self.epsilon:
            return int(
                random.choice(legal_actions.tolist())
            )

        state = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = (
                self.policy_net(state)[0].cpu().numpy()
            )

        # Never select an illegal action.
        legal_q_values = q_values[legal_actions]

        return int(
            legal_actions[np.argmax(legal_q_values)]
        )

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        discount=None,
    ):
        """Store a transition, optionally with its own discount."""

        if discount is None:
            discount = self.gamma

        state = np.asarray(
            state, dtype=np.float32
        ).copy()

        next_state = np.asarray(
            next_state, dtype=np.float32
        ).copy()

        if state.shape != (self.state_size,):
            raise ValueError("Invalid state shape.")

        if next_state.shape != (self.state_size,):
            raise ValueError("Invalid next_state shape.")

        if not 0 <= int(action) < self.action_size:
            raise ValueError("Action index is out of range.")

        self.memory.append(
            (
                state,
                int(action),
                float(reward),
                next_state,
                bool(done),
                float(discount),
            )
        )

    def train_step(self):
        """Perform one DQN update using a replay batch."""

        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(
            self.memory, self.batch_size
        )

        states = torch.as_tensor(
            np.array([item[0] for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            [item[1] for item in batch],
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.as_tensor(
            [item[2] for item in batch],
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            np.array([item[3] for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            [item[4] for item in batch],
            dtype=torch.float32,
            device=self.device,
        )

        discounts = torch.as_tensor(
            [item[5] for item in batch],
            dtype=torch.float32,
            device=self.device,
        )

        # Q(s, a): estimate for the selected action.
        current_q = self.policy_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)

            # The legal-action mask occupies indices 28:228.
            next_masks = next_states[:, 28:228] == 1

            next_q_values = next_q_values.masked_fill(
                ~next_masks, -1e9
            )

            max_next_q = next_q_values.max(
                dim=1
            ).values

            # Terminal transitions do not bootstrap.
            targets = (
                rewards
                + discounts
                * max_next_q
                * (1.0 - dones)
            )

        loss = self.loss_function(
            current_q, targets
        )

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 1.0
        )

        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        """Synchronize the target network with the policy network."""

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    def decay_epsilon(self):
        """Reduce exploration after each completed match."""

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path):
        """Save model weights and training state."""

        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "state_size": self.state_size,
                "action_size": self.action_size,
                "gamma": self.gamma,
            },
            path,
        )

    def load(self, path):
        """Restore a saved checkpoint."""

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.policy_net.load_state_dict(
            checkpoint["policy_net"]
        )

        self.target_net.load_state_dict(
            checkpoint["target_net"]
        )

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        self.epsilon = float(
            checkpoint.get("epsilon", self.epsilon)
        )
