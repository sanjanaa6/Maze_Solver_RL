import numpy as np
import random
from typing import Tuple, Dict, List, Any, Optional

class RLAgent:
    """
    Base Reinforcement Learning Agent class for GridWorld Maze.
    Handles Q-Table initialization, Epsilon-Greedy policy, and decay schedules.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        num_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        self.rows = rows
        self.cols = cols
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: dict mapping (r, c) -> np.array of size num_actions initialized to zeros
        self.q_table: Dict[Tuple[int, int], np.ndarray] = {}
        for r in range(rows):
            for c in range(cols):
                self.q_table[(r, c)] = np.zeros(num_actions, dtype=float)

    def get_q(self, state: Tuple[int, int], action: Optional[int] = None) -> Any:
        """Returns Q-values for a state, or Q-value for a state-action pair."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions, dtype=float)
        if action is not None:
            return self.q_table[state][action]
        return self.q_table[state]

    def choose_action(self, state: Tuple[int, int], deterministic: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        If deterministic is True, acts completely greedily (eval mode).
        """
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        q_vals = self.get_q(state)
        max_v = np.max(q_vals)
        # Break ties randomly among actions sharing the max Q-value
        best_actions = np.where(q_vals == max_v)[0]
        return int(random.choice(best_actions))

    def decay_epsilon(self):
        """Decays exploration rate according to schedule."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self, state: Tuple[int, int], action: int, reward: float, next_state: Tuple[int, int], done: bool, next_action: Optional[int] = None):
        """Abstract update method implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement update method.")

    def get_policy(self) -> Dict[Tuple[int, int], int]:
        """Returns deterministic greedy policy for all states."""
        policy = {}
        for state in self.q_table:
            q_vals = self.q_table[state]
            policy[state] = int(np.argmax(q_vals))
        return policy

    def export_q_table_dict(self) -> Dict[str, List[float]]:
        """Converts Q-table to JSON-serializable dictionary format."""
        return {f"{r},{c}": self.q_table[(r, c)].tolist() for r, c in self.q_table}


class QLearningAgent(RLAgent):
    """
    Q-Learning (Off-Policy TD Control) Agent.
    Q(S, A) <- Q(S, A) + alpha * [Reward + gamma * max_a Q(S', a) - Q(S, A)]
    """

    def update(self, state: Tuple[int, int], action: int, reward: float, next_state: Tuple[int, int], done: bool, next_action: Optional[int] = None):
        current_q = self.get_q(state, action)
        
        if done:
            target = reward
        else:
            max_next_q = np.max(self.get_q(next_state))
            target = reward + self.gamma * max_next_q
            
        td_error = target - current_q
        self.q_table[state][action] += self.alpha * td_error


class SARSAAgent(RLAgent):
    """
    SARSA (On-Policy TD Control) Agent.
    Q(S, A) <- Q(S, A) + alpha * [Reward + gamma * Q(S', A') - Q(S, A)]
    """

    def update(self, state: Tuple[int, int], action: int, reward: float, next_state: Tuple[int, int], done: bool, next_action: Optional[int] = None):
        current_q = self.get_q(state, action)

        if done:
            target = reward
        else:
            assert next_action is not None, "SARSA requires next_action for update."
            next_q = self.get_q(next_state, next_action)
            target = reward + self.gamma * next_q

        td_error = target - current_q
        self.q_table[state][action] += self.alpha * td_error
