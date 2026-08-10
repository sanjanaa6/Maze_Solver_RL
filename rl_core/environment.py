import numpy as np
import random
from typing import Tuple, List, Dict, Any, Optional

class MazeEnv:
    """
    GridWorld Maze Environment for Reinforcement Learning (Q-Learning & SARSA).
    
    Grid Cell Types:
    0: Empty / Path
    1: Wall / Obstacle
    2: Start (S)
    3: Goal (G)
    4: Trap (T)
    """

    # Actions: 0=Up, 1=Right, 2=Down, 3=Left
    ACTIONS = [0, 1, 2, 3]
    ACTION_NAMES = ["Up", "Right", "Down", "Left"]
    ACTION_VECTORS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(
        self,
        grid: Optional[List[List[int]]] = None,
        rows: int = 8,
        cols: int = 8,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        step_reward: float = -1.0,
        wall_penalty: float = -5.0,
        goal_reward: float = 100.0,
        trap_penalty: float = -20.0,
        max_steps: Optional[int] = None
    ):
        if grid is not None:
            self.grid = np.array(grid, dtype=int)
            self.rows, self.cols = self.grid.shape
        else:
            self.rows = rows
            self.cols = cols
            self.grid = np.zeros((rows, cols), dtype=int)

        # Detect or set Start and Goal
        if start_pos:
            self.start_pos = start_pos
        else:
            starts = np.argwhere(self.grid == 2)
            if len(starts) > 0:
                self.start_pos = (int(starts[0][0]), int(starts[0][1]))
            else:
                self.start_pos = (0, 0)

        if goal_pos:
            self.goal_pos = goal_pos
        else:
            goals = np.argwhere(self.grid == 3)
            if len(goals) > 0:
                self.goal_pos = (int(goals[0][0]), int(goals[0][1]))
            else:
                self.goal_pos = (self.rows - 1, self.cols - 1)

        # Ensure Start & Goal marked on grid
        self.grid[self.start_pos] = 2
        self.grid[self.goal_pos] = 3

        # Rewards configuration
        self.step_reward = step_reward
        self.wall_penalty = wall_penalty
        self.goal_reward = goal_reward
        self.trap_penalty = trap_penalty

        self.max_steps = max_steps or (self.rows * self.cols * 4)
        self.agent_pos = self.start_pos
        self.steps_taken = 0

    def reset(self, start_pos: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """Resets the environment and returns initial state."""
        if start_pos and self.is_valid_cell(start_pos[0], start_pos[1]) and self.grid[start_pos] != 1:
            self.agent_pos = start_pos
        else:
            self.agent_pos = self.start_pos

        self.steps_taken = 0
        return self.agent_pos

    def is_valid_cell(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        """
        Executes one step in the environment.
        Returns: (next_state, reward, done, info)
        """
        self.steps_taken += 1
        dr, dc = self.ACTION_VECTORS[action]
        next_r = self.agent_pos[0] + dr
        next_c = self.agent_pos[1] + dc

        info = {"hit_wall": False, "reached_goal": False, "hit_trap": False}

        # Check wall collision or out of bounds
        if not self.is_valid_cell(next_r, next_c) or self.grid[next_r, next_c] == 1:
            next_state = self.agent_pos  # Stay in place
            reward = self.wall_penalty
            info["hit_wall"] = True
            done = False
        else:
            next_state = (next_r, next_c)
            cell_type = self.grid[next_r, next_c]

            if cell_type == 3:  # Goal reached
                reward = self.goal_reward
                done = True
                info["reached_goal"] = True
            elif cell_type == 4:  # Trap hit
                reward = self.trap_penalty
                done = False
                info["hit_trap"] = True
            else:  # Normal path or start cell
                reward = self.step_reward
                done = False

        # Timeout check
        if self.steps_taken >= self.max_steps and not done:
            done = True
            info["timeout"] = True

        self.agent_pos = next_state
        return next_state, reward, done, info

    def state_to_idx(self, state: Tuple[int, int]) -> int:
        return state[0] * self.cols + state[1]

    def idx_to_state(self, idx: int) -> Tuple[int, int]:
        return (idx // self.cols, idx % self.cols)

    def get_grid_state(self) -> Dict[str, Any]:
        """Returns clean dictionary representation of maze state."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "grid": self.grid.tolist(),
            "start": list(self.start_pos),
            "goal": list(self.goal_pos)
        }

    @staticmethod
    def generate_random_solvable_maze(
        rows: int = 8,
        cols: int = 8,
        wall_density: float = 0.25,
        trap_density: float = 0.05
    ) -> "MazeEnv":
        """
        Generates a guaranteed solvable random maze using Randomized DFS + optional random obstacles.
        """
        grid = np.ones((rows, cols), dtype=int)
        
        # Carve paths using DFS
        def carve_passages_from(r, c):
            grid[r, c] = 0
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)
            
            for dr, dc in directions:
                nr, nc = r + dr * 2, c + dc * 2
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1:
                    grid[r + dr, c + dc] = 0
                    carve_passages_from(nr, nc)

        carve_passages_from(0, 0)

        # Set Start and Goal
        start = (0, 0)
        goal = (rows - 1, cols - 1)
        grid[start] = 2
        grid[goal] = 3

        # Add optional random walls or traps on existing paths (without blocking solution)
        empty_cells = list(zip(*np.where(grid == 0)))
        random.shuffle(empty_cells)

        num_traps = int(len(empty_cells) * trap_density)
        for i in range(min(num_traps, len(empty_cells))):
            r, c = empty_cells.pop()
            if (r, c) != start and (r, c) != goal:
                grid[r, c] = 4

        return MazeEnv(grid=grid.tolist(), start_pos=start, goal_pos=goal)

    @staticmethod
    def get_preset_maze(name: str) -> "MazeEnv":
        """Returns preset maze layouts."""
        presets = {
            "easy": [
                [2, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 1, 3]
            ],
            "medium": [
                [2, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 0, 1, 0],
                [0, 0, 0, 4, 1, 0, 0, 0],
                [1, 1, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 3]
            ],
            "cliff_walker": [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 4, 4, 4, 4, 4, 4, 4, 4, 3]
            ],
            "hard": [
                [2, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 4, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 1, 1, 1, 0, 3]
            ]
        }
        
        grid_data = presets.get(name.lower(), presets["medium"])
        return MazeEnv(grid=grid_data)
