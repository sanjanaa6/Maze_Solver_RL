import unittest
import numpy as np
from rl_core.environment import MazeEnv
from rl_core.agents import QLearningAgent, SARSAAgent
from rl_core.trainer import Trainer

class TestMazeRL(unittest.TestCase):

    def setUp(self):
        self.preset = MazeEnv.get_preset_maze("easy")

    def test_environment_step(self):
        env = MazeEnv(grid=[[2, 0], [1, 3]])
        state = env.reset()
        self.assertEqual(state, (0, 0))

        # Move right (action 1) -> (0, 1)
        next_s, reward, done, info = env.step(1)
        self.assertEqual(next_s, (0, 1))
        self.assertFalse(done)

        # Move down (action 2) -> (1, 1) Goal
        next_s, reward, done, info = env.step(2)
        self.assertEqual(next_s, (1, 1))
        self.assertTrue(done)
        self.assertTrue(info["reached_goal"])

    def test_wall_collision(self):
        env = MazeEnv(grid=[[2, 1], [0, 3]])
        env.reset()
        # Try moving right into wall (action 1)
        next_s, reward, done, info = env.step(1)
        self.assertEqual(next_s, (0, 0))
        self.assertTrue(info["hit_wall"])
        self.assertEqual(reward, env.wall_penalty)

    def test_q_learning_solver(self):
        env = MazeEnv.get_preset_maze("easy")
        agent = QLearningAgent(rows=env.rows, cols=env.cols, alpha=0.2, gamma=0.99, epsilon_decay=0.95)
        metrics = Trainer.train_agent(env, agent, episodes=200)
        eval_res = metrics["eval"]

        self.assertTrue(eval_res["success"], "Q-Learning agent failed to find path on easy maze")
        self.assertGreater(len(eval_res["path"]), 1)

    def test_sarsa_solver(self):
        env = MazeEnv.get_preset_maze("easy")
        agent = SARSAAgent(rows=env.rows, cols=env.cols, alpha=0.2, gamma=0.99, epsilon_decay=0.95)
        metrics = Trainer.train_agent(env, agent, episodes=200)
        eval_res = metrics["eval"]

        self.assertTrue(eval_res["success"], "SARSA agent failed to find path on easy maze")

    def test_comparison_runner(self):
        env = MazeEnv.get_preset_maze("easy")
        comp = Trainer.compare(env, episodes=50)
        self.assertIn("q_learning", comp)
        self.assertIn("sarsa", comp)
        self.assertIn("summary", comp)

if __name__ == "__main__":
    unittest.main()
