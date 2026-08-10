import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from rl_core.environment import MazeEnv
from rl_core.agents import RLAgent, QLearningAgent, SARSAAgent

class Trainer:
    """
    Orchestrates training, evaluation, and comparative benchmarking for RL agents.
    """

    @staticmethod
    def train_agent(
        env: MazeEnv,
        agent: RLAgent,
        episodes: int = 500,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Trains an RL agent (Q-Learning or SARSA) on the given environment.
        """
        metrics = {
            "episode_rewards": [],
            "episode_steps": [],
            "success_history": [],
            "epsilon_history": [],
            "training_time_ms": 0
        }

        start_time = time.time()
        is_sarsa = isinstance(agent, SARSAAgent)

        for ep in range(episodes):
            state = env.reset()
            total_reward = 0.0
            steps = 0
            done = False

            if is_sarsa:
                action = agent.choose_action(state)

            while not done:
                if not is_sarsa:
                    action = agent.choose_action(state)

                next_state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1

                if is_sarsa:
                    next_action = agent.choose_action(next_state) if not done else None
                    agent.update(state, action, reward, next_state, done, next_action)
                    state = next_state
                    action = next_action
                else:
                    agent.update(state, action, reward, next_state, done)
                    state = next_state

            agent.decay_epsilon()

            # Record metrics
            metrics["episode_rewards"].append(float(total_reward))
            metrics["episode_steps"].append(steps)
            metrics["success_history"].append(bool(info.get("reached_goal", False)))
            metrics["epsilon_history"].append(float(agent.epsilon))

        end_time = time.time()
        metrics["training_time_ms"] = round((end_time - start_time) * 1000, 2)

        # Evaluation post-training
        eval_res = Trainer.evaluate_agent(env, agent)
        metrics["eval"] = eval_res
        metrics["q_table"] = agent.export_q_table_dict()

        return metrics

    @staticmethod
    def evaluate_agent(env: MazeEnv, agent: RLAgent) -> Dict[str, Any]:
        """
        Runs one deterministic greedy episode (epsilon = 0) to evaluate learned path.
        """
        state = env.reset()
        path = [state]
        actions_taken = []
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state, deterministic=True)
            actions_taken.append(int(action))
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            path.append(next_state)
            state = next_state

            # Prevent infinite loop in evaluation if policy has a loop
            if steps >= env.max_steps:
                break

        reached_goal = bool(info.get("reached_goal", False))

        return {
            "path": [list(pos) for pos in path],
            "actions": actions_taken,
            "total_reward": float(total_reward),
            "steps": steps,
            "success": reached_goal
        }

    @staticmethod
    def compare(
        env: MazeEnv,
        episodes: int = 500,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_decay: float = 0.995
    ) -> Dict[str, Any]:
        """
        Runs identical training setup on Q-Learning and SARSA for head-to-head analysis.
        """
        # Q-Learning Agent
        q_agent = QLearningAgent(
            rows=env.rows,
            cols=env.cols,
            alpha=alpha,
            gamma=gamma,
            epsilon_decay=epsilon_decay
        )
        q_res = Trainer.train_agent(env, q_agent, episodes=episodes)

        # SARSA Agent
        sarsa_agent = SARSAAgent(
            rows=env.rows,
            cols=env.cols,
            alpha=alpha,
            gamma=gamma,
            epsilon_decay=epsilon_decay
        )
        sarsa_res = Trainer.train_agent(env, sarsa_agent, episodes=episodes)

        # Calculate summary comparison statistics over last 50 episodes
        window = min(50, episodes)
        
        q_avg_steps = float(np.mean(q_res["episode_steps"][-window:]))
        q_avg_reward = float(np.mean(q_res["episode_rewards"][-window:]))
        q_success_rate = float(np.mean(q_res["success_history"][-window:])) * 100

        sarsa_avg_steps = float(np.mean(sarsa_res["episode_steps"][-window:]))
        sarsa_avg_reward = float(np.mean(sarsa_res["episode_rewards"][-window:]))
        sarsa_success_rate = float(np.mean(sarsa_res["success_history"][-window:])) * 100

        return {
            "episodes": episodes,
            "q_learning": q_res,
            "sarsa": sarsa_res,
            "summary": {
                "q_learning": {
                    "avg_steps_last50": round(q_avg_steps, 2),
                    "avg_reward_last50": round(q_avg_reward, 2),
                    "success_rate": round(q_success_rate, 2),
                    "eval_path_length": len(q_res["eval"]["path"]),
                    "eval_success": q_res["eval"]["success"],
                    "time_ms": q_res["training_time_ms"]
                },
                "sarsa": {
                    "avg_steps_last50": round(sarsa_avg_steps, 2),
                    "avg_reward_last50": round(sarsa_avg_reward, 2),
                    "success_rate": round(sarsa_success_rate, 2),
                    "eval_path_length": len(sarsa_res["eval"]["path"]),
                    "eval_success": sarsa_res["eval"]["success"],
                    "time_ms": sarsa_res["training_time_ms"]
                }
            }
        }
