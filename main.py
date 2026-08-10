import argparse
import sys
import os

from rl_core.environment import MazeEnv
from rl_core.trainer import Trainer
from rl_core.utils import plot_comparison_results

def run_cli_benchmark(preset_name: str = "medium", episodes: int = 500):
    print("=" * 70)
    print("      INTELLIGENT MAZE SOLVER: REINFORCEMENT LEARNING BENCHMARK      ")
    print("=" * 70)
    print(f"[*] Loading Preset Maze: '{preset_name}'...")
    env = MazeEnv.get_preset_maze(preset_name)
    print(f"[*] Grid Dimensions: {env.rows}x{env.cols}")
    print(f"[*] Start Position: {env.start_pos} | Goal Position: {env.goal_pos}")
    print(f"[*] Training Episodes: {episodes}")
    print("-" * 70)

    print("[+] Executing Head-to-Head Comparison: Q-Learning (Off-Policy) vs SARSA (On-Policy)...")
    results = Trainer.compare(env, episodes=episodes)

    summary = results["summary"]
    q_sum = summary["q_learning"]
    sarsa_sum = summary["sarsa"]

    print("\n" + "=" * 70)
    print("                         BENCHMARK RESULTS                           ")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Q-Learning':<15} | {'SARSA':<15}")
    print("-" * 70)
    print(f"{'Avg Steps (Last 50 Ep)':<30} | {q_sum['avg_steps_last50']:<15} | {sarsa_sum['avg_steps_last50']:<15}")
    print(f"{'Avg Reward (Last 50 Ep)':<30} | {q_sum['avg_reward_last50']:<15} | {sarsa_sum['avg_reward_last50']:<15}")
    print(f"{'Success Rate (Last 50 Ep)':<30} | {str(q_sum['success_rate']) + '%':<15} | {str(sarsa_sum['success_rate']) + '%':<15}")
    print(f"{'Optimal Path Steps':<30} | {q_sum['eval_path_length'] - 1:<15} | {sarsa_sum['eval_path_length'] - 1:<15}")
    print(f"{'Execution Time (ms)':<30} | {q_sum['time_ms']:<15} | {sarsa_sum['time_ms']:<15}")
    print("=" * 70)

    # Plot results
    output_png = f"results_comparison_{preset_name}.png"
    save_path = plot_comparison_results(results, save_path=output_png)
    print(f"\n[+] High-resolution comparison plot saved to: {os.path.abspath(save_path)}")

def run_web_server(host: str = "127.0.0.1", port: int = 8000):
    from backend.app import run_server
    run_server(host=host, port=port)

def main():
    parser = argparse.ArgumentParser(description="Intelligent Maze Solver Using Reinforcement Learning (Q-Learning & SARSA)")
    parser.add_argument("--web", action="store_true", help="Launch interactive Web Studio dashboard")
    parser.add_argument("--cli", action="store_true", help="Run terminal benchmark comparison")
    parser.add_argument("--preset", type=str, default="medium", choices=["easy", "medium", "hard", "cliff_walker"], help="Preset maze for CLI mode")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--port", type=int, default=8000, help="Port for web server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for web server")

    args = parser.parse_args()

    # Default to web mode if neither flag specified
    if not args.cli:
        run_web_server(host=args.host, port=args.port)
    else:
        run_cli_benchmark(preset_name=args.preset, episodes=args.episodes)

if __name__ == "__main__":
    main()
