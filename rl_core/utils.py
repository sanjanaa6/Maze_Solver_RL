import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List

def plot_comparison_results(results: Dict[str, Any], save_path: str = "results_comparison.png"):
    """
    Generates and saves publication-quality comparison charts for Q-Learning vs SARSA.
    """
    episodes = results["episodes"]
    ep_axis = list(range(1, episodes + 1))
    
    q_rewards = results["q_learning"]["episode_rewards"]
    sarsa_rewards = results["sarsa"]["episode_rewards"]

    q_steps = results["q_learning"]["episode_steps"]
    sarsa_steps = results["sarsa"]["episode_steps"]

    # Moving average window
    window = max(1, episodes // 20)
    def moving_avg(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Intelligent Maze Solver: Q-Learning vs SARSA Performance Comparison", fontsize=16, fontweight='bold')

    # 1. Cumulative Reward per Episode
    ax1 = axes[0, 0]
    ax1.plot(ep_axis, q_rewards, alpha=0.3, color='#3B82F6', label='Q-Learning Raw')
    ax1.plot(ep_axis, sarsa_rewards, alpha=0.3, color='#10B981', label='SARSA Raw')
    
    if len(q_rewards) >= window:
        ma_ep = list(range(window, episodes + 1))
        ax1.plot(ma_ep, moving_avg(q_rewards, window), color='#1D4ED8', linewidth=2, label=f'Q-Learning ({window}-ep MA)')
        ax1.plot(ma_ep, moving_avg(sarsa_rewards, window), color='#047857', linewidth=2, label=f'SARSA ({window}-ep MA)')
    
    ax1.set_title("Cumulative Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 2. Steps to Goal per Episode
    ax2 = axes[0, 1]
    ax2.plot(ep_axis, q_steps, alpha=0.3, color='#3B82F6')
    ax2.plot(ep_axis, sarsa_steps, alpha=0.3, color='#10B981')

    if len(q_steps) >= window:
        ma_ep = list(range(window, episodes + 1))
        ax2.plot(ma_ep, moving_avg(q_steps, window), color='#1D4ED8', linewidth=2, label='Q-Learning MA')
        ax2.plot(ma_ep, moving_avg(sarsa_steps, window), color='#047857', linewidth=2, label='SARSA MA')

    ax2.set_title("Steps Taken per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 3. Success Rate (Rolling 50 episodes)
    ax3 = axes[1, 0]
    q_succ = results["q_learning"]["success_history"]
    sarsa_succ = results["sarsa"]["success_history"]
    
    s_win = min(50, episodes)
    q_roll_succ = [np.mean(q_succ[max(0, i - s_win):i+1]) * 100 for i in range(len(q_succ))]
    sarsa_roll_succ = [np.mean(sarsa_succ[max(0, i - s_win):i+1]) * 100 for i in range(len(sarsa_succ))]

    ax3.plot(ep_axis, q_roll_succ, color='#3B82F6', linewidth=2, label='Q-Learning Success %')
    ax3.plot(ep_axis, sarsa_roll_succ, color='#10B981', linewidth=2, label='SARSA Success %')
    ax3.set_title(f"Rolling Success Rate (Window={s_win})")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Success Rate (%)")
    ax3.set_ylim(-5, 105)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)

    # 4. Final Benchmark Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = results["summary"]
    table_data = [
        ["Metric", "Q-Learning", "SARSA"],
        ["Avg Steps (Last 50)", f"{summary['q_learning']['avg_steps_last50']}", f"{summary['sarsa']['avg_steps_last50']}"],
        ["Avg Reward (Last 50)", f"{summary['q_learning']['avg_reward_last50']}", f"{summary['sarsa']['avg_reward_last50']}"],
        ["Success Rate (Last 50)", f"{summary['q_learning']['success_rate']}%", f"{summary['sarsa']['success_rate']}%"],
        ["Evaluated Path Steps", f"{summary['q_learning']['eval_path_length'] - 1}", f"{summary['sarsa']['eval_path_length'] - 1}"],
        ["Training Time", f"{summary['q_learning']['time_ms']} ms", f"{summary['sarsa']['time_ms']} ms"]
    ]

    table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.8)

    # Style table header
    for i in range(3):
        cell = table[0, i]
        cell.set_facecolor('#1E293B')
        cell.get_text().set_color('white')
        cell.get_text().set_weight('bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
