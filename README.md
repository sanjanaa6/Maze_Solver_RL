# Intelligent Maze Solver Using Reinforcement Learning

An end-to-end Reinforcement Learning system that enables an agent to learn and find an optimal path through a maze without explicit pre-programmed rules. Built with **Q-Learning** (Off-Policy TD Control) and **SARSA** (On-Policy TD Control), featuring an interactive **Web Studio Visualizer** and a **CLI Benchmarking Suite**.

---

## 📌 Abstract & Overview

The **Intelligent Maze Solver** models a maze as a 2D GridWorld environment where an agent interacts with states, chooses actions (*Up, Right, Down, Left*), and receives numerical rewards or penalties:
- **Goal Reached**: $+100$
- **Normal Step**: $-1$ (encourages shortest path)
- **Wall Collision**: $-5$ (agent stays in place)
- **Trap / Obstacle**: $-20$

Using an $\epsilon$-greedy strategy, the agent balances exploration of unknown paths with exploitation of accumulated Q-value knowledge.

---

## 🔬 Q-Learning vs SARSA

| Feature | Q-Learning (Off-Policy) | SARSA (On-Policy) |
| :--- | :--- | :--- |
| **Update Strategy** | Uses maximum Q-value of next state ($\max_{a'} Q(S', a')$) | Uses actual action $A'$ taken by $\epsilon$-greedy policy in state $S'$ |
| **Formula** | $Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma \max_{a'} Q(S', a') - Q(S, A) \right]$ | $Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]$ |
| **Risk Preference** | Aggressive; learns optimal shortest path regardless of exploratory risks | Conservative; avoids dangerous paths with neighboring traps during training |

---

## 🚀 Key Features

1. **Custom GridWorld Environment (`rl_core/environment.py`)**:
   - Supports preset mazes (*Easy 5x5, Medium 8x8 with Traps, Cliff Walker, Hard 10x10*).
   - Guaranteed solvable random maze generator using Randomized Depth-First Search (DFS).
2. **Core RL Engine (`rl_core/agents.py`, `rl_core/trainer.py`)**:
   - Pure Python / NumPy implementations of Q-Learning and SARSA.
   - Comprehensive metric tracking: episode rewards, steps per episode, rolling success rate, training execution time.
3. **Interactive Web Studio (`frontend/` & `backend/`)**:
   - Built with **FastAPI** + **HTML5 Canvas** + **Chart.js**.
   - Canvas features: interactive cell painting (Start, Goal, Walls, Traps), policy vector arrows ($\uparrow \rightarrow \downarrow \leftarrow$), Q-value heatmap overlays, and animated agent traversal.
4. **Head-to-Head Benchmarking (`main.py --cli`)**:
   - Automated side-by-side performance evaluation.
   - Saves publication-ready comparison charts to PNG.

---

## 💻 Quick Start & Setup

### 1. Installation
Clone or navigate to the project directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch Interactive Web Studio
```bash
python main.py --web
```
Open your browser and navigate to: **`http://127.0.0.1:8000`**

### 3. Run Headless CLI Benchmark
```bash
python main.py --cli --preset medium --episodes 500
```
This prints the comparison table and saves `results_comparison_medium.png`.

### 4. Run Unit Tests
```bash
python -m unittest tests/test_rl.py
```

---

## 📁 Project Architecture

```
rl_workspace/
├── rl_core/
│   ├── environment.py    # Custom GridWorld Maze Environment & Maze Generators
│   ├── agents.py         # QLearningAgent & SARSAAgent implementations
│   ├── trainer.py        # Single training loops & Head-to-Head Comparison engine
│   └── utils.py          # Matplotlib chart plotting utilities
├── backend/
│   └── app.py            # FastAPI REST endpoints & static file server
├── frontend/
│   ├── index.html        # Glassmorphism visual studio dashboard
│   ├── style.css         # UI dark mode theme, typography & grid layout
│   └── app.js            # Canvas rendering engine, Chart.js integrations & API calls
├── tests/
│   └── test_rl.py        # Automated test suite for RL algorithms & environment
├── main.py               # Application entry point (Web / CLI)
├── requirements.txt      # Dependencies
└── README.md             # Complete project documentation
```
