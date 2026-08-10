from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os

from rl_core.environment import MazeEnv
from rl_core.agents import QLearningAgent, SARSAAgent
from rl_core.trainer import Trainer

app = FastAPI(
    title="Intelligent Maze Solver API",
    description="Reinforcement Learning API for Q-Learning and SARSA GridWorld Maze Solver",
    version="1.0.0"
)

# CORS middleware for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas ---

class MazeConfigSchema(BaseModel):
    grid: Optional[List[List[int]]] = None
    rows: int = 8
    cols: int = 8
    start_pos: Optional[List[int]] = None
    goal_pos: Optional[List[int]] = None
    step_reward: float = -1.0
    wall_penalty: float = -5.0
    goal_reward: float = 100.0
    trap_penalty: float = -20.0

class TrainRequestSchema(BaseModel):
    algorithm: str = Field("q_learning", description="q_learning or sarsa")
    maze: MazeConfigSchema
    episodes: int = 400
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995

class CompareRequestSchema(BaseModel):
    maze: MazeConfigSchema
    episodes: int = 400
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_decay: float = 0.995

class GenerateMazeRequestSchema(BaseModel):
    rows: int = 8
    cols: int = 8
    wall_density: float = 0.25
    trap_density: float = 0.05

# --- Endpoints ---

@app.get("/api/presets")
def get_presets():
    """Returns available preset maze names and structures."""
    return {
        "presets": ["easy", "medium", "hard", "cliff_walker"],
        "easy": MazeEnv.get_preset_maze("easy").get_grid_state(),
        "medium": MazeEnv.get_preset_maze("medium").get_grid_state(),
        "hard": MazeEnv.get_preset_maze("hard").get_grid_state(),
        "cliff_walker": MazeEnv.get_preset_maze("cliff_walker").get_grid_state()
    }

@app.post("/api/maze/generate")
def generate_maze(req: GenerateMazeRequestSchema):
    """Generates a random solvable maze with path guarantee."""
    try:
        env = MazeEnv.generate_random_solvable_maze(
            rows=req.rows,
            cols=req.cols,
            wall_density=req.wall_density,
            trap_density=req.trap_density
        )
        return env.get_grid_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
def train_agent_endpoint(req: TrainRequestSchema):
    """Trains a single Q-Learning or SARSA agent and returns metrics, policy, and Q-table."""
    start_tuple = tuple(req.maze.start_pos) if req.maze.start_pos else None
    goal_tuple = tuple(req.maze.goal_pos) if req.maze.goal_pos else None

    env = MazeEnv(
        grid=req.maze.grid,
        rows=req.maze.rows,
        cols=req.maze.cols,
        start_pos=start_tuple,
        goal_pos=goal_tuple,
        step_reward=req.maze.step_reward,
        wall_penalty=req.maze.wall_penalty,
        goal_reward=req.maze.goal_reward,
        trap_penalty=req.maze.trap_penalty
    )

    algo = req.algorithm.lower()
    if algo == "q_learning":
        agent = QLearningAgent(
            rows=env.rows,
            cols=env.cols,
            alpha=req.alpha,
            gamma=req.gamma,
            epsilon=req.epsilon,
            epsilon_min=req.epsilon_min,
            epsilon_decay=req.epsilon_decay
        )
    elif algo == "sarsa":
        agent = SARSAAgent(
            rows=env.rows,
            cols=env.cols,
            alpha=req.alpha,
            gamma=req.gamma,
            epsilon=req.epsilon,
            epsilon_min=req.epsilon_min,
            epsilon_decay=req.epsilon_decay
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported algorithm '{req.algorithm}'. Must be 'q_learning' or 'sarsa'.")

    results = Trainer.train_agent(env, agent, episodes=req.episodes)
    results["policy"] = {f"{r},{c}": action for (r, c), action in agent.get_policy().items()}
    results["maze"] = env.get_grid_state()
    return results

@app.post("/api/compare")
def compare_endpoint(req: CompareRequestSchema):
    """Runs head-to-head benchmarking for Q-Learning vs SARSA on the given maze."""
    start_tuple = tuple(req.maze.start_pos) if req.maze.start_pos else None
    goal_tuple = tuple(req.maze.goal_pos) if req.maze.goal_pos else None

    env = MazeEnv(
        grid=req.maze.grid,
        rows=req.maze.rows,
        cols=req.maze.cols,
        start_pos=start_tuple,
        goal_pos=goal_tuple,
        step_reward=req.maze.step_reward,
        wall_penalty=req.maze.wall_penalty,
        goal_reward=req.maze.goal_reward,
        trap_penalty=req.maze.trap_penalty
    )

    comparison = Trainer.compare(
        env,
        episodes=req.episodes,
        alpha=req.alpha,
        gamma=req.gamma,
        epsilon_decay=req.epsilon_decay
    )
    comparison["maze"] = env.get_grid_state()
    return comparison

# Mount Static Frontend
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")
