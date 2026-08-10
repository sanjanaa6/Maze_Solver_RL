import http.server
import socketserver
import json
import os
import urllib.parse
from typing import Dict, Any

from rl_core.environment import MazeEnv
from rl_core.agents import QLearningAgent, SARSAAgent
from rl_core.trainer import Trainer

PORT = 8000
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

class RLStudioHandler(http.server.SimpleHTTPRequestHandler):
    """
    Standard library HTTP request handler serving REST API endpoints and frontend static files.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def _send_json_response(self, data: Any, status_code: int = 200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path).path

        if parsed_path == "/api/presets":
            res = {
                "presets": ["easy", "medium", "hard", "cliff_walker"],
                "easy": MazeEnv.get_preset_maze("easy").get_grid_state(),
                "medium": MazeEnv.get_preset_maze("medium").get_grid_state(),
                "hard": MazeEnv.get_preset_maze("hard").get_grid_state(),
                "cliff_walker": MazeEnv.get_preset_maze("cliff_walker").get_grid_state()
            }
            return self._send_json_response(res)

        # Fallback to serving static files from FRONTEND_DIR
        return super().do_GET()

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path).path
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
            req = json.loads(post_data.decode('utf-8')) if post_data else {}
        except Exception:
            return self._send_json_response({"error": "Invalid JSON body"}, 400)

        if parsed_path == "/api/maze/generate":
            rows = req.get("rows", 8)
            cols = req.get("cols", 8)
            wall_density = req.get("wall_density", 0.25)
            trap_density = req.get("trap_density", 0.05)

            env = MazeEnv.generate_random_solvable_maze(rows, cols, wall_density, trap_density)
            return self._send_json_response(env.get_grid_state())

        elif parsed_path == "/api/train":
            algo = req.get("algorithm", "q_learning").lower()
            maze_cfg = req.get("maze", {})
            episodes = req.get("episodes", 400)
            alpha = req.get("alpha", 0.1)
            gamma = req.get("gamma", 0.99)
            epsilon = req.get("epsilon", 1.0)
            epsilon_min = req.get("epsilon_min", 0.01)
            epsilon_decay = req.get("epsilon_decay", 0.995)

            start = tuple(maze_cfg["start_pos"]) if "start_pos" in maze_cfg and maze_cfg["start_pos"] else None
            goal = tuple(maze_cfg["goal_pos"]) if "goal_pos" in maze_cfg and maze_cfg["goal_pos"] else None

            env = MazeEnv(
                grid=maze_cfg.get("grid"),
                rows=maze_cfg.get("rows", 8),
                cols=maze_cfg.get("cols", 8),
                start_pos=start,
                goal_pos=goal,
                step_reward=maze_cfg.get("step_reward", -1.0),
                wall_penalty=maze_cfg.get("wall_penalty", -5.0),
                goal_reward=maze_cfg.get("goal_reward", 100.0),
                trap_penalty=maze_cfg.get("trap_penalty", -20.0)
            )

            if algo == "q_learning":
                agent = QLearningAgent(env.rows, env.cols, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay)
            elif algo == "sarsa":
                agent = SARSAAgent(env.rows, env.cols, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay)
            else:
                return self._send_json_response({"error": f"Unsupported algorithm '{algo}'"}, 400)

            results = Trainer.train_agent(env, agent, episodes=episodes)
            results["policy"] = {f"{r},{c}": action for (r, c), action in agent.get_policy().items()}
            results["maze"] = env.get_grid_state()
            return self._send_json_response(results)

        elif parsed_path == "/api/compare":
            maze_cfg = req.get("maze", {})
            episodes = req.get("episodes", 400)
            alpha = req.get("alpha", 0.1)
            gamma = req.get("gamma", 0.99)
            epsilon_decay = req.get("epsilon_decay", 0.995)

            start = tuple(maze_cfg["start_pos"]) if "start_pos" in maze_cfg and maze_cfg["start_pos"] else None
            goal = tuple(maze_cfg["goal_pos"]) if "goal_pos" in maze_cfg and maze_cfg["goal_pos"] else None

            env = MazeEnv(
                grid=maze_cfg.get("grid"),
                rows=maze_cfg.get("rows", 8),
                cols=maze_cfg.get("cols", 8),
                start_pos=start,
                goal_pos=goal,
                step_reward=maze_cfg.get("step_reward", -1.0),
                wall_penalty=maze_cfg.get("wall_penalty", -5.0),
                goal_reward=maze_cfg.get("goal_reward", 100.0),
                trap_penalty=maze_cfg.get("trap_penalty", -20.0)
            )

            comparison = Trainer.compare(env, episodes=episodes, alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay)
            comparison["maze"] = env.get_grid_state()
            return self._send_json_response(comparison)

        else:
            return self._send_json_response({"error": "Not Found"}, 404)

def run_server(host: str = "127.0.0.1", port: int = 8000):
    handler = RLStudioHandler
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), handler) as httpd:
        print(f"[*] Intelligent Maze Solver Studio running at http://{host}:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
