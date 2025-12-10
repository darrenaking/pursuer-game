# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Pursuer is a browser game where visitors collectively train an AI agent to catch their cursor using reinforcement learning. The AI runs inference in the browser at 60fps while training happens server-side with PPO.

## Commands

### Development

```bash
# Setup (from server/)
cd server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run server (serves frontend + API on localhost:8000)
python main.py

# Run with hot-reload
python main.py --dev
```

### Production

```bash
gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

Note: Use only 1 worker since training state is in-memory.

## Architecture

### Dual-Runtime Model Execution

The same GRU-based neural network (~25k parameters) runs in two places:
- **Browser (JavaScript)**: `frontend/network.js` - Pure JS implementation of GRU forward pass for real-time inference
- **Server (PyTorch)**: `server/model.py` - Full implementation for training

Weights are exported as JSON from PyTorch and loaded by the JS implementation. Both must produce identical outputs for the same inputs.

### Data Flow

1. Browser loads weights from `/api/weights`
2. Game runs at 60fps, network produces actions from 13-dimensional state observations
3. Browser records (state, action, reward) trajectories during gameplay
4. On episode end, trajectory is POSTed to `/api/trajectory`
5. Server buffers trajectories and runs PPO updates when enough data is collected
6. New weights become available at `/api/weights`

### Key Components

- **State observation** (13 dims): target distance/angle/velocity (if visible), self velocity, wall distances, time remaining, visibility flag
- **Action output** (2 dims): turn rate [-1,1] and acceleration [-1,1], scaled to max_turn_rate and max_acceleration
- **PPO trainer**: Background thread runs training loop every 2 seconds when buffer has ≥16 trajectories

### Configuration

All hyperparameters are in `server/config.py` as dataclasses:
- `ModelConfig`: Network architecture (hidden_size=64, dense_size=64)
- `TrainingConfig`: PPO hyperparameters (lr=3e-4, clip_epsilon=0.2, etc.)
- `GameConfig`: Physics constraints (must match frontend's GAME_CONFIG in game.js)
- `ServerConfig`: Paths and intervals

### API Endpoints

- `GET /api/weights` - Current model weights as JSON
- `POST /api/trajectory` - Submit gameplay trajectory
- `GET /api/stats` - Training statistics
- `GET /api/health` - Health check

## Important Constraints

- Physics parameters in `server/config.py` must match `frontend/game.js` GAME_CONFIG
- GRU implementation in `network.js` must exactly match PyTorch GRU behavior (gate ordering: reset, update, new)
- Training uses single worker/thread to maintain consistent in-memory state
