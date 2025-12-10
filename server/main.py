"""
FastAPI server for the Pursuer game.

Endpoints:
    GET  /api/weights    - Get current model weights
    POST /api/trajectory - Submit gameplay trajectory
    GET  /api/stats      - Get training statistics
    
Static files served from ../frontend/
"""

import os
import sys
import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from model import PursuerNetwork
from buffer import TrajectoryBuffer
from trainer import PPOTrainer
from config import server_config, model_config


# ============================================================
# Request/Response Models
# ============================================================

class TrajectoryData(BaseModel):
    """Trajectory submitted by client."""
    states: List[List[float]]
    actions: List[List[float]]
    rewards: List[float]
    caught: bool
    time_to_catch: Optional[float] = None
    episode_length: int
    arena_width: Optional[float] = None
    arena_height: Optional[float] = None


class StatsResponse(BaseModel):
    """Training statistics."""
    buffer_size: int
    total_episodes: int
    total_catches: int
    catch_rate: float
    average_reward: float
    best_time: Optional[float]
    updates: int
    policy_loss: float
    value_loss: float
    entropy: float


# ============================================================
# Global State
# ============================================================

model: PursuerNetwork = None
buffer: TrajectoryBuffer = None
trainer: PPOTrainer = None


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and trainer on startup."""
    global model, buffer, trainer
    
    print("Initializing Pursuer training system...")
    
    # Create model
    model = PursuerNetwork()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create buffer
    buffer = TrajectoryBuffer()
    
    # Create trainer
    trainer = PPOTrainer(model, buffer)
    
    # Load existing weights if available
    if os.path.exists(server_config.weights_path):
        trainer.load_weights()
    else:
        # Save initial weights
        os.makedirs(os.path.dirname(server_config.weights_path) or '.', exist_ok=True)
        trainer.save_weights()
    
    # Start background training
    trainer.start(interval=2.0)
    
    print("Server ready!")
    
    yield
    
    # Cleanup
    print("Shutting down...")
    trainer.stop()
    trainer.save_weights()


# ============================================================
# App
# ============================================================

app = FastAPI(
    title="Pursuer Training Server",
    description="RL training server for the Pursuer game",
    lifespan=lifespan
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API Routes
# ============================================================

@app.get("/api/weights")
async def get_weights():
    """Return current model weights as JSON."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    weights = model.export_weights()
    return JSONResponse(content=weights)


@app.post("/api/trajectory")
async def submit_trajectory(data: TrajectoryData):
    """
    Submit a gameplay trajectory for training.
    
    Returns acknowledgment with current stats.
    """
    if buffer is None:
        raise HTTPException(status_code=503, detail="Buffer not initialized")
    
    try:
        trajectory = buffer.add_from_dict(data.model_dump())
        
        return {
            "status": "ok",
            "episode_reward": trajectory.total_reward,
            "buffer_size": len(buffer),
            "ready_for_training": buffer.ready_for_training()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get training statistics."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Trainer not initialized")
    
    stats = trainer.get_stats()
    return StatsResponse(**stats)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


# ============================================================
# Static Files (Frontend)
# ============================================================

# Get the frontend directory path
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


@app.get("/")
async def serve_index():
    """Serve the main game page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend not found")


# Mount static files
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pursuer Training Server")
    parser.add_argument("--host", default=server_config.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=server_config.port, help="Port to bind to")
    parser.add_argument("--dev", action="store_true", help="Enable development mode with auto-reload")
    
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.dev
    )


if __name__ == "__main__":
    main()
