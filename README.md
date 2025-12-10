# The Pursuer

A browser game where visitors collectively train an AI pursuer to catch their cursor using reinforcement learning.

## Architecture

- **Frontend**: HTML/JS game that runs GRU inference locally at 60fps
- **Backend**: Python server that collects trajectories and trains with PPO
- **Model**: GRU-based policy network with ~25k parameters

## Local Development

### Prerequisites

- Python 3.10+
- Node.js (optional, for serving frontend in dev)

### Setup

1. Create a virtual environment:
```bash
cd server
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
```

4. Open `http://localhost:8000` in your browser

The server serves both the API and the static frontend files.

### Development mode

For hot-reloading during development:
```bash
python main.py --dev
```

## Project Structure

```
pursuer-game/
├── server/
│   ├── main.py              # FastAPI server entry point
│   ├── model.py             # Neural network architecture
│   ├── trainer.py           # PPO training logic
│   ├── buffer.py            # Trajectory storage
│   ├── config.py            # Hyperparameters
│   ├── requirements.txt     # Python dependencies
│   └── weights/             # Saved model weights
│       └── latest.json      # Current best weights (served to clients)
├── frontend/
│   ├── index.html           # Game page
│   ├── game.js              # Game logic and rendering
│   ├── network.js           # GRU inference in JavaScript
│   └── style.css            # Styling
└── README.md
```

## API Endpoints

- `GET /api/weights` - Get current model weights as JSON
- `POST /api/trajectory` - Submit a gameplay trajectory for training
- `GET /api/stats` - Get training statistics

## Production Deployment

### Option 1: Single server (simple)

1. Set environment variables:
```bash
export PURSUER_ENV=production
export PURSUER_HOST=0.0.0.0
export PURSUER_PORT=8000
```

2. Run with gunicorn:
```bash
gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

Note: Use only 1 worker since training state is in-memory.

### Option 2: Separate frontend/backend

Serve `frontend/` from any static host (GitHub Pages, Netlify, etc.) and point it to your API server by editing `frontend/game.js`:

```javascript
const API_BASE = 'https://your-api-server.com';
```

### Option 3: Docker

```bash
docker build -t pursuer-game .
docker run -p 8000:8000 pursuer-game
```

## Configuration

Edit `server/config.py` to tune:

- `HIDDEN_SIZE` - GRU hidden dimension (default: 64)
- `LEARNING_RATE` - PPO learning rate (default: 3e-4)
- `BATCH_SIZE` - Trajectories per training batch (default: 32)
- `CLIP_EPSILON` - PPO clipping parameter (default: 0.2)
- `ENTROPY_COEF` - Exploration bonus (default: 0.01)

## How Training Works

1. Player loads page, downloads current weights
2. Player plays game, browser records (state, action, reward) tuples
3. On episode end, browser sends trajectory to server
4. Server stores trajectory in buffer
5. When buffer has enough data, server runs PPO update
6. Updated weights become available at `/api/weights`
7. Clients periodically fetch new weights

## License

MIT
