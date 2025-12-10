/**
 * The Pursuer - Main Game Logic
 * 
 * Handles:
 *   - Game loop and rendering
 *   - Physics simulation
 *   - State observation
 *   - Trajectory recording
 *   - Communication with training server
 */

// ============================================
// Configuration
// ============================================

const API_BASE = '';  // Same origin, change for separate backend

const GAME_CONFIG = {
    episodeDuration: 10.0,      // seconds
    catchRadius: 30,            // pixels
    
    // Pursuer constraints
    maxSpeed: 350,              // pixels/second
    maxAccel: 800,              // pixels/second^2
    maxTurnRate: 4,             // radians/second
    
    // Field of view
    fovAngle: Math.PI * 0.6,    // ~108 degrees
    fovRange: 400,              // pixels
    
    // Updates
    weightsRefreshInterval: 30000,  // ms - how often to check for new weights
};

// ============================================
// Game Class
// ============================================

class PursuerGame {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.cursor = document.getElementById('cursor');
        
        // Network
        this.network = new PursuerNetwork();
        this.lastWeightsRefresh = 0;
        
        // Mouse state
        this.mouseX = 0;
        this.mouseY = 0;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.mouseVelX = 0;
        this.mouseVelY = 0;
        
        // Pursuer state
        this.pursuer = {
            x: 0,
            y: 0,
            vx: 0,
            vy: 0,
            angle: 0,
            angularVel: 0,
            radius: 15
        };
        
        // Game state
        this.timeRemaining = GAME_CONFIG.episodeDuration;
        this.running = false;
        this.paused = false;
        
        // Trajectory recording
        this.trajectory = {
            states: [],
            actions: [],
            rewards: []
        };
        
        // Stats
        this.localStats = {
            episodes: 0,
            catches: 0,
            bestTime: Infinity
        };
        
        // Server stats
        this.serverStats = null;
        
        this.init();
    }
    
    async init() {
        // Setup canvas
        this.resize();
        window.addEventListener('resize', () => this.resize());
        
        // Setup mouse tracking
        document.addEventListener('mousemove', (e) => this.onMouseMove(e));
        
        // Load weights
        await this.loadWeights();
        
        // Hide loading screen
        document.getElementById('loading').classList.add('hidden');
        
        // Fetch server stats
        this.fetchStats();
        setInterval(() => this.fetchStats(), 10000);
        
        // Start game
        this.startEpisode();
        this.lastTime = performance.now();
        this.gameLoop();
    }
    
    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }
    
    onMouseMove(e) {
        this.lastMouseX = this.mouseX;
        this.lastMouseY = this.mouseY;
        this.mouseX = e.clientX;
        this.mouseY = e.clientY;
        
        this.cursor.style.left = e.clientX + 'px';
        this.cursor.style.top = e.clientY + 'px';
    }
    
    async loadWeights() {
        const success = await this.network.loadWeights(API_BASE);
        this.updateConnectionStatus(success);
        this.lastWeightsRefresh = performance.now();
        return success;
    }
    
    updateConnectionStatus(connected) {
        const dot = document.querySelector('.status-dot');
        const text = document.querySelector('#connection-status span');
        
        if (connected) {
            dot.classList.add('connected');
            text.textContent = 'Connected';
        } else {
            dot.classList.remove('connected');
            text.textContent = 'Offline';
        }
    }
    
    async fetchStats() {
        try {
            const response = await fetch(`${API_BASE}/api/stats`);
            if (response.ok) {
                this.serverStats = await response.json();
                this.updateStatsDisplay();
            }
        } catch (e) {
            // Ignore fetch errors
        }
    }
    
    updateStatsDisplay() {
        const stats = this.serverStats || {};
        
        document.getElementById('episodes').textContent = stats.total_episodes || 0;
        document.getElementById('catches').textContent = stats.total_catches || 0;
        
        const rate = stats.total_episodes > 0 
            ? Math.round((stats.total_catches / stats.total_episodes) * 100) 
            : 0;
        document.getElementById('catch-rate').textContent = rate + '%';
        
        document.getElementById('best-time').textContent = 
            stats.best_time ? stats.best_time.toFixed(2) + 's' : '--';
        
        document.getElementById('updates').textContent = stats.updates || 0;
    }
    
    startEpisode() {
        // Random starting position away from mouse
        const angle = Math.random() * Math.PI * 2;
        const dist = 300 + Math.random() * 200;
        
        this.pursuer.x = this.canvas.width / 2 + Math.cos(angle) * dist;
        this.pursuer.y = this.canvas.height / 2 + Math.sin(angle) * dist;
        this.pursuer.vx = 0;
        this.pursuer.vy = 0;
        this.pursuer.angle = Math.random() * Math.PI * 2;
        this.pursuer.angularVel = 0;
        
        // Clamp to bounds
        this.pursuer.x = Math.max(50, Math.min(this.canvas.width - 50, this.pursuer.x));
        this.pursuer.y = Math.max(50, Math.min(this.canvas.height - 50, this.pursuer.y));
        
        this.timeRemaining = GAME_CONFIG.episodeDuration;
        
        // Reset trajectory
        this.trajectory = {
            states: [],
            actions: [],
            rewards: []
        };
        
        // Reset network hidden state
        this.network.resetHidden();
        
        this.running = true;
    }
    
    endEpisode(caught) {
        this.running = false;
        
        const timeToCapture = GAME_CONFIG.episodeDuration - this.timeRemaining;
        
        // Update local stats
        this.localStats.episodes++;
        if (caught) {
            this.localStats.catches++;
            if (timeToCapture < this.localStats.bestTime) {
                this.localStats.bestTime = timeToCapture;
            }
        }
        
        // Submit trajectory to server
        this.submitTrajectory(caught, timeToCapture);
        
        // Show message
        const msg = document.getElementById('message');
        msg.textContent = caught ? 'CAPTURED' : 'EXPIRED';
        msg.className = caught ? 'show caught' : 'show died';
        
        if (caught) {
            this.cursor.classList.add('caught');
        }
        
        // Restart after delay
        setTimeout(() => {
            msg.className = '';
            this.cursor.classList.remove('caught');
            this.startEpisode();
        }, 1500);
    }
    
    async submitTrajectory(caught, timeToCapture) {
        if (this.trajectory.states.length === 0) return;
        
        const data = {
            states: this.trajectory.states,
            actions: this.trajectory.actions,
            rewards: this.trajectory.rewards,
            caught: caught,
            time_to_catch: caught ? timeToCapture : null,
            episode_length: this.trajectory.states.length,
            arena_width: this.canvas.width,
            arena_height: this.canvas.height
        };
        
        try {
            const response = await fetch(`${API_BASE}/api/trajectory`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            if (response.ok) {
                console.log('Trajectory submitted');
                
                // Maybe refresh weights
                if (performance.now() - this.lastWeightsRefresh > GAME_CONFIG.weightsRefreshInterval) {
                    this.loadWeights();
                }
            }
        } catch (e) {
            console.error('Failed to submit trajectory:', e);
        }
    }
    
    // ============================================
    // State Observation
    // ============================================
    
    canSeeTarget() {
        const dx = this.mouseX - this.pursuer.x;
        const dy = this.mouseY - this.pursuer.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        if (dist > GAME_CONFIG.fovRange) return false;
        
        const angleToTarget = Math.atan2(dy, dx);
        let angleDiff = angleToTarget - this.pursuer.angle;
        
        // Normalize
        while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
        while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;
        
        return Math.abs(angleDiff) < GAME_CONFIG.fovAngle / 2;
    }
    
    getState() {
        const canSee = this.canSeeTarget();
        const p = this.pursuer;
        
        // Target relative position
        const dx = this.mouseX - p.x;
        const dy = this.mouseY - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const maxDist = Math.sqrt(this.canvas.width ** 2 + this.canvas.height ** 2);
        
        const angleToTarget = Math.atan2(dy, dx);
        let relAngle = angleToTarget - p.angle;
        while (relAngle > Math.PI) relAngle -= Math.PI * 2;
        while (relAngle < -Math.PI) relAngle += Math.PI * 2;
        
        // Normalize values
        const maxCursorSpeed = 2000;  // Reasonable max for mouse movement
        
        return [
            // Target (zeroed if not visible)
            canSee ? dist / maxDist : 0,
            canSee ? relAngle / Math.PI : 0,
            canSee ? this.mouseVelX / maxCursorSpeed : 0,
            canSee ? this.mouseVelY / maxCursorSpeed : 0,
            
            // Self
            p.vx / GAME_CONFIG.maxSpeed,
            p.vy / GAME_CONFIG.maxSpeed,
            p.angularVel / GAME_CONFIG.maxTurnRate,
            
            // Walls (distance to each edge, normalized)
            p.x / this.canvas.width,
            (this.canvas.width - p.x) / this.canvas.width,
            p.y / this.canvas.height,
            (this.canvas.height - p.y) / this.canvas.height,
            
            // Meta
            this.timeRemaining / GAME_CONFIG.episodeDuration,
            canSee ? 1 : 0
        ];
    }
    
    // ============================================
    // Physics Update
    // ============================================
    
    update(dt) {
        if (!this.running) return;
        
        // Update mouse velocity estimate
        this.mouseVelX = (this.mouseX - this.lastMouseX) / dt;
        this.mouseVelY = (this.mouseY - this.lastMouseY) / dt;
        
        // Get state and action
        const state = this.getState();
        const action = this.network.getAction(state);
        
        const turnInput = action[0];   // -1 to 1
        const accelInput = action[1];  // -1 to 1
        
        // Record for trajectory
        const prevDist = Math.sqrt(
            (this.mouseX - this.pursuer.x) ** 2 + 
            (this.mouseY - this.pursuer.y) ** 2
        );
        
        // Apply turn (with some smoothing via angular velocity)
        const targetAngularVel = turnInput * GAME_CONFIG.maxTurnRate;
        this.pursuer.angularVel += (targetAngularVel - this.pursuer.angularVel) * 0.3;
        this.pursuer.angle += this.pursuer.angularVel * dt;
        
        // Apply acceleration in facing direction
        const accel = accelInput * GAME_CONFIG.maxAccel;
        this.pursuer.vx += Math.cos(this.pursuer.angle) * accel * dt;
        this.pursuer.vy += Math.sin(this.pursuer.angle) * accel * dt;
        
        // Limit speed
        const speed = Math.sqrt(this.pursuer.vx ** 2 + this.pursuer.vy ** 2);
        if (speed > GAME_CONFIG.maxSpeed) {
            this.pursuer.vx = (this.pursuer.vx / speed) * GAME_CONFIG.maxSpeed;
            this.pursuer.vy = (this.pursuer.vy / speed) * GAME_CONFIG.maxSpeed;
        }
        
        // Update position
        this.pursuer.x += this.pursuer.vx * dt;
        this.pursuer.y += this.pursuer.vy * dt;
        
        // Bounce off walls
        const r = this.pursuer.radius;
        if (this.pursuer.x < r) {
            this.pursuer.x = r;
            this.pursuer.vx *= -0.5;
        }
        if (this.pursuer.x > this.canvas.width - r) {
            this.pursuer.x = this.canvas.width - r;
            this.pursuer.vx *= -0.5;
        }
        if (this.pursuer.y < r) {
            this.pursuer.y = r;
            this.pursuer.vy *= -0.5;
        }
        if (this.pursuer.y > this.canvas.height - r) {
            this.pursuer.y = this.canvas.height - r;
            this.pursuer.vy *= -0.5;
        }
        
        // Calculate reward
        const newDist = Math.sqrt(
            (this.mouseX - this.pursuer.x) ** 2 + 
            (this.mouseY - this.pursuer.y) ** 2
        );
        
        let reward = (prevDist - newDist) * 0.01;  // Reward for getting closer
        reward -= dt * 0.1;  // Small time penalty
        
        // Record trajectory step
        this.trajectory.states.push(state);
        this.trajectory.actions.push(action);
        this.trajectory.rewards.push(reward);
        
        // Check catch
        if (newDist < GAME_CONFIG.catchRadius) {
            // Add catch bonus to final reward
            this.trajectory.rewards[this.trajectory.rewards.length - 1] += 10;
            this.endEpisode(true);
            return;
        }
        
        // Update timer
        this.timeRemaining -= dt;
        document.getElementById('time-display').textContent = this.timeRemaining.toFixed(1);
        
        if (this.timeRemaining <= 0) {
            // Add death penalty to final reward
            this.trajectory.rewards[this.trajectory.rewards.length - 1] -= 5;
            this.endEpisode(false);
        }
    }
    
    // ============================================
    // Rendering
    // ============================================
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        const canSee = this.canSeeTarget();
        const p = this.pursuer;
        
        // Draw FOV cone
        this.ctx.save();
        this.ctx.globalAlpha = 0.1;
        this.ctx.beginPath();
        this.ctx.moveTo(p.x, p.y);
        this.ctx.arc(
            p.x, p.y,
            GAME_CONFIG.fovRange,
            p.angle - GAME_CONFIG.fovAngle / 2,
            p.angle + GAME_CONFIG.fovAngle / 2
        );
        this.ctx.closePath();
        this.ctx.fillStyle = canSee ? '#ff00ff' : '#00ffff';
        this.ctx.fill();
        this.ctx.restore();
        
        // Draw pursuer glow
        const gradient = this.ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius * 3);
        gradient.addColorStop(0, canSee ? 'rgba(255, 0, 255, 0.4)' : 'rgba(0, 255, 255, 0.4)');
        gradient.addColorStop(1, 'transparent');
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(p.x, p.y, p.radius * 3, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Draw pursuer body
        this.ctx.save();
        this.ctx.translate(p.x, p.y);
        this.ctx.rotate(p.angle);
        
        this.ctx.beginPath();
        this.ctx.moveTo(p.radius * 1.5, 0);
        this.ctx.lineTo(-p.radius, -p.radius * 0.8);
        this.ctx.lineTo(-p.radius * 0.5, 0);
        this.ctx.lineTo(-p.radius, p.radius * 0.8);
        this.ctx.closePath();
        
        this.ctx.fillStyle = canSee ? '#ff00ff' : '#00ffff';
        this.ctx.shadowColor = canSee ? '#ff00ff' : '#00ffff';
        this.ctx.shadowBlur = 20;
        this.ctx.fill();
        
        this.ctx.restore();
        
        // Draw targeting line if visible
        if (canSee) {
            this.ctx.save();
            this.ctx.globalAlpha = 0.3;
            this.ctx.strokeStyle = '#ff00ff';
            this.ctx.lineWidth = 1;
            this.ctx.setLineDash([5, 10]);
            this.ctx.beginPath();
            this.ctx.moveTo(p.x, p.y);
            this.ctx.lineTo(this.mouseX, this.mouseY);
            this.ctx.stroke();
            this.ctx.restore();
        }
    }
    
    // ============================================
    // Game Loop
    // ============================================
    
    gameLoop() {
        const now = performance.now();
        const dt = Math.min((now - this.lastTime) / 1000, 0.1);
        this.lastTime = now;
        
        if (!this.paused) {
            this.update(dt);
        }
        
        this.draw();
        
        requestAnimationFrame(() => this.gameLoop());
    }
}

// ============================================
// Start Game
// ============================================

window.addEventListener('load', () => {
    window.game = new PursuerGame();
});
