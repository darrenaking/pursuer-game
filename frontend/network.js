/**
 * Neural network inference in JavaScript.
 * 
 * Implements GRU forward pass to run the trained policy in the browser.
 * Weights are fetched from the server and updated periodically.
 */

class PursuerNetwork {
    constructor() {
        this.weights = null;
        this.config = null;
        this.hiddenState = null;
    }

    /**
     * Load weights from server.
     */
    async loadWeights(apiBase = '') {
        try {
            const response = await fetch(`${apiBase}/api/weights`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            this.weights = data;
            this.config = data.config;
            
            // Initialize hidden state
            this.resetHidden();
            
            console.log('Weights loaded:', this.config);
            return true;
        } catch (error) {
            console.error('Failed to load weights:', error);
            return false;
        }
    }

    /**
     * Reset hidden state to zeros (call at episode start).
     */
    resetHidden() {
        if (this.config) {
            this.hiddenState = new Float32Array(this.config.hidden_size);
        }
    }

    /**
     * Matrix-vector multiplication: y = Wx + b
     */
    linear(x, weight, bias) {
        const outSize = weight.length;
        const inSize = weight[0].length;
        const y = new Float32Array(outSize);
        
        for (let i = 0; i < outSize; i++) {
            let sum = bias[i];
            for (let j = 0; j < inSize; j++) {
                sum += weight[i][j] * x[j];
            }
            y[i] = sum;
        }
        
        return y;
    }

    /**
     * ReLU activation.
     */
    relu(x) {
        const y = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) {
            y[i] = Math.max(0, x[i]);
        }
        return y;
    }

    /**
     * Sigmoid activation.
     */
    sigmoid(x) {
        const y = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) {
            y[i] = 1 / (1 + Math.exp(-x[i]));
        }
        return y;
    }

    /**
     * Tanh activation.
     */
    tanh(x) {
        const y = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) {
            y[i] = Math.tanh(x[i]);
        }
        return y;
    }

    /**
     * Element-wise operations.
     */
    add(a, b) {
        const y = new Float32Array(a.length);
        for (let i = 0; i < a.length; i++) {
            y[i] = a[i] + b[i];
        }
        return y;
    }

    mul(a, b) {
        const y = new Float32Array(a.length);
        for (let i = 0; i < a.length; i++) {
            y[i] = a[i] * b[i];
        }
        return y;
    }

    sub(a, b) {
        const y = new Float32Array(a.length);
        for (let i = 0; i < a.length; i++) {
            y[i] = a[i] - b[i];
        }
        return y;
    }

    scale(a, s) {
        const y = new Float32Array(a.length);
        for (let i = 0; i < a.length; i++) {
            y[i] = a[i] * s;
        }
        return y;
    }

    ones(size) {
        const y = new Float32Array(size);
        for (let i = 0; i < size; i++) {
            y[i] = 1;
        }
        return y;
    }

    /**
     * GRU cell forward pass.
     * 
     * PyTorch GRU uses:
     *   r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
     *   z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
     *   n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
     *   h' = (1 - z) * n + z * h
     * 
     * The weights are packed as [r, z, n] in the matrices.
     */
    gruStep(input, hidden) {
        const gru = this.weights.gru;
        const hiddenSize = this.config.hidden_size;
        
        // Weight matrices are (3*hidden_size, input_size) and (3*hidden_size, hidden_size)
        // Bias vectors are (3*hidden_size,)
        // They're packed as [reset, update, new] gates
        
        // Compute input contributions: W_ih @ x + b_ih
        const ih = new Float32Array(3 * hiddenSize);
        for (let i = 0; i < 3 * hiddenSize; i++) {
            let sum = gru.bias_ih[i];
            for (let j = 0; j < input.length; j++) {
                sum += gru.weight_ih[i][j] * input[j];
            }
            ih[i] = sum;
        }
        
        // Compute hidden contributions: W_hh @ h + b_hh
        const hh = new Float32Array(3 * hiddenSize);
        for (let i = 0; i < 3 * hiddenSize; i++) {
            let sum = gru.bias_hh[i];
            for (let j = 0; j < hiddenSize; j++) {
                sum += gru.weight_hh[i][j] * hidden[j];
            }
            hh[i] = sum;
        }
        
        // Extract gates
        const r_input = ih.slice(0, hiddenSize);
        const z_input = ih.slice(hiddenSize, 2 * hiddenSize);
        const n_input = ih.slice(2 * hiddenSize, 3 * hiddenSize);
        
        const r_hidden = hh.slice(0, hiddenSize);
        const z_hidden = hh.slice(hiddenSize, 2 * hiddenSize);
        const n_hidden = hh.slice(2 * hiddenSize, 3 * hiddenSize);
        
        // Reset gate: r = sigmoid(r_input + r_hidden)
        const r = this.sigmoid(this.add(r_input, r_hidden));
        
        // Update gate: z = sigmoid(z_input + z_hidden)
        const z = this.sigmoid(this.add(z_input, z_hidden));
        
        // New gate: n = tanh(n_input + r * n_hidden)
        const n = this.tanh(this.add(n_input, this.mul(r, n_hidden)));
        
        // New hidden: h' = (1 - z) * n + z * h
        const oneMinusZ = this.sub(this.ones(hiddenSize), z);
        const newHidden = this.add(this.mul(oneMinusZ, n), this.mul(z, hidden));
        
        return newHidden;
    }

    /**
     * Full forward pass.
     * 
     * Args:
     *   state: Array of input features
     *   
     * Returns:
     *   {action: [turn, accel], value: number}
     */
    forward(state) {
        if (!this.weights) {
            console.warn('Weights not loaded, returning random action');
            return {
                action: [Math.random() * 2 - 1, Math.random() * 2 - 1],
                value: 0
            };
        }
        
        const w = this.weights;
        
        // Input dense: ReLU(W @ x + b)
        let x = this.linear(state, w.input_dense.weight, w.input_dense.bias);
        x = this.relu(x);
        
        // GRU step
        this.hiddenState = this.gruStep(x, this.hiddenState);
        
        // Post-GRU dense: ReLU(W @ h + b)
        let post = this.linear(this.hiddenState, w.post_gru_dense.weight, w.post_gru_dense.bias);
        post = this.relu(post);
        
        // Policy head: tanh(W @ post + b)
        let actionMean = this.linear(post, w.policy_mean.weight, w.policy_mean.bias);
        actionMean = this.tanh(actionMean);
        
        // Value head
        const value = this.linear(post, w.value_head.weight, w.value_head.bias);
        
        // Sample from Gaussian (or just use mean for deterministic)
        // For gameplay, we'll add a small amount of noise for exploration
        const actionStd = w.policy_log_std.map(ls => Math.exp(ls));
        const action = [
            actionMean[0] + this.sampleNormal() * actionStd[0] * 0.1,
            actionMean[1] + this.sampleNormal() * actionStd[1] * 0.1
        ];
        
        // Clamp to [-1, 1]
        action[0] = Math.max(-1, Math.min(1, action[0]));
        action[1] = Math.max(-1, Math.min(1, action[1]));
        
        return {
            action: action,
            actionMean: [actionMean[0], actionMean[1]],
            value: value[0]
        };
    }

    /**
     * Sample from standard normal distribution (Box-Muller).
     */
    sampleNormal() {
        const u1 = Math.random();
        const u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    /**
     * Get action for current state.
     */
    getAction(state) {
        const result = this.forward(state);
        return result.action;
    }

    /**
     * Check if weights are loaded.
     */
    isReady() {
        return this.weights !== null;
    }
}

// Export for use in game.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PursuerNetwork;
}
