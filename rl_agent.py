"""
RL Agent for Microgrid Energy Management
Deep Q-Network implementation
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

class DQNAgent:
    """Deep Q-Network Agent for Microgrid Control"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """Build neural network for Q-function approximation"""
        # State input
        state_input = Input(shape=(self.state_size,))
        
        # Shared layers
        x = Dense(64, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        
        # Advantage stream
        advantage = Dense(32, activation='relu')(x)
        advantage = Dense(self.action_size, activation='linear')(advantage)
        
        # Value stream
        value = Dense(32, activation='relu')(x)
        value = Dense(1, activation='linear')(value)
        
        # Combine using Dueling DQN architecture
        output = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        
        model = Model(inputs=state_input, outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def update_target_model(self):
        """Copy weights from model to target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Select action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=32):
        """Train on random samples from replay memory"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([experience[0][0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3][0] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Predict Q-values for current states
        current_q = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        next_q = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values using Bellman equation
        for i in range(batch_size):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q[i])
        
        # Train the model
        self.model.fit(states, current_q, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model from file"""
        self.model = tf.keras.models.load_model(name)
        self.target_model = tf.keras.models.load_model(name)
        self.epsilon = self.epsilon_min  # Start with minimal exploration
    
    def save(self, name):
        """Save model to file"""
        self.model.save(name)

class MicrogridEnv:
    """Microgrid Environment for RL Training"""
    
    def __init__(self):
        # State: [demand, solar, wind, battery_soc, grid_price, carbon_intensity, hour, day_of_week]
        self.observation_space = type('obj', (object,), {'shape': (8,)})()
        
        # Actions: combination of battery charge/discharge and grid import/export
        # 3 battery actions (charge, discharge, idle) × 3 grid actions (import, export, idle) × 3 curtailment levels
        self.action_space = type('obj', (object,), {'n': 27})()
        
        # Environment parameters
        self.battery_capacity = 200  # kWh
        self.max_battery_power = 30  # kW
        self.max_grid_power = 50     # kW
        
    def reset(self):
        """Reset environment to initial state"""
        # Random initial state
        state = np.array([
            np.random.uniform(20, 100),      # demand_kw
            np.random.uniform(0, 80),        # solar_kw
            np.random.uniform(0, 40),        # wind_kw
            np.random.uniform(0.2, 0.8),     # battery_soc (ratio)
            np.random.uniform(0.05, 0.2),    # grid_price
            np.random.uniform(200, 500),     # carbon_intensity
            np.random.uniform(0, 24),        # hour (normalized)
            np.random.uniform(0, 7)          # day_of_week (normalized)
        ])
        return state.reshape(1, -1)
    
    def step(self, action):
        """Execute one time step in environment"""
        # Decode action
        battery_action = (action // 9) % 3 - 1  # -1: discharge, 0: idle, 1: charge
        grid_action = (action // 3) % 3 - 1     # -1: export, 0: idle, 1: import
        curtailment_action = action % 3         # 0: none, 1: moderate, 2: high
        
        # Get current state (simplified - in real env this would be actual state)
        state = self.reset().flatten()
        
        # Calculate reward components
        cost = -state[4] * abs(grid_action) * 10  # Cost for grid import
        emission = -state[5] * (grid_action > 0) * 0.1  # Penalty for carbon emissions
        battery_penalty = -abs(battery_action) * 0.5  # Small penalty for battery usage
        reward = cost + emission + battery_penalty
        
        # Next state (simplified transition)
        next_state = state + np.random.randn(8) * 0.1
        next_state = np.clip(next_state, [20, 0, 0, 0, 0.05, 200, 0, 0], 
                                       [100, 80, 40, 1, 0.2, 500, 24, 7])
        
        done = np.random.random() < 0.05  # 5% chance episode ends
        
        return next_state.reshape(1, -1), reward, done, {}
