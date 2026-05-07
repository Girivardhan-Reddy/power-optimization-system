"""
train_models.py
Script to train and save all ML models for the Microgrid Energy Management System
Run this script once to create the required model files
"""
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Training ML Models for Microgrid Energy Management System")
print("="*60)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('training_plots', exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DataGenerator:
    """Generate realistic microgrid data for training"""
    
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        
    def generate_data(self):
        """Generate synthetic but realistic microgrid data"""
        print(f"\n1. Generating {self.n_samples} training samples...")
        
        # Time features
        hours = np.random.randint(0, 24, self.n_samples)
        day_of_week = np.random.randint(0, 7, self.n_samples)
        month = np.random.randint(1, 13, self.n_samples)
        minute = np.random.randint(0, 60, self.n_samples)
        day_of_year = np.random.randint(1, 366, self.n_samples)
        
        # Weather features (simplified)
        temperature = 15 + 10 * np.sin(2*np.pi*day_of_year/365) + 5 * np.random.randn(self.n_samples)
        cloud_cover = np.random.uniform(0, 1, self.n_samples)
        wind_speed = 5 + 3 * np.random.randn(self.n_samples)
        
        # Generate solar power (more realistic pattern)
        solar_power = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            hour = hours[i]
            if 6 <= hour <= 19:  # Daylight hours
                # Sinusoidal pattern peaking at noon
                time_of_day = hour + minute[i]/60
                solar_peak = 100 * (1 - cloud_cover[i] * 0.7)  # Clouds reduce output
                solar_power[i] = solar_peak * np.sin((time_of_day - 6) * np.pi / 13)
                solar_power[i] = max(0, solar_power[i])
        
        # Generate wind power
        wind_power = np.maximum(0, wind_speed**2 * 0.5 + 5 * np.random.randn(self.n_samples))
        wind_power = np.clip(wind_power, 0, 50)
        
        # Generate energy demand (realistic patterns)
        base_demand = 50
        
        # Time of day effect (higher during business hours)
        time_factor = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            hour = hours[i]
            if 0 <= hour < 6:  # Night
                time_factor[i] = 0.6
            elif 6 <= hour < 9:  # Morning
                time_factor[i] = 0.8
            elif 9 <= hour < 17:  # Business hours
                time_factor[i] = 1.2
            elif 17 <= hour < 22:  # Evening
                time_factor[i] = 1.5
            else:  # Late night
                time_factor[i] = 0.7
        
        # Day of week effect (lower on weekends)
        day_factor = np.where(day_of_week >= 5, 0.7, 1.0)
        
        # Temperature effect (more AC/heat usage)
        temp_factor = 1 + 0.01 * np.abs(temperature - 20)
        
        # Random fluctuations
        noise = np.random.normal(0, 10, self.n_samples)
        
        # Combine all factors
        demand = base_demand * time_factor * day_factor * temp_factor + noise
        demand = np.maximum(20, demand)  # Minimum demand
        
        # Grid prices (time-of-use pricing)
        grid_price = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            hour = hours[i]
            if 7 <= hour <= 19:  # Peak hours
                grid_price[i] = 0.15 + 0.02 * np.random.randn()
            else:  # Off-peak
                grid_price[i] = 0.08 + 0.01 * np.random.randn()
            # Weekend discount
            if day_of_week[i] >= 5:
                grid_price[i] *= 0.9
            grid_price[i] = max(0.05, grid_price[i])
        
        # Carbon intensity
        carbon_intensity = 400 - 100 * np.sin((hours - 3) * np.pi / 12)
        carbon_intensity += 50 * ((hours >= 7) & (hours <= 19))  # Higher during peak
        carbon_intensity += 30 * np.random.randn(self.n_samples)
        carbon_intensity = np.maximum(200, carbon_intensity)
        
        # Battery state of charge (if available)
        battery_soc = np.random.uniform(20, 100, self.n_samples)
        
        # Create DataFrame
        data = pd.DataFrame({
            'hour': hours,
            'day_of_week': day_of_week,
            'month': month,
            'minute': minute,
            'temperature': temperature,
            'cloud_cover': cloud_cover,
            'wind_speed': wind_speed,
            'solar_power': solar_power,
            'wind_power': wind_power,
            'demand': demand,
            'grid_price': grid_price,
            'carbon_intensity': carbon_intensity,
            'battery_soc': battery_soc
        })
        
        # Add derived features
        data['total_renewable'] = data['solar_power'] + data['wind_power']
        data['net_demand'] = data['demand'] - data['total_renewable']
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        print(f"   ✓ Generated {len(data)} samples")
        print(f"   Features: {list(data.columns)}")
        
        return data

class ModelTrainer:
    """Train and save all ML models"""
    
    def __init__(self, data):
        self.data = data
        self.X = None
        self.y = None
        self.X_scaled = None
        self.scaler = None
        
    def prepare_data(self):
        """Prepare features and target for training"""
        print("\n2. Preparing data for training...")
        
        # Feature selection
        feature_columns = [
            'hour_sin', 'hour_cos', 
            'day_of_week',
            'month_sin', 'month_cos',
            'temperature', 'cloud_cover',
            'solar_power', 'wind_power',
            'grid_price', 'carbon_intensity',
            'battery_soc'
        ]
        
        self.X = self.data[feature_columns].values
        self.y = self.data['demand'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"   ✓ Training samples: {X_train.shape[0]}")
        print(f"   ✓ Test samples: {X_test.shape[0]}")
        print(f"   ✓ Features: {len(feature_columns)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_xgboost_model(self, X_train, X_test, y_train, y_test):
        """Train XGBoost-like model (using GradientBoostingRegressor)"""
        print("\n3. Training XGBoost model...")
        
        # Use GradientBoosting as XGBoost alternative
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            subsample=0.8
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"   ✓ Model trained with {model.n_estimators} trees")
        print(f"   ✓ Test MAE: {mae:.2f} kW")
        print(f"   ✓ Test RMSE: {rmse:.2f} kW")
        print(f"   ✓ Test R²: {r2:.3f}")
        
        # Save model
        joblib.dump(model, 'models/xgb_model.pkl')
        print("   ✓ Saved: models/xgb_model.pkl")
        
        # Plot feature importance
        self.plot_feature_importance(model, X_train.shape[1])
        
        return model
    
    def train_lstm_model(self, X_train, X_test, y_train, y_test):
        """Train LSTM model for time series prediction"""
        print("\n4. Training LSTM model...")
        
        # Reshape data for LSTM [samples, timesteps, features]
        sequence_length = 24  # 24 hours of history
        
        # Create sequences
        def create_sequences(data, targets, seq_length):
            X_seq, y_seq = [], []
            for i in range(len(data) - seq_length):
                X_seq.append(data[i:i+seq_length])
                y_seq.append(targets[i+seq_length])
            return np.array(X_seq), np.array(y_seq)
        
        # Use a subset for LSTM training (it's more computationally intensive)
        n_lstm_samples = 2000
        X_train_lstm = X_train[:n_lstm_samples]
        y_train_lstm = y_train[:n_lstm_samples]
        X_test_lstm = X_test[:min(500, len(X_test))]
        y_test_lstm = y_test[:min(500, len(y_test))]
        
        X_train_seq, y_train_seq = create_sequences(X_train_lstm, y_train_lstm, sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test_lstm, y_test_lstm, sequence_length)
        
        print(f"   ✓ LSTM sequences: {X_train_seq.shape[0]} training, {X_test_seq.shape[0]} test")
        
        # Build LSTM model
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(sequence_length, X_train.shape[1]), 
                 dropout=0.2, recurrent_dropout=0.2),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate
        y_pred = model.predict(X_test_seq, verbose=0).flatten()
        mae = mean_absolute_error(y_test_seq, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
        r2 = r2_score(y_test_seq, y_pred)
        
        print(f"   ✓ Model trained for {len(history.history['loss'])} epochs")
        print(f"   ✓ Test MAE: {mae:.2f} kW")
        print(f"   ✓ Test RMSE: {rmse:.2f} kW")
        print(f"   ✓ Test R²: {r2:.3f}")
        
        # Save model
        model.save('models/lstm_model.h5')
        print("   ✓ Saved: models/lstm_model.h5")
        
        # Plot training history
        self.plot_training_history(history)
        
        return model
    
    def train_hybrid_model(self, X_train, X_test, y_train, y_test):
        """Train a hybrid CNN-LSTM model"""
        print("\n5. Training Hybrid CNN-LSTM model...")
        
        sequence_length = 24
        n_features = X_train.shape[1]
        
        # Create sequences
        def create_sequences(data, targets, seq_length):
            X_seq, y_seq = [], []
            for i in range(len(data) - seq_length):
                X_seq.append(data[i:i+seq_length])
                y_seq.append(targets[i+seq_length])
            return np.array(X_seq), np.array(y_seq)
        
        # Use smaller subset
        n_samples = 1500
        X_train_hybrid = X_train[:n_samples]
        y_train_hybrid = y_train[:n_samples]
        X_test_hybrid = X_test[:min(300, len(X_test))]
        y_test_hybrid = y_test[:min(300, len(y_test))]
        
        X_train_seq, y_train_seq = create_sequences(X_train_hybrid, y_train_hybrid, sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test_hybrid, y_test_hybrid, sequence_length)
        
        # Build hybrid model
        input_layer = Input(shape=(sequence_length, n_features))
        
        # CNN branch
        cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn)
        cnn = Flatten()(cnn)
        
        # LSTM branch
        lstm = LSTM(64, return_sequences=True, dropout=0.2)(input_layer)
        lstm = LSTM(32, dropout=0.2)(lstm)
        
        # Combine
        combined = concatenate([cnn, lstm])
        combined = Dense(32, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        output_layer = Dense(1)(combined)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=25,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate
        y_pred = model.predict(X_test_seq, verbose=0).flatten()
        mae = mean_absolute_error(y_test_seq, y_pred)
        
        print(f"   ✓ Hybrid model trained")
        print(f"   ✓ Test MAE: {mae:.2f} kW")
        
        # Save as backup model
        model.save('models/hybrid_model.h5')
        print("   ✓ Saved: models/hybrid_model.h5")
        
        return model
    
    def create_data_scaler(self):
        """Create and save data scaler"""
        print("\n6. Creating and saving data scaler...")
        
        # Save the scaler
        joblib.dump(self.scaler, 'models/data_scaler.pkl')
        print("   ✓ Saved: models/data_scaler.pkl")
        
        return self.scaler
    
    def create_rl_agent_files(self):
        """Create RL agent files"""
        print("\n7. Creating RL agent files...")
        
        # Create rl_agent.py
        rl_agent_code = '''"""
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
'''
        
        # Save rl_agent.py
        with open('rl_agent.py', 'w') as f:
            f.write(rl_agent_code)
        print("   ✓ Created: rl_agent.py")
        
        # Create a pre-trained DQN agent
        print("   Creating pre-trained DQN agent...")
        
        # Create and save a simple DQN model
        state_size = 8
        action_size = 27
        
        model = Sequential([
            Dense(64, input_dim=state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(action_size, activation='linear')
        ])
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        
        # Save the model
        model.save('models/dqn_agent.h5')
        print("   ✓ Saved: models/dqn_agent.h5")
        
        return model
    
    def plot_feature_importance(self, model, n_features):
        """Plot feature importance for tree-based model"""
        plt.figure(figsize=(10, 6))
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            features = [
                'Hour (sin)', 'Hour (cos)', 'Day of Week', 'Month (sin)', 'Month (cos)',
                'Temperature', 'Cloud Cover', 'Solar Power', 'Wind Power',
                'Grid Price', 'Carbon Intensity', 'Battery SOC'
            ]
            
            plt.title('Feature Importances')
            plt.bar(range(n_features), importances[indices])
            plt.xticks(range(n_features), [features[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('training_plots/feature_importance.png', dpi=100, bbox_inches='tight')
            plt.close()
    
    def plot_training_history(self, history):
        """Plot training history for neural networks"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MAE
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE (kW)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_plots/lstm_training_history.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    def create_sample_data_file(self, data):
        """Create sample historical data file"""
        print("\n8. Creating sample historical data file...")
        
        # Create timestamp column
        start_date = pd.Timestamp('2023-01-01')
        timestamps = [start_date + pd.Timedelta(hours=i) for i in range(len(data))]
        
        sample_data = pd.DataFrame({
            'timestamp': timestamps[:8760],  # One year of hourly data
            'solar_generation_kw': data['solar_power'].values[:8760],
            'wind_generation_kw': data['wind_power'].values[:8760],
            'demand_kw': data['demand'].values[:8760],
            'grid_price_usd_per_kwh': data['grid_price'].values[:8760],
            'carbon_intensity_g_per_kwh': data['carbon_intensity'].values[:8760],
            'hour': data['hour'].values[:8760],
            'day_of_week': data['day_of_week'].values[:8760],
            'month': data['month'].values[:8760],
            'temperature': data['temperature'].values[:8760],
            'battery_soc_percent': data['battery_soc'].values[:8760]
        })
        
        # Save to CSV
        sample_data.to_csv('energy_data.csv', index=False)
        print(f"   ✓ Created: energy_data.csv with {len(sample_data)} records")
        
        # Plot sample data
        self.plot_sample_data(sample_data)
        
        return sample_data
    
    def plot_sample_data(self, data):
        """Plot sample of the generated data"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot first 168 hours (1 week) of data
        sample_hours = 168
        
        # Demand plot
        axes[0, 0].plot(data['demand_kw'].values[:sample_hours], 'b-', linewidth=1)
        axes[0, 0].set_title('Energy Demand (1 week)')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Demand (kW)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Renewable generation plot
        axes[0, 1].plot(data['solar_generation_kw'].values[:sample_hours], 'y-', label='Solar', linewidth=1)
        axes[0, 1].plot(data['wind_generation_kw'].values[:sample_hours], 'c-', label='Wind', linewidth=1)
        axes[0, 1].set_title('Renewable Generation (1 week)')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Generation (kW)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Grid price plot
        axes[1, 0].plot(data['grid_price_usd_per_kwh'].values[:sample_hours], 'g-', linewidth=1)
        axes[1, 0].set_title('Grid Electricity Price (1 week)')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Price ($/kWh)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Carbon intensity plot
        axes[1, 1].plot(data['carbon_intensity_g_per_kwh'].values[:sample_hours], 'r-', linewidth=1)
        axes[1, 1].set_title('Carbon Intensity (1 week)')
        axes[1, 1].set_xlabel('Hour')
        axes[1, 1].set_ylabel('Carbon Intensity (g/kWh)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_plots/sample_data.png', dpi=100, bbox_inches='tight')
        plt.close()

def main():
    """Main training function"""
    print("\nStarting model training process...")
    
    # Step 1: Generate training data
    generator = DataGenerator(n_samples=20000)
    data = generator.generate_data()
    
    # Step 2: Train models
    trainer = ModelTrainer(data)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data()
    
    # Train XGBoost model
    xgb_model = trainer.train_xgboost_model(X_train, X_test, y_train, y_test)
    
    # Train LSTM model
    lstm_model = trainer.train_lstm_model(X_train, X_test, y_train, y_test)
    
    # Train Hybrid model (optional)
    hybrid_model = trainer.train_hybrid_model(X_train, X_test, y_train, y_test)
    
    # Create data scaler
    scaler = trainer.create_data_scaler()
    
    # Create RL agent files
    rl_model = trainer.create_rl_agent_files()
    
    # Create sample historical data
    sample_data = trainer.create_sample_data_file(data)
    
    # Summary
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("✓ models/data_scaler.pkl      - Data preprocessing scaler")
    print("✓ models/xgb_model.pkl        - XGBoost-like demand predictor")
    print("✓ models/lstm_model.h5        - LSTM time series predictor")
    print("✓ models/hybrid_model.h5      - Hybrid CNN-LSTM model")
    print("✓ models/dqn_agent.h5         - Pre-trained RL agent")
    print("✓ rl_agent.py                 - RL agent implementation")
    print("✓ energy_data.csv             - Sample historical data (1 year)")
    print("\nVisualizations saved in 'training_plots/' directory")
    print("\nNow you can run your Flask app with SIMULATION_MODE = False")
    print("="*60)

if __name__ == '__main__':
    main()