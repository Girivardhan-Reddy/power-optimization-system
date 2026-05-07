"""
Flask Web Application for Microgrid Energy Management System
With robust error handling and simulation mode
"""
from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import json
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import warnings
import os
import random
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'microgrid_energy_management_secret_key'

# Global variables for models
forecast_model = None
xgb_model = None
rl_agent = None
data_scaler = None

# Configuration - SET TO FALSE TO LOAD MODELS
SIMULATION_MODE = False  # Set to False if you want to try loading real models

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

class MicrogridSimulator:
    """Simulate microgrid operations with ML and RL"""
    
    def __init__(self):
        self.load_models()
        self.battery_capacity_kwh = 200
        self.battery_soc_kwh = self.battery_capacity_kwh * 0.5  # Start at 50% SOC
        
        # Try to load historical data, generate if not available
        try:
            self.historical_data = pd.read_csv('energy_data.csv', parse_dates=['timestamp'])
            print(f"✓ Loaded historical data: {len(self.historical_data)} records")
        except:
            print("⚠ Historical data not found, using synthetic generation")
            self.historical_data = None
        
    def load_models(self):
        """Load trained ML and RL models with graceful fallback"""
        global forecast_model, xgb_model, rl_agent, data_scaler
        
        print("\n" + "="*60)
        print("Loading Models for Microgrid System")
        print("="*60)
        
        models_loaded = 0
        
        # Always create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('templates', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        if SIMULATION_MODE:
            print("Running in SIMULATION MODE")
            print("Using synthetic data and heuristic strategies")
            print("Set SIMULATION_MODE = False to try loading ML models")
            return
        
        # Try to load data scaler
        try:
            if os.path.exists('models/data_scaler.pkl'):
                data_scaler = joblib.load('models/data_scaler.pkl')
                print("✓ Data scaler loaded successfully")
                models_loaded += 1
            else:
                print("⚠ Data scaler not found, using default scaling")
                from sklearn.preprocessing import StandardScaler
                data_scaler = StandardScaler()
        except Exception as e:
            print(f"✗ Error loading scaler: {e}")
            from sklearn.preprocessing import StandardScaler
            data_scaler = StandardScaler()
        
        # Try to load LSTM model
        try:
            if os.path.exists('models/lstm_model.h5'):
                forecast_model = load_model('models/lstm_model.h5')
                print("✓ LSTM model loaded successfully")
                models_loaded += 1
            else:
                print("⚠ LSTM model not found")
        except Exception as e:
            print(f"✗ Error loading LSTM model: {e}")
            forecast_model = None
        
        # Try to load XGBoost model
        try:
            if os.path.exists('models/xgb_model.pkl'):
                xgb_model = joblib.load('models/xgb_model.pkl')
                print("✓ XGBoost model loaded successfully")
                models_loaded += 1
            else:
                print("⚠ XGBoost model not found")
        except Exception as e:
            print(f"✗ Error loading XGBoost model: {e}")
            xgb_model = None
        
        # Try to load RL agent
        try:
            if os.path.exists('models/dqn_agent.h5'):
                # For RL agent, we need to check for the class definition
                try:
                    # Try to import the RL agent module
                    from rl_agent import DQNAgent, MicrogridEnv
                    env = MicrogridEnv()
                    rl_agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
                    rl_agent.load('models/dqn_agent.h5')
                    print("✓ RL agent loaded successfully")
                    models_loaded += 1
                except ImportError as e:
                    print(f"✗ RL agent module not found: {e}")
                    print("   Make sure rl_agent.py exists with DQNAgent and MicrogridEnv classes")
                    rl_agent = None
            else:
                print("⚠ RL agent not found")
        except Exception as e:
            print(f"✗ Error loading RL agent: {e}")
            rl_agent = None
        
        print(f"\n✓ Total models loaded: {models_loaded}/4")
        if models_loaded < 4:
            print("⚠ Some models are missing - using simulation strategies")
            print("   To use real models:")
            print("   1. Set SIMULATION_MODE = False")
            print("   2. Ensure models are in 'models/' directory:")
            print("      - lstm_model.h5")
            print("      - xgb_model.pkl")
            print("      - data_scaler.pkl")
            print("      - dqn_agent.h5")
            print("   3. Create rl_agent.py with DQNAgent and MicrogridEnv classes")
        print("="*60)
    
    def generate_synthetic_data(self, hours=24):
        """Generate synthetic data for simulation with realistic patterns"""
        print(f"Generating {hours} hours of synthetic energy data...")
        
        timestamps = [datetime.now() + timedelta(hours=i) for i in range(hours)]
        
        data = []
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            day_of_week = ts.weekday()
            month = ts.month
            minute = ts.minute
            
            # More realistic solar generation
            solar = 0
            if 6 <= hour <= 19:  # Daylight hours
                # Peak at noon, sinusoidal pattern
                time_of_day = hour + minute/60
                solar = 80 * np.sin((time_of_day - 6) * np.pi / 13)
                # Add some randomness
                solar *= (1 + 0.2 * np.random.randn())
                # Cloud effects
                if np.random.random() < 0.3:  # 30% chance of clouds
                    solar *= 0.5
                solar = max(0, solar)
            
            # Wind generation (higher at night, more variable)
            wind_base = 25 + 10 * np.sin(hour * np.pi / 12)
            if 22 <= hour or hour <= 4:  # Higher at night
                wind_base += 15
            wind = max(0, wind_base + 15 * np.random.randn())
            
            # Demand pattern (higher during business hours)
            demand_base = 50
            if 8 <= hour <= 18:  # Business hours
                demand_base += 30
            if 17 <= hour <= 20:  # Evening peak
                demand_base += 20
            
            # Weekend adjustment
            if day_of_week >= 5:  # Weekend
                demand_base *= 0.7
            
            # Add random fluctuations
            demand = max(20, demand_base + 10 * np.random.randn())
            
            # Time-of-use pricing
            if 7 <= hour <= 19:  # Peak hours
                price = 0.15 + 0.03 * np.random.randn()
            else:  # Off-peak
                price = 0.08 + 0.02 * np.random.randn()
            
            # Weekend/Weekday price difference
            if day_of_week >= 5:
                price *= 0.9
            
            price = max(0.05, price)
            
            # Carbon intensity (lower at night, higher during peak)
            carbon_base = 400 - 100 * np.sin((hour - 3) * np.pi / 12)
            if 7 <= hour <= 19:  # Higher during peak hours
                carbon_base += 50
            carbon = max(200, carbon_base + 50 * np.random.randn())
            
            data.append({
                'timestamp': ts,
                'solar_generation_kw': float(round(solar, 2)),
                'wind_generation_kw': float(round(wind, 2)),
                'demand_kw': float(round(demand, 2)),
                'grid_price_usd_per_kwh': float(round(price, 3)),
                'carbon_intensity_g_per_kwh': float(round(carbon, 1)),
                'hour': int(hour),
                'day_of_week': int(day_of_week),
                'month': int(month),
                'minute': int(minute)
            })
        
        df = pd.DataFrame(data)
        print(f"✓ Generated {len(df)} records")
        return df
    
    def predict_demand(self, current_state):
        """Predict energy demand - uses ML if available, otherwise simulation"""
        try:
            # Check if we have both models loaded and SIMULATION_MODE is False
            if not SIMULATION_MODE and forecast_model is not None and xgb_model is not None:
                # Use ML models for prediction
                features = self._prepare_ml_features(current_state)
                prediction = self._predict_with_ml(features)
                return float(prediction)
            else:
                # Fallback to rule-based prediction
                return float(self._predict_with_rules(current_state))
        except Exception as e:
            print(f"Prediction error: {e}")
            return float(self._predict_with_rules(current_state))
    
    def _prepare_ml_features(self, current_state):
        """Prepare features for ML models"""
        # This is a simplified version - in production you'd extract more features
        features = np.array([
            current_state['hour'],
            current_state['day_of_week'],
            current_state['month'],
            current_state.get('solar_kw', 0),
            current_state.get('wind_kw', 0),
            current_state.get('grid_price', 0.12),
            current_state.get('carbon_intensity', 350),
            current_state.get('battery_soc', 50)
        ]).reshape(1, -1)
        return features
    
    def _predict_with_ml(self, features):
        """Make prediction using ML models"""
        try:
            # Scale features if scaler exists
            if data_scaler is not None:
                scaled_features = data_scaler.transform(features)
            else:
                scaled_features = features
            
            # For LSTM, we need sequence data
            if forecast_model is not None:
                # Create a simple sequence by repeating current features
                lstm_input = np.repeat(scaled_features.reshape(1, 1, -1), 24, axis=1)
                lstm_pred = forecast_model.predict(lstm_input, verbose=0)[0][0]
            else:
                lstm_pred = np.mean(scaled_features[0, 0:3]) * 10  # Fallback
            
            if xgb_model is not None:
                xgb_pred = xgb_model.predict(features)[0]
            else:
                xgb_pred = np.mean(features[0, 0:3]) * 10  # Fallback
            
            # Hybrid prediction (weighted average)
            hybrid_pred = 0.6 * lstm_pred + 0.4 * xgb_pred
            
            return float(max(20, hybrid_pred * 1.1))  # Add some safety margin
            
        except Exception as e:
            print(f"ML prediction failed: {e}")
            return float(self._predict_with_rules({
                'hour': features[0][0],
                'day_of_week': features[0][1]
            }))
    
    def _predict_with_rules(self, current_state):
        """Rule-based demand prediction"""
        hour = current_state['hour']
        day_of_week = current_state.get('day_of_week', 0)
        
        # Base prediction
        if 0 <= hour < 6:
            prediction = 30  # Night
        elif 6 <= hour < 9:
            prediction = 60  # Morning
        elif 9 <= hour < 17:
            prediction = 80  # Day
        elif 17 <= hour < 22:
            prediction = 90  # Evening
        else:
            prediction = 40  # Late night
        
        # Weekend adjustment
        if day_of_week >= 5:
            prediction *= 0.7
        
        # Add some randomness
        prediction *= (1 + 0.1 * np.random.randn())
        
        return float(max(20, prediction))
    
    def optimize_allocation(self, current_state):
        """Optimize resource allocation - uses RL if available, otherwise heuristic"""
        try:
            if not SIMULATION_MODE and rl_agent is not None:
                return self._optimize_with_rl(current_state)
            else:
                return self._optimize_with_heuristic(current_state)
        except Exception as e:
            print(f"Optimization error: {e}")
            return self._optimize_with_heuristic(current_state)
    
    def _optimize_with_rl(self, current_state):
        """Optimize using RL agent"""
        try:
            # Prepare state for RL agent
            state = np.array([
                current_state['demand_kw'] / 100,
                current_state['solar_kw'] / 100,
                current_state['wind_kw'] / 100,
                self.battery_soc_kwh / self.battery_capacity_kwh,
                current_state['grid_price'],
                current_state['carbon_intensity'] / 500,
                current_state['hour'] / 24,
                current_state['day_of_week'] / 7
            ]).reshape(1, -1)
            
            # Get action from RL agent
            action = np.argmax(rl_agent.model.predict(state, verbose=0))
            
            # Decode action (simplified)
            battery_action = (action // 9) % 3 - 1
            grid_action = (action // 3) % 3 - 1
            
            return {
                'battery_charge_kw': float(max(0, battery_action * 20)),
                'battery_discharge_kw': float(max(0, -battery_action * 20)),
                'grid_import_kw': float(max(0, grid_action * 30)),
                'grid_export_kw': float(max(0, -grid_action * 30)),
                'renewable_curtailment': 0.0,
                'method': 'RL Optimization'
            }
        except Exception as e:
            print(f"RL optimization failed: {e}")
            return self._optimize_with_heuristic(current_state)
    
    def _optimize_with_heuristic(self, current_state):
        """Heuristic allocation strategy"""
        demand = current_state['demand_kw']
        solar = current_state['solar_kw']
        wind = current_state['wind_kw']
        price = current_state['grid_price']
        hour = current_state['hour']
        
        available_renewables = solar + wind
        net_demand = max(0, demand - available_renewables)
        
        # Smart heuristic rules
        if price > 0.12 and hour >= 7 and hour <= 19:  # Expensive peak hours
            # Use battery aggressively
            battery_discharge = min(30, net_demand, self.battery_soc_kwh)
            grid_import = max(0, net_demand - battery_discharge)
            battery_charge = 0
            grid_export = 0
        elif price < 0.10 and (hour <= 5 or hour >= 22):  # Cheap off-peak
            # Charge battery
            battery_charge = min(20, self.battery_capacity_kwh - self.battery_soc_kwh)
            battery_discharge = 0
            grid_import = net_demand + battery_charge
            grid_export = 0
        elif available_renewables > demand * 1.5:  # Excess renewables
            # Export to grid
            battery_charge = min(10, self.battery_capacity_kwh - self.battery_soc_kwh)
            grid_export = max(0, available_renewables - demand - battery_charge)
            grid_import = 0
            battery_discharge = 0
        else:  # Normal operation
            # Balance battery usage
            if self.battery_soc_kwh > self.battery_capacity_kwh * 0.7:  # High SOC
                battery_discharge = min(15, net_demand)
                battery_charge = 0
            elif self.battery_soc_kwh < self.battery_capacity_kwh * 0.3:  # Low SOC
                battery_charge = min(10, net_demand)
                battery_discharge = 0
            else:
                battery_discharge = min(10, net_demand)
                battery_charge = 0
            
            grid_import = max(0, net_demand - battery_discharge)
            grid_export = 0
        
        return {
            'battery_charge_kw': float(battery_charge),
            'battery_discharge_kw': float(battery_discharge),
            'grid_import_kw': float(grid_import),
            'grid_export_kw': float(grid_export),
            'renewable_curtailment': 0.0,
            'method': 'Smart Heuristic'
        }
    
    def run_simulation(self, user_inputs, simulation_hours=24):
        """Run complete microgrid simulation"""
        print(f"\n{'='*60}")
        print(f"Starting Simulation")
        print(f"Duration: {simulation_hours} hours")
        print(f"Battery Capacity: {user_inputs.get('battery_capacity', 200)} kWh")
        print(f"Strategy: {user_inputs.get('strategy', 'heuristic')}")
        print(f"{'='*60}")
        
        # Update simulator parameters
        self.battery_capacity_kwh = float(user_inputs.get('battery_capacity', 200))
        battery_soc = self.battery_capacity_kwh * 0.5  # Start at 50%
        
        # Generate synthetic data for simulation period
        simulation_data = self.generate_synthetic_data(simulation_hours)
        
        results = []
        metrics = {
            'total_cost': 0.0,
            'total_emissions_kg': 0.0,
            'renewable_utilization': 0.0,
            'battery_cycles': 0.0,
            'grid_dependence': 0.0,
            'peak_demand': 0.0,
            'renewable_penetration': 0.0
        }
        
        for i, row in simulation_data.iterrows():
            # Current state
            current_state = {
                'timestamp': row['timestamp'],
                'demand_kw': float(row['demand_kw']),
                'solar_kw': float(row['solar_generation_kw']),
                'wind_kw': float(row['wind_generation_kw']),
                'grid_price': float(row['grid_price_usd_per_kwh']),
                'carbon_intensity': float(row['carbon_intensity_g_per_kwh']),
                'hour': int(row['hour']),
                'day_of_week': int(row['day_of_week']),
                'month': int(row['month'])
            }
            
            # Get optimized allocation
            allocation = self.optimize_allocation(current_state)
            
            # Calculate power balance
            available_renewables = (current_state['solar_kw'] + current_state['wind_kw']) * \
                                 (1 - allocation['renewable_curtailment'])
            
            net_demand = max(0, current_state['demand_kw'] - available_renewables)
            
            # Update battery
            battery_power = allocation['battery_charge_kw'] - allocation['battery_discharge_kw']
            battery_soc += battery_power
            battery_soc = max(0, min(self.battery_capacity_kwh, battery_soc))
            battery_soc_percent = (battery_soc / self.battery_capacity_kwh) * 100
            
            # Calculate grid import/export
            grid_import = max(0, net_demand - allocation['battery_discharge_kw'])
            grid_export = allocation['grid_export_kw']
            
            # Calculate costs and emissions
            hour_cost = (grid_import * current_state['grid_price'] -
                        grid_export * current_state['grid_price'] * 0.8)  # Sell at 80% of buy price
            
            hour_emissions = grid_import * current_state['carbon_intensity'] / 1000  # Convert to kg
            
            # Update metrics
            metrics['total_cost'] += float(hour_cost)
            metrics['total_emissions_kg'] += float(hour_emissions)
            metrics['renewable_utilization'] += float(min(1.0, available_renewables / 
                                                   (current_state['solar_kw'] + current_state['wind_kw'] + 1e-6)))
            if battery_power != 0:
                metrics['battery_cycles'] += float(abs(battery_power) / (2 * self.battery_capacity_kwh))
            metrics['grid_dependence'] += float(grid_import / (current_state['demand_kw'] + 1e-6))
            metrics['peak_demand'] = max(float(metrics['peak_demand']), float(current_state['demand_kw']))
            metrics['renewable_penetration'] += float(available_renewables / (current_state['demand_kw'] + 1e-6))
            
            # Store results
            results.append({
                'timestamp': current_state['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'demand_kw': float(round(current_state['demand_kw'], 2)),
                'solar_kw': float(round(current_state['solar_kw'], 2)),
                'wind_kw': float(round(current_state['wind_kw'], 2)),
                'available_renewables_kw': float(round(available_renewables, 2)),
                'battery_soc_percent': float(round(battery_soc_percent, 1)),
                'battery_power_kw': float(round(battery_power, 2)),
                'grid_import_kw': float(round(grid_import, 2)),
                'grid_export_kw': float(round(grid_export, 2)),
                'grid_price': float(round(current_state['grid_price'], 3)),
                'hour_cost_usd': float(round(hour_cost, 3)),
                'hour_emissions_kg': float(round(hour_emissions, 3)),
                'allocation_method': allocation['method']
            })
            
            # Progress indicator
            if i % 4 == 0:
                print(f"  Hour {i+1:3d}/{simulation_hours}: "
                      f"Demand: {current_state['demand_kw']:5.1f} kW, "
                      f"Renewables: {available_renewables:5.1f} kW, "
                      f"Battery SOC: {battery_soc_percent:5.1f}%")
        
        # Calculate averages
        metrics['renewable_utilization'] = float(metrics['renewable_utilization'] / len(simulation_data))
        metrics['grid_dependence'] = float(metrics['grid_dependence'] / len(simulation_data))
        metrics['renewable_penetration'] = float(metrics['renewable_penetration'] / len(simulation_data))
        
        # Calculate eco-score (0-100)
        eco_score = 100 - (
            metrics['total_emissions_kg'] * 0.5 +
            metrics['grid_dependence'] * 30 +
            (1 - metrics['renewable_utilization']) * 20 +
            (1 - metrics['renewable_penetration']) * 15
        )
        metrics['eco_score'] = float(max(0, min(100, eco_score)))
        
        # Calculate cost savings (compared to baseline - all from grid)
        baseline_cost = float(sum(simulation_data['demand_kw'] * simulation_data['grid_price_usd_per_kwh']))
        metrics['cost_savings_usd'] = float(baseline_cost - metrics['total_cost'])
        if baseline_cost > 0:
            metrics['cost_savings_percent'] = float(max(0, (metrics['cost_savings_usd'] / baseline_cost) * 100))
        else:
            metrics['cost_savings_percent'] = 0.0
        
        print(f"\n{'='*60}")
        print(f"Simulation Complete!")
        print(f"Total Cost: ${metrics['total_cost']:.2f}")
        print(f"Cost Savings: ${metrics['cost_savings_usd']:.2f} ({metrics['cost_savings_percent']:.1f}%)")
        print(f"Emissions: {metrics['total_emissions_kg']:.1f} kg CO₂")
        print(f"Eco-Score: {metrics['eco_score']:.1f}/100")
        print(f"{'='*60}")
        
        return results, metrics

# Initialize simulator
simulator = MicrogridSimulator()

@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Run simulation with user inputs"""
    try:
        data = request.get_json()
        
        # Extract user inputs
        user_inputs = {
            'simulation_duration': int(data.get('duration', 24)),
            'battery_capacity': float(data.get('battery_capacity', 200)),
            'solar_capacity': float(data.get('solar_capacity', 100)),
            'wind_capacity': float(data.get('wind_capacity', 50)),
            'strategy': data.get('strategy', 'heuristic')
        }
        
        # Run simulation
        results, metrics = simulator.run_simulation(
            user_inputs, 
            simulation_hours=user_inputs['simulation_duration']
        )
        
        # Create visualizations
        charts = create_visualizations(results, metrics)
        
        # Convert all metrics to Python native types
        metrics_native = {k: float(v) for k, v in metrics.items()}
        
        response = {
            'success': True,
            'results': results[-min(24, len(results)):],  # Last 24 hours or less
            'metrics': metrics_native,
            'charts': charts,
            'summary': {
                'total_cost': f"${metrics['total_cost']:.2f}",
                'total_emissions': f"{metrics['total_emissions_kg']:.1f} kg",
                'eco_score': f"{metrics['eco_score']:.1f}/100",
                'cost_savings': f"${metrics['cost_savings_usd']:.2f} ({metrics['cost_savings_percent']:.1f}%)",
                'renewable_utilization': f"{metrics['renewable_utilization']*100:.1f}%",
                'renewable_penetration': f"{metrics['renewable_penetration']*100:.1f}%",
                'peak_demand': f"{metrics['peak_demand']:.1f} kW"
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Simulation failed. Please check the server logs.'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make demand prediction"""
    try:
        data = request.get_json()
        
        # Create input data structure
        input_data = {
            'demand_kw': float(data.get('current_demand', 60)),
            'solar_kw': float(data.get('solar_generation', 40)),
            'wind_kw': float(data.get('wind_generation', 20)),
            'grid_price': float(data.get('grid_price', 0.12)),
            'carbon_intensity': float(data.get('carbon_intensity', 350)),
            'hour': int(data.get('hour', datetime.now().hour)),
            'day_of_week': int(data.get('day_of_week', datetime.now().weekday())),
            'month': int(data.get('month', datetime.now().month))
        }
        
        # Make prediction
        prediction = simulator.predict_demand(input_data)
        
        return jsonify({
            'success': True,
            'prediction': float(round(prediction, 2)),
            'current_demand': input_data['demand_kw'],
            'change_percent': float(round(((prediction - input_data['demand_kw']) / input_data['demand_kw']) * 100, 1))
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/system_status')
def system_status():
    """Get current system status"""
    models_loaded = (
        forecast_model is not None and 
        xgb_model is not None and 
        rl_agent is not None and 
        data_scaler is not None
    )
    
    status = {
        'simulation_mode': SIMULATION_MODE,
        'models_loaded': bool(models_loaded),
        'battery_capacity': float(simulator.battery_capacity_kwh),
        'timestamp': datetime.now().isoformat(),
        'system_status': 'Active',
        'version': '1.0.0',
        'forecast_model_loaded': bool(forecast_model is not None),
        'xgb_model_loaded': bool(xgb_model is not None),
        'rl_agent_loaded': bool(rl_agent is not None),
        'data_scaler_loaded': bool(data_scaler is not None)
    }
    return jsonify(status)

def create_visualizations(results, metrics):
    """Create Plotly charts for visualization"""
    
    if not results:
        return {}
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    try:
        # Chart 1: Demand vs Supply
        fig1 = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Energy Demand vs Supply", "Grid Interaction"),
            vertical_spacing=0.15,
            shared_xaxes=True
        )
        
        # Demand and renewables
        fig1.add_trace(
            go.Scatter(x=df['timestamp'], y=df['demand_kw'], 
                      name="Demand", line=dict(color='#00b894', width=2),
                      mode='lines'),
            row=1, col=1
        )
        
        fig1.add_trace(
            go.Scatter(x=df['timestamp'], y=df['available_renewables_kw'], 
                      name="Renewables", line=dict(color='#0984e3', width=2),
                      fill='tozeroy', fillcolor='rgba(9, 132, 227, 0.2)'),
            row=1, col=1
        )
        
        # Grid import/export
        fig1.add_trace(
            go.Bar(x=df['timestamp'], y=df['grid_import_kw'],
                  name="Grid Import", marker_color='#fdcb6e'),
            row=2, col=1
        )
        
        fig1.add_trace(
            go.Bar(x=df['timestamp'], y=df['grid_export_kw'],
                  name="Grid Export", marker_color='#d63031'),
            row=2, col=1
        )
        
        fig1.update_layout(
            height=400,
            showlegend=True,
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6e6e6'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig1.update_xaxes(title_text="Time", row=2, col=1, gridcolor='rgba(255, 255, 255, 0.1)')
        fig1.update_yaxes(title_text="Power (kW)", row=1, col=1, gridcolor='rgba(255, 255, 255, 0.1)')
        fig1.update_yaxes(title_text="Power (kW)", row=2, col=1, gridcolor='rgba(255, 255, 255, 0.1)')
        
        # Chart 2: Battery SOC
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=df['timestamp'], y=df['battery_soc_percent'],
            name="Battery SOC",
            line=dict(color='#00b894', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 184, 148, 0.2)'
        ))
        
        fig2.update_layout(
            title="Battery State of Charge",
            xaxis_title="Time",
            yaxis_title="SOC (%)",
            yaxis_range=[0, 100],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6e6e6'),
            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
        )
        
        # Chart 3: Renewable Mix Pie Chart
        total_solar = float(df['solar_kw'].sum())
        total_wind = float(df['wind_kw'].sum())
        
        fig3 = go.Figure(data=[go.Pie(
            labels=['Solar', 'Wind'],
            values=[total_solar, total_wind],
            hole=0.4,
            marker_colors=['#FFD700', '#87CEEB'],
            textinfo='label+percent',
            hoverinfo='label+value+percent',
            textfont=dict(color='#e6e6e6')
        )])
        
        fig3.update_layout(
            title="Renewable Energy Mix",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6e6e6')
        )
        
        # Chart 4: Cost and Emissions
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig4.add_trace(
            go.Bar(x=df['timestamp'], y=df['hour_cost_usd'], 
                  name="Hourly Cost", marker_color='#6c5ce7'),
            secondary_y=False
        )
        
        fig4.add_trace(
            go.Scatter(x=df['timestamp'], y=df['hour_emissions_kg'].cumsum(), 
                      name="Cumulative Emissions", line=dict(color='#d63031', width=2)),
            secondary_y=True
        )
        
        fig4.update_layout(
            title="Cost and Emissions Over Time",
            xaxis_title="Time",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6e6e6'),
            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
        )
        
        fig4.update_yaxes(title_text="Cost ($)", secondary_y=False, gridcolor='rgba(255, 255, 255, 0.1)')
        fig4.update_yaxes(title_text="Emissions (kg CO₂)", secondary_y=True, gridcolor='rgba(255, 255, 255, 0.1)')
        
        # Convert figures to JSON
        charts = {
            'demand_supply': json.loads(plotly.io.to_json(fig1, cls=NumpyEncoder)),
            'battery_soc': json.loads(plotly.io.to_json(fig2, cls=NumpyEncoder)),
            'renewable_mix': json.loads(plotly.io.to_json(fig3, cls=NumpyEncoder)),
            'cost_emissions': json.loads(plotly.io.to_json(fig4, cls=NumpyEncoder))
        }
        
        return charts
        
    except Exception as e:
        print(f"Error creating charts: {e}")
        return {}

@app.route('/api/demo_data')
def demo_data():
    """Return demo data for initial page load"""
    try:
        # Generate some demo data
        demo_results = []
        now = datetime.now()
        
        for i in range(24):
            ts = now - timedelta(hours=23-i)
            demo_results.append({
                'timestamp': ts.strftime('%Y-%m-%d %H:%M'),
                'demand_kw': float(round(50 + 20 * np.sin(i * np.pi / 12) + 5 * np.random.randn(), 2)),
                'solar_kw': float(round(max(0, 60 * np.sin((i-6) * np.pi / 12) + 10 * np.random.randn()), 2)),
                'wind_kw': float(round(30 + 10 * np.sin(i * np.pi / 6) + 5 * np.random.randn(), 2)),
                'available_renewables_kw': float(round(40 + 20 * np.sin(i * np.pi / 6), 2)),
                'battery_soc_percent': float(round(50 + 20 * np.sin(i * np.pi / 6), 1)),
                'battery_power_kw': float(round(10 * np.sin(i * np.pi / 12), 2)),
                'grid_import_kw': float(round(20 + 10 * np.random.randn(), 2)),
                'grid_export_kw': float(round(5 * np.random.randn(), 2)),
                'grid_price': float(round(0.12 + 0.03 * np.sin(i * np.pi / 12), 3)),
                'hour_cost_usd': float(round(0.5 + 0.3 * np.random.randn(), 3)),
                'hour_emissions_kg': float(round(0.3 + 0.2 * np.random.randn(), 3)),
                'allocation_method': 'Demo Heuristic'
            })
        
        demo_metrics = {
            'total_cost': 24.5,
            'total_emissions_kg': 45.6,
            'eco_score': 78.5,
            'cost_savings_percent': 15.3,
            'renewable_utilization': 0.85,
            'renewable_penetration': 0.62,
            'grid_dependence': 0.35,
            'peak_demand': 95.2,
            'battery_cycles': 0.12,
            'cost_savings_usd': 4.3
        }
        
        return jsonify({
            'success': True,
            'results': demo_results,
            'metrics': demo_metrics,
            'summary': {
                'total_cost': '$24.50',
                'total_emissions': '45.6 kg',
                'eco_score': '78.5/100',
                'cost_savings': '$4.30 (15.3%)',
                'renewable_utilization': '85.0%',
                'renewable_penetration': '62.0%',
                'peak_demand': '95.2 kW'
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Energy-Aware Microgrid Management System")
    print("="*60)
    print(f"Simulation Mode: {'ON' if SIMULATION_MODE else 'OFF'}")
    print(f"Dashboard available at: http://localhost:5000")
    print("="*60)
    
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Save the HTML template
    html_template = '''
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Energy-Aware Microgrid Resource Allocation</title>
    
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <!-- Plotly -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    
    <!-- Custom CSS -->
    <style>
        :root {
            --primary-color: #00b894;
            --secondary-color: #0984e3;
            --danger-color: #d63031;
            --warning-color: #fdcb6e;
            --dark-bg: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #e6e6e6;
        }
        
        body {
            background-color: var(--dark-bg);
            color: var(--text-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .navbar {
            background-color: var(--card-bg) !important;
            border-bottom: 2px solid var(--primary-color);
        }
        
        .card {
            background-color: var(--card-bg);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }
        
        .card-header {
            background-color: rgba(0, 0, 0, 0.2);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            font-weight: 600;
        }
        
        .btn-primary {
            background-color: var(--primary-color);
            border-color: var(--primary-color);
        }
        
        .btn-primary:hover {
            background-color: #00a085;
            border-color: #00a085;
        }
        
        .metric-card {
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .eco-score-excellent { color: #00b894; }
        .eco-score-good { color: #fdcb6e; }
        .eco-score-poor { color: #d63031; }
        
        .gauge-container {
            width: 100%;
            height: 200px;
        }
        
        .chart-container {
            height: 400px;
            width: 100%;
        }
        
        .simulation-controls {
            background: linear-gradient(135deg, var(--card-bg) 0%, #1a1a2e 100%);
            border: 1px solid var(--primary-color);
        }
        
        .form-range::-webkit-slider-thumb {
            background-color: var(--primary-color);
        }
        
        .form-range::-moz-range-thumb {
            background-color: var(--primary-color);
        }
        
        .form-control, .form-select {
            background-color: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
            color: var(--text-color);
        }
        
        .form-control:focus, .form-select:focus {
            background-color: rgba(255, 255, 255, 0.15);
            border-color: var(--primary-color);
            color: var(--text-color);
            box-shadow: 0 0 0 0.25rem rgba(0, 184, 148, 0.25);
        }
        
        .table {
            color: var(--text-color);
        }
        
        .table-dark {
            --bs-table-bg: var(--card-bg);
            --bs-table-striped-bg: rgba(255, 255, 255, 0.05);
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-active { background-color: var(--primary-color); }
        .status-warning { background-color: var(--warning-color); }
        .status-inactive { background-color: var(--danger-color); }
    </style>
</head>
<body>
    <!-- Navigation Bar -->
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="fas fa-solar-panel me-2"></i>
                Energy-Aware Microgrid Manager
            </a>
            <div class="d-flex">
                <span class="navbar-text me-3">
                    <span class="status-indicator status-active"></span>
                    System Active
                </span>
                <span id="current-time" class="navbar-text"></span>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <!-- Header -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-body">
                        <h1 class="h3 mb-2">Energy-Aware Resource Allocation in Microgrids</h1>
                        <p class="mb-0 text-muted">
                            Machine Learning & Reinforcement Learning based optimization for efficient energy distribution
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <!-- Left Column: Controls and Metrics -->
            <div class="col-lg-4">
                <!-- Simulation Controls -->
                <div class="card simulation-controls">
                    <div class="card-header">
                        <i class="fas fa-sliders-h me-2"></i>Simulation Controls
                    </div>
                    <div class="card-body">
                        <form id="simulation-form">
                            <div class="mb-3">
                                <label class="form-label">Simulation Duration</label>
                                <input type="range" class="form-range" id="duration" min="1" max="168" value="24">
                                <div class="d-flex justify-content-between">
                                    <small>1 hour</small>
                                    <span id="duration-value">24 hours</span>
                                    <small>1 week</small>
                                </div>
                            </div>
                            
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label class="form-label">Battery Capacity (kWh)</label>
                                    <input type="number" class="form-control" id="battery-capacity" value="200" min="50" max="1000">
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Solar Capacity (kW)</label>
                                    <input type="number" class="form-control" id="solar-capacity" value="100" min="10" max="500">
                                </div>
                            </div>
                            
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label class="form-label">Wind Capacity (kW)</label>
                                    <input type="number" class="form-control" id="wind-capacity" value="50" min="10" max="300">
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Optimization Strategy</label>
                                    <select class="form-select" id="strategy">
                                        <option value="rl">Reinforcement Learning</option>
                                        <option value="heuristic" selected>Heuristic Rules</option>
                                        <option value="greedy">Cost Minimization</option>
                                    </select>
                                </div>
                            </div>
                            
                            <div class="d-grid gap-2">
                                <button type="submit" class="btn btn-primary btn-lg" id="run-simulation">
                                    <i class="fas fa-play me-2"></i>Run Simulation
                                </button>
                                <button type="button" class="btn btn-outline-primary" id="predict-demand">
                                    <i class="fas fa-chart-line me-2"></i>Predict Next Hour Demand
                                </button>
                            </div>
                        </form>
                    </div>
                </div>

                <!-- Key Metrics -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-chart-bar me-2"></i>Performance Metrics
                    </div>
                    <div class="card-body">
                        <div class="row" id="metrics-container">
                            <div class="col-6">
                                <div class="metric-card" style="background: rgba(0, 184, 148, 0.1);">
                                    <div class="metric-label">Eco-Score</div>
                                    <div class="metric-value eco-score-excellent" id="eco-score">--</div>
                                    <small>Sustainability Rating</small>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="metric-card" style="background: rgba(253, 203, 110, 0.1);">
                                    <div class="metric-label">Cost Savings</div>
                                    <div class="metric-value" id="cost-savings">--</div>
                                    <small>vs Baseline</small>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="metric-card" style="background: rgba(9, 132, 227, 0.1);">
                                    <div class="metric-label">Total Cost</div>
                                    <div class="metric-value" id="total-cost">--</div>
                                    <small>Simulation Period</small>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="metric-card" style="background: rgba(214, 48, 49, 0.1);">
                                    <div class="metric-label">Emissions</div>
                                    <div class="metric-value" id="total-emissions">--</div>
                                    <small>CO₂ Equivalent</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Battery Status -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-battery-three-quarters me-2"></i>Battery Status
                    </div>
                    <div class="card-body">
                        <div class="gauge-container" id="battery-gauge"></div>
                        <div class="text-center mt-3">
                            <div class="row">
                                <div class="col-6">
                                    <small>Capacity</small>
                                    <div class="h5" id="battery-capacity-display">200 kWh</div>
                                </div>
                                <div class="col-6">
                                    <small>Current SOC</small>
                                    <div class="h5" id="battery-soc-display">50%</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Column: Charts and Results -->
            <div class="col-lg-8">
                <!-- Energy Demand vs Supply Chart -->
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <div>
                            <i class="fas fa-bolt me-2"></i>Energy Demand vs Supply
                        </div>
                        <div class="btn-group btn-group-sm">
                            <button class="btn btn-outline-primary active" data-chart="demand">Demand</button>
                            <button class="btn btn-outline-primary" data-chart="renewables">Renewables</button>
                            <button class="btn btn-outline-primary" data-chart="grid">Grid</button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="chart-container" id="demand-supply-chart"></div>
                    </div>
                </div>

                <!-- Renewable Energy Mix -->
                <div class="row">
                    <div class="col-md-6">
                        <div class="card h-100">
                            <div class="card-header">
                                <i class="fas fa-leaf me-2"></i>Renewable Energy Mix
                            </div>
                            <div class="card-body">
                                <div class="chart-container" id="renewable-mix-chart"></div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Cost and Emissions -->
                    <div class="col-md-6">
                        <div class="card h-100">
                            <div class="card-header">
                                <i class="fas fa-dollar-sign me-2"></i>Cost & Emissions
                            </div>
                            <div class="card-body">
                                <div class="chart-container" id="cost-emissions-chart"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Recent Allocation Results -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-table me-2"></i>Recent Allocation Decisions
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-dark table-hover" id="results-table">
                                <thead>
                                    <tr>
                                        <th>Time</th>
                                        <th>Demand (kW)</th>
                                        <th>Renewables (kW)</th>
                                        <th>Battery (kW)</th>
                                        <th>Grid (kW)</th>
                                        <th>Cost ($)</th>
                                        <th>Method</th>
                                    </tr>
                                </thead>
                                <tbody id="results-body">
                                    <tr>
                                        <td colspan="7" class="text-center">No simulation data yet. Run a simulation to see results.</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- System Status -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-microchip me-2"></i>System Status & Models
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <div class="text-center">
                                    <div class="display-6 mb-2">
                                        <i class="fas fa-brain text-primary"></i>
                                    </div>
                                    <div class="h6">LSTM Model</div>
                                    <small class="text-success">
                                        <span class="status-indicator status-active"></span> Active
                                    </small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <div class="display-6 mb-2">
                                        <i class="fas fa-project-diagram text-warning"></i>
                                    </div>
                                    <div class="h6">XGBoost Model</div>
                                    <small class="text-success">
                                        <span class="status-indicator status-active"></span> Active
                                    </small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <div class="display-6 mb-2">
                                        <i class="fas fa-robot text-info"></i>
                                    </div>
                                    <div class="h6">RL Agent</div>
                                    <small class="text-success">
                                        <span class="status-indicator status-active"></span> Active
                                    </small>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <div class="display-6 mb-2">
                                        <i class="fas fa-database text-success"></i>
                                    </div>
                                    <div class="h6">Data Pipeline</div>
                                    <small class="text-success">
                                        <span class="status-indicator status-active"></span> Active
                                    </small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer class="footer mt-5 py-3" style="background-color: var(--card-bg); border-top: 1px solid rgba(255, 255, 255, 0.1);">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <p class="mb-0">Energy-Aware Microgrid Resource Allocation System</p>
                    <small class="text-muted">Machine Learning & Reinforcement Learning Project</small>
                </div>
                <div class="col-md-6 text-end">
                    <small class="text-muted">Simulation Time: <span id="simulation-timestamp">--</span></small>
                </div>
            </div>
        </div>
    </footer>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Custom JavaScript -->
    <script>
        // Update current time
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = 
                now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        }
        setInterval(updateTime, 1000);
        updateTime();

        // Update simulation duration display
        const durationSlider = document.getElementById('duration');
        const durationValue = document.getElementById('duration-value');
        
        durationSlider.addEventListener('input', function() {
            durationValue.textContent = `${this.value} hours`;
        });

        // Battery gauge
        function createBatteryGauge(soc) {
            const data = [{
                type: "indicator",
                mode: "gauge+number",
                value: soc,
                title: { text: "State of Charge" },
                gauge: {
                    axis: { range: [0, 100] },
                    bar: { color: "#00b894" },
                    steps: [
                        { range: [0, 20], color: "#d63031" },
                        { range: [20, 50], color: "#fdcb6e" },
                        { range: [50, 100], color: "#00b894" }
                    ],
                    threshold: {
                        line: { color: "red", width: 4 },
                        thickness: 0.75,
                        value: 20
                    }
                }
            }];
            
            const layout = {
                margin: { t: 30, r: 30, l: 30, b: 30 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e6e6e6' }
            };
            
            Plotly.newPlot('battery-gauge', data, layout, {responsive: true});
        }

        // Initialize battery gauge
        createBatteryGauge(50);

        // Run simulation
        document.getElementById('simulation-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const runButton = document.getElementById('run-simulation');
            const originalText = runButton.innerHTML;
            runButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Running...';
            runButton.disabled = true;
            
            try {
                const simulationData = {
                    duration: durationSlider.value,
                    battery_capacity: document.getElementById('battery-capacity').value,
                    solar_capacity: document.getElementById('solar-capacity').value,
                    wind_capacity: document.getElementById('wind-capacity').value,
                    strategy: document.getElementById('strategy').value
                };
                
                const response = await fetch('/api/simulate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(simulationData)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    updateDashboard(result);
                    showNotification('Simulation completed successfully!', 'success');
                } else {
                    throw new Error(result.error || 'Simulation failed');
                }
            } catch (error) {
                console.error('Simulation error:', error);
                showNotification(`Error: ${error.message}`, 'danger');
            } finally {
                runButton.innerHTML = originalText;
                runButton.disabled = false;
            }
        });

        // Predict demand
        document.getElementById('predict-demand').addEventListener('click', async function() {
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        current_demand: 60,
                        solar_generation: 40,
                        wind_generation: 20,
                        grid_price: 0.12,
                        carbon_intensity: 350,
                        hour: new Date().getHours(),
                        day_of_week: new Date().getDay(),
                        month: new Date().getMonth() + 1
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    showNotification(
                        `Next hour demand prediction: ${result.prediction.toFixed(1)} kW ` +
                        `(${result.change_percent > 0 ? '+' : ''}${result.change_percent.toFixed(1)}%)`,
                        'info'
                    );
                }
            } catch (error) {
                console.error('Prediction error:', error);
            }
        });

        // Update dashboard with simulation results
        function updateDashboard(data) {
            // Update metrics
            document.getElementById('eco-score').textContent = data.summary.eco_score;
            document.getElementById('cost-savings').textContent = data.summary.cost_savings;
            document.getElementById('total-cost').textContent = data.summary.total_cost;
            document.getElementById('total-emissions').textContent = data.summary.total_emissions;
            
            // Update battery display
            const lastResult = data.results[data.results.length - 1];
            if (lastResult) {
                document.getElementById('battery-soc-display').textContent = 
                    `${lastResult.battery_soc_percent.toFixed(1)}%`;
                createBatteryGauge(lastResult.battery_soc_percent);
            }
            
            // Update charts
            updateCharts(data.charts);
            
            // Update results table
            updateResultsTable(data.results);
            
            // Update timestamp
            document.getElementById('simulation-timestamp').textContent = 
                new Date().toLocaleString();
            
            // Update eco-score color
            const ecoScore = parseFloat(data.summary.eco_score);
            const ecoScoreElement = document.getElementById('eco-score');
            ecoScoreElement.className = 'metric-value ';
            if (ecoScore >= 80) {
                ecoScoreElement.classList.add('eco-score-excellent');
            } else if (ecoScore >= 60) {
                ecoScoreElement.classList.add('eco-score-good');
            } else {
                ecoScoreElement.classList.add('eco-score-poor');
            }
        }

        // Update charts
        function updateCharts(charts) {
            if (charts.demand_supply) {
                Plotly.newPlot('demand-supply-chart', charts.demand_supply.data, 
                    charts.demand_supply.layout, {responsive: true});
            }
            
            if (charts.renewable_mix) {
                Plotly.newPlot('renewable-mix-chart', charts.renewable_mix.data, 
                    charts.renewable_mix.layout, {responsive: true});
            }
            
            if (charts.cost_emissions) {
                Plotly.newPlot('cost-emissions-chart', charts.cost_emissions.data, 
                    charts.cost_emissions.layout, {responsive: true});
            }
        }

        // Update results table
        function updateResultsTable(results) {
            const tbody = document.getElementById('results-body');
            tbody.innerHTML = '';
            
            // Show only last 10 results
            const displayResults = results.slice(-10);
            
            if (displayResults.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="text-center">No simulation data available</td></tr>';
                return;
            }
            
            displayResults.forEach(result => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${result.timestamp.split(' ')[1]}</td>
                    <td>${result.demand_kw.toFixed(1)}</td>
                    <td>${result.available_renewables_kw.toFixed(1)}</td>
                    <td>${result.battery_power_kw.toFixed(1)}</td>
                    <td>${result.grid_import_kw.toFixed(1)}</td>
                    <td>$${result.hour_cost_usd.toFixed(2)}</td>
                    <td><span class="badge bg-primary">${result.allocation_method}</span></td>
                `;
                tbody.appendChild(row);
            });
        }

        // Show notification
        function showNotification(message, type) {
            // Create notification element
            const notification = document.createElement('div');
            notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
            notification.style.cssText = `
                top: 20px;
                right: 20px;
                z-index: 9999;
                min-width: 300px;
            `;
            notification.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            
            // Add to page
            document.body.appendChild(notification);
            
            // Auto remove after 5 seconds
            setTimeout(() => {
                notification.remove();
            }, 5000);
        }

        // Chart filter buttons
        document.querySelectorAll('[data-chart]').forEach(button => {
            button.addEventListener('click', function() {
                // Remove active class from all buttons
                document.querySelectorAll('[data-chart]').forEach(btn => {
                    btn.classList.remove('active');
                });
                
                // Add active class to clicked button
                this.classList.add('active');
                
                // Here you would filter the chart data
                // This is a placeholder for actual filtering logic
            });
        });

        // Check system status on load
        async function checkSystemStatus() {
            try {
                const response = await fetch('/api/system_status');
                const status = await response.json();
                
                if (!status.models_loaded) {
                    showNotification('Some ML models are not loaded. Using simulation mode only.', 'warning');
                }
            } catch (error) {
                console.error('Status check error:', error);
            }
        }

        // Load demo data on page load
        async function loadDemoData() {
            try {
                const response = await fetch('/api/demo_data');
                const data = await response.json();
                
                if (data.success) {
                    updateDashboard(data);
                }
            } catch (error) {
                console.error('Error loading demo data:', error);
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            checkSystemStatus();
            loadDemoData();
        });
    </script>
</body>
</html>
'''
    
    # Save the HTML template to templates folder
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("✓ HTML template created in templates/index.html")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)