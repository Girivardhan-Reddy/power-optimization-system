"""
Synthetic Energy Data Generator for Microgrid Simulation
Generates 1 year of hourly data with realistic patterns
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class EnergyDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_timestamps(self, start_date="2023-01-01", n_hours=8760):
        """Generate hourly timestamps for one year"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        timestamps = [start + timedelta(hours=i) for i in range(n_hours)]
        return timestamps
    
    def generate_solar(self, timestamps, latitude=40.0):
        """Generate solar generation with daily/seasonal patterns"""
        solar = []
        for ts in timestamps:
            hour = ts.hour
            day_of_year = ts.timetuple().tm_yday
            month = ts.month
            
            # Base solar pattern (sinusoidal)
            base = max(0, np.sin((hour - 6) * np.pi / 12))  # Daylight hours
            
            # Seasonal adjustment (more sun in summer)
            seasonal = 1 + 0.3 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
            
            # Cloud cover randomness
            cloud = 1 - 0.3 * np.random.beta(2, 2)
            
            # Capacity factor
            capacity = 100  # kW
            
            generation = capacity * base * seasonal * cloud * (1 + 0.1 * np.random.randn())
            solar.append(max(0, generation))
        return solar
    
    def generate_wind(self, timestamps):
        """Generate wind generation with Weibull distribution"""
        wind = []
        for ts in timestamps:
            hour = ts.hour
            month = ts.month
            
            # Base wind speed (higher at night and in winter)
            base_speed = 5 + 2 * np.sin(hour * np.pi / 12) + (2 if month in [11, 12, 1, 2] else 0)
            
            # Add randomness
            speed = max(0, base_speed + 3 * np.random.randn())
            
            # Wind turbine power curve (simplified)
            if speed < 3:
                power = 0
            elif speed < 12:
                power = 50 * ((speed - 3) / 9) ** 3
            elif speed <= 25:
                power = 50
            else:
                power = 0
            
            wind.append(power)
        return wind
    
    def generate_demand(self, timestamps):
        """Generate energy demand with daily/weekly patterns"""
        demand = []
        for ts in timestamps:
            hour = ts.hour
            day_of_week = ts.weekday()
            month = ts.month
            
            # Base daily pattern (higher during daytime)
            daily = 50 + 30 * np.sin((hour - 12) * np.pi / 12)
            
            # Weekend adjustment
            if day_of_week >= 5:  # Weekend
                daily *= 0.7
            
            # Seasonal adjustment (higher in winter/summer for HVAC)
            if month in [12, 1, 2]:  # Winter
                seasonal = 1.3
            elif month in [6, 7, 8]:  # Summer
                seasonal = 1.2
            else:
                seasonal = 1.0
            
            # Random fluctuations
            noise = 0.1 * np.random.randn()
            
            total = daily * seasonal * (1 + noise) + 5 * np.random.randn()
            demand.append(max(20, total))
        return demand
    
    def generate_grid_price(self, timestamps):
        """Generate time-varying electricity prices"""
        prices = []
        for ts in timestamps:
            hour = ts.hour
            day_of_week = ts.weekday()
            
            # Time-of-Use pricing
            if 7 <= hour <= 19:  # Peak hours
                base = 0.15  # $/kWh
            else:  # Off-peak
                base = 0.08
            
            # Weekend adjustment
            if day_of_week >= 5:
                base *= 0.9
            
            # Random variations
            price = base * (1 + 0.1 * np.random.randn())
            prices.append(max(0.05, price))
        return prices
    
    def generate_carbon_intensity(self, timestamps):
        """Generate carbon intensity based on time and renewables"""
        intensity = []
        for ts in timestamps:
            hour = ts.hour
            month = ts.month
            
            # Base intensity (higher during peak, lower at night)
            base = 400 - 100 * np.sin((hour - 3) * np.pi / 12)
            
            # Seasonal variation
            if month in [12, 1, 2]:  # More fossil fuels in winter
                base += 50
            
            # Random component
            carbon = base * (1 + 0.15 * np.random.randn())
            intensity.append(max(200, carbon))
        return intensity
    
    def generate_battery_soc(self, timestamps, initial_soc=50):
        """Generate battery state of charge with charging/discharging cycles"""
        soc = [initial_soc]
        for i in range(1, len(timestamps)):
            hour = timestamps[i].hour
            
            # Charge during cheap hours, discharge during expensive hours
            if 1 <= hour <= 5:  # Late night charging
                change = 5
            elif 17 <= hour <= 20:  # Evening discharge
                change = -7
            else:
                change = np.random.choice([-2, -1, 0, 1, 2])
            
            new_soc = soc[-1] + change
            new_soc = max(0, min(100, new_soc))
            soc.append(new_soc)
        return soc
    
    def generate_dataset(self):
        """Generate complete dataset"""
        print("Generating synthetic energy data...")
        
        timestamps = self.generate_timestamps()
        
        data = {
            'timestamp': timestamps,
            'solar_generation_kw': self.generate_solar(timestamps),
            'wind_generation_kw': self.generate_wind(timestamps),
            'demand_kw': self.generate_demand(timestamps),
            'grid_price_usd_per_kwh': self.generate_grid_price(timestamps),
            'carbon_intensity_g_per_kwh': self.generate_carbon_intensity(timestamps),
            'battery_soc_percent': self.generate_battery_soc(timestamps)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate net demand
        df['renewable_generation_kw'] = df['solar_generation_kw'] + df['wind_generation_kw']
        df['net_demand_kw'] = df['demand_kw'] - df['renewable_generation_kw']
        df['net_demand_kw'] = df['net_demand_kw'].apply(lambda x: max(0, x))
        
        # Add datetime features for ML
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        print(f"Generated {len(df)} records")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def save_dataset(self, df, filename='energy_data.csv'):
        """Save dataset to CSV"""
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        return df

if __name__ == "__main__":
    generator = EnergyDataGenerator()
    df = generator.generate_dataset()
    generator.save_dataset(df)
    
    # Display sample
    print("\nSample data:")
    print(df.head())
    print("\nDataset statistics:")
    print(df.describe())