import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FinancialDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic user data for testing"""
        np.random.seed(42)
        data = {
            'age': np.random.randint(20, 70, num_samples),
            'income': np.random.exponential(scale=50000, size=num_samples),
            'savings': np.random.lognormal(mean=10, sigma=1, size=num_samples),
            'risk_tolerance': np.random.choice([1, 2, 3], size=num_samples),
            'investment_goal': np.random.choice([0, 1, 2], size=num_samples),  # 0=conservative, 1=moderate, 2=aggressive
            'portfolio_returns': np.random.uniform(-0.2, 0.3, num_samples)
        }
        return pd.DataFrame(data)
    
    def preprocess(self, df):
        """Preprocess financial data"""
        # Feature engineering
        df['net_worth'] = df['income'] + df['savings']
        features = df[['age', 'income', 'savings', 'risk_tolerance', 'investment_goal']]
        
        # Normalize features
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features, df['portfolio_returns'].values 