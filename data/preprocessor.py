import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .market_data import YahooFinanceScraper

class FinancialDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.market_scraper = YahooFinanceScraper()
        self.asset_tickers = self.market_scraper.tickers + ['CASH']
    
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic user data with market data"""
        np.random.seed(42)
        data = {
            'age': np.random.randint(20, 70, num_samples),
            'income': np.random.exponential(scale=50000, size=num_samples),
            'savings': np.random.lognormal(mean=10, sigma=1, size=num_samples),
            'risk_tolerance': np.random.choice([1, 2, 3], size=num_samples),
            'investment_horizon': np.random.choice([1, 3, 5, 10], size=num_samples),
            'goal': np.random.choice([0, 1, 2], size=num_samples),  # 0=retirement, 1=wealth, 2=house
            'historical_returns': np.random.uniform(-0.1, 0.2, num_samples),
            'volatility': np.random.uniform(0.05, 0.3, num_samples)
        }
        
        # Generate synthetic historical returns for assets
        for ticker in self.asset_tickers:
            data[ticker] = np.random.normal(loc=0.08, scale=0.15, size=num_samples)
            
        return pd.DataFrame(data)
    
    def preprocess(self, df):
        """Preprocess financial data for MPT"""
        # User features
        user_features = df[['age', 'income', 'savings', 'risk_tolerance', 
                           'investment_horizon', 'goal', 'historical_returns', 'volatility']]
        
        # Asset returns data
        asset_returns = df[self.asset_tickers]
        
        # Normalize features
        scaled_features = self.scaler.fit_transform(user_features)
        return scaled_features, asset_returns.values 

    def get_market_data(self):
        hist_data = self.market_scraper.get_historical_data()
        return self.market_scraper.process_market_data(hist_data) 