import yfinance as yf
import numpy as np
import pandas as pd

class YahooFinanceScraper:
    def __init__(self, tickers=['VOO', 'BND', 'VXUS', 'GLD']):
        self.tickers = tickers
        
    def get_historical_data(self, period="10y"):
        """Get historical market data for all tickers"""
        data = yf.download(
            tickers=self.tickers,
            period=period,
            interval="1d",
            group_by='ticker',
            auto_adjust=True
        )
        return data

    def process_market_data(self, historical_data):
        """Calculate expected returns and covariance matrix"""
        closes = pd.DataFrame()
        
        for ticker in self.tickers:
            closes[ticker] = historical_data[ticker]['Close']
            
        returns = np.log(closes / closes.shift(1)).dropna()
        
        return {
            'expected_returns': returns.mean().values,
            'cov_matrix': returns.cov().values,
            'tickers': self.tickers,
            'historical_prices': closes
        }

    def get_current_prices(self):
        """Get real-time price data"""
        data = yf.download(
            tickers=self.tickers,
            period="1d",
            interval="1m",
            prepost=True
        )
        return data.iloc[-1]['Close'].values

# Example usage:
if __name__ == "__main__":
    scraper = YahooFinanceScraper()
    hist_data = scraper.get_historical_data()
    market_data = scraper.process_market_data(hist_data)
    print("Expected Returns:", market_data['expected_returns'])
    print("Covariance Matrix:\n", market_data['cov_matrix'])
