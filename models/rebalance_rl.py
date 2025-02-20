import torch
from stable_baselines3 import PPO
import numpy as np

class PortfolioRebalanceEnv:
    def __init__(self, initial_portfolio, asset_data):
        self.portfolio = initial_portfolio
        self.asset_data = asset_data
        self.current_step = 0
        
    def step(self, action):
        # Execute rebalancing action
        self.portfolio = self._apply_action(action)
        self.current_step += 1
        
        # Calculate reward (Sharpe ratio)
        reward = self._calculate_reward()
        done = self.current_step >= len(self.asset_data) - 1
        
        return self._get_state(), reward, done, {}
    
    def _apply_action(self, action):
        # Normalize action to sum to 1
        return torch.softmax(torch.tensor(action), dim=0).numpy()
    
    def _calculate_reward(self):
        # Calculate portfolio Sharpe ratio
        returns = np.dot(self.portfolio, self.asset_data[self.current_step])
        volatility = np.sqrt(np.dot(self.portfolio.T, np.dot(self.cov_matrix, self.portfolio)))
        return (returns - self.risk_free_rate) / volatility

class PPOTrainer:
    def __init__(self, env_config):
        self.env = PortfolioRebalanceEnv(**env_config)
        self.model = PPO('MlpPolicy', self.env, verbose=1)
        
    def train(self, timesteps=10000):
        self.model.learn(total_timesteps=timesteps)
        self.model.save("portfolio_rebalance_ppo") 