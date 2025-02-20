import torch
import numpy as np
from models.mpt_optimizer import MPTOptimizer

class RoboAdvisor:
    def __init__(self, classifier_path='models/classifier.pth', 
                 mpt_optimizer=None, rl_model_path=None):
        self.classifier = torch.jit.load(classifier_path)
        self.mpt_optimizer = mpt_optimizer
        self.rl_model = self._load_rl_model(rl_model_path) if rl_model_path else None
        
    def preprocess_input(self, user_data):
        """Normalize user input features"""
        features = np.array([
            user_data['age'],
            user_data['income'],
            user_data['savings'],
            user_data['risk_tolerance'],
            user_data['investment_goal']
        ])
        return torch.FloatTensor(features).unsqueeze(0)
    
    def generate_portfolio(self, user_data, market_data):
        with torch.no_grad():
            # Classify user risk profile
            features = self.preprocess_input(user_data)
            risk_profile = self.classifier(features).argmax().item()
            
            # Get MPT-optimized allocation
            if self.mpt_optimizer:
                allocation = self.mpt_optimizer.max_sharpe_ratio(
                    expected_returns=market_data['expected_returns'],
                    cov_matrix=market_data['cov_matrix']
                )
            else:
                allocation = self.fallback_allocation(risk_profile)
                
            # Apply RL rebalancing if available
            if self.rl_model:
                allocation = self.apply_rebalancing(allocation, market_data)
                
        return {
            'risk_profile': ['Low', 'Medium', 'High'][risk_profile],
            'allocation': dict(zip(self.mpt_optimizer.asset_tickers, allocation)),
            'statistics': self.calculate_statistics(allocation, market_data)
        }
    
    def calculate_statistics(self, allocation, market_data):
        returns = np.dot(allocation, market_data['expected_returns'])
        volatility = np.sqrt(np.dot(allocation.T, np.dot(market_data['cov_matrix'], allocation)))
        return {
            'expected_return': returns,
            'volatility': volatility,
            'sharpe_ratio': (returns - self.mpt_optimizer.risk_free_rate) / volatility
        }

    def _load_rl_model(self, rl_model_path):
        # Implementation of _load_rl_model method
        pass

    def apply_rebalancing(self, allocation, market_data):
        # Implementation of apply_rebalancing method
        pass

    def fallback_allocation(self, risk_profile):
        # Implementation of fallback_allocation method
        pass 