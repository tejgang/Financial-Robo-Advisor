import torch
import cvxpy as cp

class MPTOptimizer:
    def __init__(self, expected_returns, covariance_matrix, risk_free_rate=0.02):
        self.returns = expected_returns
        self.cov_matrix = covariance_matrix
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(expected_returns)
        
    def efficient_frontier(self, target_return=None):
        """Calculate efficient frontier weights"""
        weights = cp.Variable(self.num_assets)
        expected_return = self.returns @ weights
        risk = cp.quad_form(weights, self.cov_matrix)
        
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        if target_return:
            constraints.append(expected_return >= target_return)
            
        problem = cp.Problem(cp.Minimize(risk), constraints)
        problem.solve()
        return weights.value
    
    def max_sharpe_ratio(self):
        """Calculate maximum Sharpe ratio portfolio"""
        excess_returns = self.returns - self.risk_free_rate
        weights = cp.Variable(self.num_assets)
        
        sharpe = (excess_returns @ weights) / cp.quad_form(weights, self.cov_matrix)
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        problem = cp.Problem(cp.Maximize(sharpe), constraints)
        problem.solve()
        return weights.value 