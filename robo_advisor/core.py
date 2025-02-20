import torch
import numpy as np

class RoboAdvisor:
    def __init__(self, classifier_path='models/classifier.pth', 
                 optimizer_path='models/optimizer.pth'):
        self.classifier = torch.jit.load(classifier_path)
        self.optimizer = torch.jit.load(optimizer_path)
        self.classifier.eval()
        self.optimizer.eval()
        
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
    
    def generate_portfolio(self, user_data):
        with torch.no_grad():
            features = self.preprocess_input(user_data)
            risk_profile = self.classifier(features).argmax().item()
            
            # Get optimal asset allocation
            allocation = self.optimizer(features).numpy().flatten()
            
        return {
            'risk_profile': ['Low', 'Medium', 'High'][risk_profile],
            'allocation': {
                'stocks': allocation[0],
                'bonds': allocation[1],
                'crypto': allocation[2],
                'real_estate': allocation[3],
                'cash': allocation[4]
            }
        } 