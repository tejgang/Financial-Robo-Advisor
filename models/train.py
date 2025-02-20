import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .classifier import RiskProfileClassifier
from .optimizer import PortfolioOptimizer

class ModelTrainer:
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def train_classifier(self, X, y, epochs=100, batch_size=32):
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = RiskProfileClassifier().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        return model
    
    def train_optimizer(self, X, y, epochs=200, batch_size=32):
        """
        Train the portfolio optimizer
        X: feature matrix
        y: target allocations (num_samples x 5 assets)
        """
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = PortfolioOptimizer().to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}')
                
        return model 