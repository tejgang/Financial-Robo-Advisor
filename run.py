from data.preprocessor import FinancialDataPreprocessor
from models.train import ModelTrainer
from robo_advisor.core import RoboAdvisor
import torch

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate and preprocess data
    preprocessor = FinancialDataPreprocessor()
    df = preprocessor.generate_synthetic_data()
    X, y = preprocessor.preprocess(df)
    
    # Train models
    trainer = ModelTrainer(device=device)
    
    print("\nTraining risk profile classifier...")
    classifier = trainer.train_classifier(X, df['risk_tolerance'].values - 1)  # Adjust to 0-based indexing
    
    print("\nTraining portfolio optimizer...")
    optimizer = trainer.train_optimizer(X, y)
    
    # Save models
    torch.jit.save(torch.jit.script(classifier), 'models/classifier.pth')
    torch.jit.save(torch.jit.script(optimizer), 'models/optimizer.pth')
    
    print("\nModels trained and saved successfully!")
    
    # Test the advisor
    print("\nTesting the advisor with sample data:")
    advisor = RoboAdvisor(device=device)
    user_data = {
        'age': 23,
        'income': 100000,
        'savings': 35000,
        'risk_tolerance': 3,
        'investment_goal': 3
    }
    
    # Get market data
    market_data = preprocessor.get_market_data()
    
    # Test the advisor with market data
    portfolio = advisor.generate_portfolio(user_data, market_data)
    
    print("\nRecommended Portfolio:")
    print(f"Risk Profile: {portfolio['risk_profile']}")
    print("\nAsset Allocation:")
    for asset, allocation in portfolio['allocation'].items():
        print(f"{asset}: {allocation:.2%}")

if __name__ == "__main__":
    main()