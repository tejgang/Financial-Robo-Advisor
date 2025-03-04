import torch
import os
import pickle
from chatbot.processing import FinancialTokenizer
from chatbot.model import FinancialIntentRecognizer, PortfolioRecommender
from chatbot.train import FinancialDataset, train_intent_model, train_recommender

# Sample training data
texts = [
    "I'm 30 with $75k income, $50k savings, moderate risk tolerance",
    "25 years old, $90k salary, $100k savings, high risk appetite",
    "Looking to invest $200k for retirement with low risk",
    "35 years old, $120k income, medium risk tolerance"
]

intents = [1, 1, 3, 1]  # 0-4 based on your intent classes
profiles = [
    [30, 75000, 50000, 3, 5, 0],
    [25, 90000, 100000, 4, 10, 15000],
    [45, 120000, 200000, 2, 20, 50000],
    [35, 120000, 150000, 3, 15, 25000]
]

allocations = [
    [0.6, 0.3, 0.1, 0.0, 0.0],
    [0.7, 0.2, 0.1, 0.0, 0.0],
    [0.4, 0.5, 0.0, 0.1, 0.0],
    [0.5, 0.3, 0.1, 0.1, 0.0]
]

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = FinancialTokenizer(vocab_size=1000)
    tokenizer.fit(texts)
    
    # Save tokenizer
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Create dataset
    dataset = FinancialDataset(texts, intents, profiles, allocations)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train intent model
    vocab_size = len(tokenizer.vocab) + 1
    intent_model = FinancialIntentRecognizer(vocab_size).to(device)
    train_intent_model(intent_model, dataset, tokenizer, device, epochs=50)
    
    # Save intent model
    torch.save({
        'state_dict': intent_model.state_dict(),
        'vocab_size': vocab_size
    }, 'models/intent_model.pth')
    
    # Train recommender
    recommender = PortfolioRecommender().to(device)
    train_recommender(recommender, dataset, tokenizer, device, epochs=100)
    
    # Save recommender
    torch.save(recommender.state_dict(), 'models/recommender.pth')
    
    print("Models trained and saved successfully!") 