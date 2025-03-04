import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class FinancialIntentRecognizer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # 5 financial intents
        )
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(hidden)

class PortfolioRecommender(nn.Module):
    def __init__(self, input_size=6, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 5),  # 5 asset classes
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.net(x)

class ChatBot:
    def __init__(self, intent_model, recommender, tokenizer):
        self.intent_model = intent_model
        self.recommender = recommender
        self.tokenizer = tokenizer
        self.user_profile = {}
        
    def process_message(self, text):
        # Tokenize and get intent
        tokens = self.tokenizer(text)
        lengths = torch.tensor([len(tokens)])
        padded = pad_sequence([torch.tensor(tokens)], batch_first=True)
        
        with torch.no_grad():
            intent_logits = self.intent_model(padded, lengths)
            intent = torch.argmax(intent_logits, dim=1).item()
            
        return self._handle_intent(intent, text)
    
    def _handle_intent(self, intent, text):
        intents = [
            'risk_assessment', 'portfolio_recommendation',
            'debt_management', 'retirement_planning', 'generic_advice'
        ]
        return getattr(self, f"_handle_{intents[intent]}")(text)
    
    def _handle_portfolio_recommendation(self, text):
        if not self._validate_profile():
            return "Please provide your age, income, risk tolerance (1-5), investment horizon (in years), and debt"
            
        features = torch.tensor([
            self.user_profile['age'],
            self.user_profile['income'],
            self.user_profile['savings'],
            self.user_profile['risk_tolerance'],
            self.user_profile['investment_horizon'],
            self.user_profile['debt']
        ]).float().unsqueeze(0)
        
        with torch.no_grad():
            allocation = self.recommender(features).squeeze()
            
        return self._format_allocation(allocation) 