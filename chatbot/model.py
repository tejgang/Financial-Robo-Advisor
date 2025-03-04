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
        self.device = next(intent_model.parameters()).device  # Get the device from the model
        
    def process_message(self, text):
        # Tokenize and get intent
        tokens = self.tokenizer(text)
        lengths = torch.tensor([len(tokens)])
        padded = pad_sequence([torch.tensor(tokens)], batch_first=True).to(self.device)
        
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
            
        features = torch.tensor([[
            self.user_profile['age'],
            self.user_profile['income'],
            self.user_profile['savings'],
            self.user_profile['risk_tolerance'],
            self.user_profile['investment_horizon'],
            self.user_profile['debt']
        ]]).float().to(self.device)
        
        with torch.no_grad():
            allocation = self.recommender(features).squeeze().cpu()  # Move back to CPU for numpy conversion
            
        return self._format_allocation(allocation)
    
    def _handle_risk_assessment(self, text):
        if not self._validate_profile():
            return "I need your complete profile to assess risk. Please provide your details first."
        
        risk_score = self.user_profile['risk_tolerance']
        risk_levels = ['Very Conservative', 'Conservative', 'Moderate', 'Aggressive', 'Very Aggressive']
        risk_level = risk_levels[min(int(risk_score) - 1, 4)]
        
        return f"Based on your profile, your risk tolerance is {risk_level}. " \
               f"This is determined by your age ({self.user_profile['age']}), " \
               f"investment horizon ({self.user_profile['investment_horizon']} years), " \
               f"and stated risk preference ({risk_score}/5)."

    def _handle_retirement_planning(self, text):
        if not self._validate_profile():
            return "I need your complete profile for retirement planning. Please provide your details first."
        
        years_to_retire = max(65 - self.user_profile['age'], 0)
        monthly_savings = self.user_profile['income'] * 0.15  # Recommended 15% savings
        
        return f"Based on your age of {self.user_profile['age']}, you have {years_to_retire} years until typical retirement age. " \
               f"Consider saving ${monthly_savings:.2f} monthly (15% of income) for retirement. " \
               f"With your current savings of ${self.user_profile['savings']}, you're on track for retirement planning."

    def _handle_debt_management(self, text):
        if not self._validate_profile():
            return "I need your complete profile to provide debt management advice. Please provide your details first."
        
        debt = self.user_profile['debt']
        income = self.user_profile['income']
        debt_to_income = (debt / income) * 100 if income > 0 else 0
        
        if debt_to_income == 0:
            return "You have no debt! Keep maintaining good financial habits."
        elif debt_to_income < 30:
            return f"Your debt-to-income ratio is {debt_to_income:.1f}%. This is healthy! Consider maintaining current payments."
        else:
            return f"Your debt-to-income ratio is {debt_to_income:.1f}%. Consider debt consolidation or accelerated payments."

    def _handle_generic_advice(self, text):
        return "I can help you with portfolio recommendations, risk assessment, retirement planning, and debt management. " \
               "What would you like to know more about?"

    def _validate_profile(self):
        required_fields = ['age', 'income', 'savings', 'risk_tolerance', 'investment_horizon', 'debt']
        return all(field in self.user_profile for field in required_fields)
    
    def _format_allocation(self, allocation):
        assets = ['Stocks', 'Bonds', 'Crypto', 'Real Estate', 'Cash']
        formatted = [f"{assets[i]}: {allocation[i]*100:.1f}%" for i in range(len(assets))]
        return "Recommended portfolio allocation:\n" + "\n".join(formatted) 