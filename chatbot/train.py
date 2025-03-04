import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class FinancialDataset(Dataset):
    def __init__(self, texts, intents, profiles, allocations):
        self.texts = texts
        self.intents = intents
        self.profiles = profiles
        self.allocations = allocations
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return (
            self.texts[idx],
            self.intents[idx],
            self.profiles[idx],
            self.allocations[idx]
        )

def collate_batch(batch, tokenizer):
    texts, intents, profiles, allocations = zip(*batch)
    
    # Tokenize texts
    tokenized = [torch.tensor(tokenizer(text)) for text in texts]
    lengths = torch.tensor([len(tokens) for tokens in tokenized])
    padded_texts = pad_sequence(tokenized, batch_first=True)
    
    return (
        padded_texts,
        lengths,
        torch.tensor(intents),
        torch.tensor(profiles, dtype=torch.float32),
        torch.tensor(allocations, dtype=torch.float32)
    )

def train_intent_model(model, dataset, tokenizer, device, epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=lambda b: collate_batch(b, tokenizer),
        shuffle=True
    )
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for texts, lengths, intents, _, _ in train_loader:
            texts, lengths, intents = texts.to(device), lengths.to(device), intents.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts, lengths)
            loss = criterion(outputs, intents)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (torch.argmax(outputs, 1) == intents).sum().item()
        
        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Acc: {acc:.2%}")

def train_recommender(model, dataset, tokenizer, device, epochs=20):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=lambda b: collate_batch(b, tokenizer),
        shuffle=True
    )
    
    for epoch in range(epochs):
        total_loss = 0
        for _, _, _, profiles, allocations in train_loader:
            profiles, allocations = profiles.to(device), allocations.to(device)
            
            optimizer.zero_grad()
            outputs = model(profiles)
            loss = criterion(outputs, allocations)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}") 