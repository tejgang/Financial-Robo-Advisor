from collections import defaultdict
import re

class FinancialTokenizer:
    def __init__(self, vocab_size=10000):
        self.word_counts = defaultdict(int)
        self.vocab = {}
        self.vocab_size = vocab_size
        
    def fit(self, texts):
        for text in texts:
            for word in self._tokenize(text):
                self.word_counts[word] += 1
                
        sorted_words = sorted(self.word_counts.items(), key=lambda x: -x[1])
        self.vocab = {word: i+1 for i, (word, _) in enumerate(sorted_words[:self.vocab_size])}
        
    def _tokenize(self, text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def __call__(self, text):
        return [self.vocab.get(word, 0) for word in self._tokenize(text)] 