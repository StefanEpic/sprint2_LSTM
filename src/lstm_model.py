import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.init_weights()

    def forward(self, x):
        emb = self.embedding(x)  
        lstm_out, hidden = self.lstm(emb, hidden)  
        output = self.fc(lstm_out)
        return output, hidden

    def init_weights(self):
        # Инициализация embedding
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Инициализация LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Forget gate bias = 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        # Инициализация Linear
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)