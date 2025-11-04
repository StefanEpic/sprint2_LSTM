import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, num_layers=2, dropout=0.5):
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

    def forward(self, x, hidden=None):
        """Прямой проход."""
        emb = self.embedding(x)  
        lstm_out, hidden = self.lstm(emb, hidden)  
        output = self.fc(lstm_out)
        return output, hidden

    def init_weights(self):
        """Инициализация embedding."""
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
    
    def generate(self, input_ids, max_length=64, temperature=1.0, eos_token_id=None):
        """Генерация продолжения текста."""
        self.eval()
        
        # input_ids должен иметь размерность [batch_size, seq_len]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Добавляем batch dimension если нужно
        
        batch_size = input_ids.size(0)
        with torch.no_grad():
                # Получаем hidden state из входной последовательности
                emb = self.embedding(input_ids)
                _, hidden = self.lstm(emb)
                
                last_token = input_ids[:, -1:]
                generated = input_ids.clone()
                
                for _ in range(max_length):
                    emb = self.embedding(last_token)
                    lstm_out, hidden = self.lstm(emb, hidden)
                    output = self.fc(lstm_out.squeeze(1))
                    
                    # Применяем температуру и сэмплируем
                    output = output / temperature
                    probabilities = torch.softmax(output, dim=-1)
                    next_token = torch.multinomial(probabilities, 1)
                    
                    generated = torch.cat([generated, next_token], dim=1)
                    last_token = next_token
                    
                    # Более гибкая проверка конца последовательности
                    if eos_token_id is not None and (next_token == eos_token_id).all():
                        break
                        
        return generated.squeeze(0) if batch_size == 1 else generated

            