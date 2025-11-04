from src.lstm_train import train_model
from src.lstm_model import LSTMModel
from src.next_token_dataset import tokenizer, train_loader, val_loader


vocab_size = len(tokenizer)
model = LSTMModel(vocab_size=vocab_size, hidden_dim=64, num_layers=3)
model.tokenizer = tokenizer


train_model(model, train_loader, val_loader, tokenizer, num_epochs=10, learning_rate=0.001)
