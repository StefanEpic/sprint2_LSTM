from src.eval_transformer_pipeline import run_transformer_tests
from src.lstm_train import train_model, save_training_model, save_training_curves
from src.lstm_model import LSTMModel
from src.next_token_dataset import tokenizer, train_loader, val_loader

# Создаем модель
vocab_size = len(tokenizer)
model = LSTMModel(vocab_size=vocab_size, hidden_dim=128, num_layers=3, dropout=0.5)
model.tokenizer = tokenizer

# Запускаем процесс обучения
train_results = train_model(model, train_loader, val_loader, tokenizer, num_epochs=10, learning_rate=0.001)

# Сохраняем итоговые результаты
save_dir = save_training_model(train_results['model'])
save_training_curves(train_results['train_losses'], train_results['val_rouge_scores'])

# # Тестирование готовой модели distilgpt2
# run_transformer_tests()