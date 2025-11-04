import torch
from torch import nn
from tqdm import tqdm

from src.eval_lstm import calculate_rouge


def train_model(model, train_loader, val_loader, tokenizer, num_epochs=10, learning_rate=0.001):
    """Тренировка модели c вычислением метрик."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    train_losses = []
    val_rouge_scores = []
    
    print(f"Starting training on {device}...")
    print(f"Vocabulary size: {model.embedding.num_embeddings}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Тренировка
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Прямой проход
            outputs, _ = model(input_ids)
            
            # Вычисление потерь (игнорируем паддинг)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
            
            # Обратный проход
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/batch_count:.4f}'
            })
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Валидация с ROUGE
        print("Calculating validation metrics...")
        rouge_scores = calculate_rouge(model, val_loader, tokenizer)
        val_rouge_scores.append(rouge_scores)
        
        scheduler.step(avg_train_loss)
        
        # Вывод метрик эпохи
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"ROUGE-1:  {rouge_scores['rouge1']:.4f}")
        print(f"ROUGE-2:  {rouge_scores['rouge2']:.4f}")
        print("-" * 50)
        
        # Показываем примеры генерации каждые 2 эпохи
        if (epoch + 1) % 2 == 0:
            show_generation_examples(model, val_loader, tokenizer, device, num_examples=3)
    
    return {
        'train_losses': train_losses,
        'val_rouge_scores': val_rouge_scores,
        'model': model
    }


def show_generation_examples(model, dataloader, tokenizer, device, num_examples=3):
    """Показывает примеры генерации модели."""
    model.eval()
    examples_shown = 0
    
    print("\nGeneration Examples:")
    print("=" * 80)
    
    with torch.no_grad():
        for batch in dataloader:
            if examples_shown >= num_examples:
                break
                
            input_ids = batch['input_ids'].to(device)
            
            # Берем первый пример из батча
            input_seq = input_ids[0:1]
            
            # Определяем длину для генерации - берем только начало
            seq_len = (input_seq != tokenizer.pad_token_id).sum().item()
            input_length = max(1, int(seq_len * 0.75))  # Берем только 75% для входа
            print(input_length)
            
            # Вход для генерации (только начало)
            input_for_gen = input_seq[:, :input_length]
            
            print(input_length)
            # Генерируем продолжение
            generated = model.generate(
                input_for_gen,
                max_length=64,  
                temperature=0.5
            )
            
            # Декодируем отдельно
            original_input = tokenizer.decode(input_for_gen[0], skip_special_tokens=True)
            full_generated = tokenizer.decode(generated[0], skip_special_tokens=True)
            
            # Полный целевой текст для сравнения
            full_target = tokenizer.decode(input_seq[0][:seq_len], skip_special_tokens=True)
            
            print(f"Input:      {original_input}")
            print(f"Generated:  {full_generated}")
            print(f"Full target:{full_target}")
            print(f"→ Generated NEW text: '{full_generated[len(original_input):].strip()}'")
            print("-" * 80)
            
            examples_shown += 1