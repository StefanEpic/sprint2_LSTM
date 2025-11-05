import json
import os

import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from src.eval_lstm import calculate_rouge_and_test_results


def train_model(model, train_loader, val_loader, tokenizer, num_epochs=10, learning_rate=0.001):
    """Тренировка модели с вычислением метрик."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Работаем на {device}...")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    train_losses = []
    val_rouge_scores = []

    for epoch in range(num_epochs):
        # Тренировка
        model.train()
        epoch_loss = 0
        batch_count = 0

        train_pbar = tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{num_epochs}")
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
                'avg_loss': f'{epoch_loss / batch_count:.4f}'
            })

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Валидация с ROUGE, получение последних генераций для вывода
        rouge_scores = calculate_rouge_and_test_results(model, val_loader, tokenizer)
        val_rouge_scores.append(rouge_scores)

        scheduler.step(avg_train_loss)

        # Вывод метрик эпохи
        print(f"\nЭпоха {epoch + 1}/{num_epochs} Итог:")
        print(f"Train Loss:   {avg_train_loss:.4f}")
        print(f"ROUGE-1:      {rouge_scores['rouge1']:.4f}")
        print(f"ROUGE-2:      {rouge_scores['rouge2']:.4f}")
        print(f"Исходник:                     {rouge_scores['etalon_text']}")
        print(f"Введенный текст:              {rouge_scores['input_text']}")
        print(f"Полный ответ нейросети:       {rouge_scores['full_generated_text']}")
        print(f"Ожидаемая дополненная часть:  {rouge_scores['expected_text']}")
        print(f"Полученная дополненная часть: {rouge_scores['expected_generated_text']}")
        print("-" * 50)

        # Сохраняем этапы и модель для бэкапов/откатов
        dir_for_epoch_res_save = f"./train/epoch_{epoch + 1}"
        save_training_model(model, dir_for_epoch_res_save)
        save_training_curves(train_losses, val_rouge_scores, dir_for_epoch_res_save)
        save_json({
            "rouge1": round(rouge_scores['rouge1'], 2),
            "rouge2": round(rouge_scores['rouge2'], 2),
            "epoch_loss": epoch_loss,
            "avg_train_loss": avg_train_loss,
            "source": rouge_scores['etalon_text'],
            "input_text": rouge_scores['input_text'],
            "full_generated_text": rouge_scores['full_generated_text'],
            "expected_text": rouge_scores['expected_text'],
            "expected_generated_text": rouge_scores['expected_generated_text'],
        }, dir_for_epoch_res_save)
        print(f"Промежуточные результаты сохранены в {dir_for_epoch_res_save}")

    return {
        'train_losses': train_losses,
        'val_rouge_scores': val_rouge_scores,
        'model': model
    }


def save_training_model(model, save_dir="./model"):
    """Сохраняет модель и графики обучения."""
    # Создаем директорию если не существует
    os.makedirs(save_dir, exist_ok=True)

    # Сохранение всей модели
    full_model_path = os.path.join(save_dir, "full_model.pth")
    torch.save(model, full_model_path)
    return save_dir


def save_training_curves(train_losses, val_rouge_scores, save_dir="./graphics"):
    """Создает и сохраняет графики обучения."""
    # Создаем директорию если не существует
    os.makedirs(save_dir, exist_ok=True)

    # График потерь
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # График ROUGE scores
    plt.subplot(1, 2, 2)
    rouge1_scores = [score['rouge1'] for score in val_rouge_scores]
    rouge2_scores = [score['rouge2'] for score in val_rouge_scores]

    plt.plot(rouge1_scores, 'r-', label='ROUGE-1')
    plt.plot(rouge2_scores, 'g-', label='ROUGE-2')
    plt.title('Validation ROUGE Scores')
    plt.xlabel('Epoch')
    plt.ylabel('ROUGE Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Сохраняем графики
    plot_path = os.path.join(save_dir, "training_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_json(data, save_dir="./model"):
    """Создает и сохраняет json."""
    # Создаем директорию если не существует
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'metrics.json'), 'w') as file:
        json.dump(data, ensure_ascii=False, indent=2, fp=file)
