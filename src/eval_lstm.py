import torch
from rouge_score import rouge_scorer
import numpy as np


def calculate_rouge_and_test_results(model, dataloader, tokenizer, num_examples=100):
    """Вычисление метрики ROUGE для модели с автодополнением."""
    device = next(model.parameters()).device
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i * dataloader.batch_size >= num_examples:
                break
                
            input_ids = batch['input_ids'].to(device)
            
            for j in range(input_ids.size(0)):
                # Берем только первые 3/4 входной последовательности для генерации
                seq_len = (input_ids[j] != tokenizer.pad_token_id).sum().item()
                split_idx = int(seq_len * 0.75)
                
                if split_idx == 0 and seq_len > 1:
                    split_idx = 1
                elif split_idx == seq_len and seq_len > 1:
                    split_idx = seq_len - 1
                
                input_part = input_ids[j][:split_idx]  # Первые 3/4 для входа
                expected_part = input_ids[j][split_idx:]  # Оставшиеся 1/4 для сравнения
                
                # Генерируем продолжение
                generated = model.generate(input_part.unsqueeze(0), max_length=64, temperature=0.5)

                # Декодируем тексты
                input_text = tokenizer.decode(input_part, skip_special_tokens=True)
                generated_text = tokenizer.decode(generated, skip_special_tokens=True)
                expected_text = tokenizer.decode(expected_part[expected_part != tokenizer.pad_token_id], 
                                               skip_special_tokens=True)
                expected_generated_text = generated_text.replace(input_text, "").strip()
                
                # Вычисляем ROUGE между сгенерированным и ожидаемым продолжением
                if expected_text.strip():  # Проверяем, что есть что сравнивать
                    scores = scorer.score(expected_text, expected_generated_text)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)

    return {
        'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0,
        'rouge2': np.mean(rouge2_scores) if rouge2_scores else 0,
        'etalon_text': f"{input_text} {expected_text}",
        'input_text': input_text,
        'full_generated_text': generated_text,
        'expected_text': expected_text,
        'expected_generated_text': expected_generated_text,
    }
