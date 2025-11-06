import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from src.lstm_train import save_json, save_training_curves


class BaseTransformer:
    def __init__(self):
        self.device = None
        self.model = None
        self.tokenizer = None
        self.generator = None

    def init_model(self, device, model):
        """Инициализация модели."""
        if self.device is None:
            self.device = device
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(model)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.generator is None:
            self.generator = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )

    def run(self, text):
        """Генерация дополнительного текста."""
        # Токенизируем входной текст
        input_tokens = self.tokenizer.encode(text, return_tensors="pt")
        input_length = input_tokens.shape[1]

        # Рассчитываем целевую длину (если входные 75%, то общая длина = input_length / 0.75)
        target_length = int(input_length / 0.75)

        # Длина генерации = общая длина - входная длина
        max_new_tokens = target_length - input_length

        out = self.generator(
            text,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            temperature=0.5,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return out[0]["generated_text"]


def get_75_percent_words(text):
    """Получить 75% текста."""
    words = text.split()
    total_words = len(words)
    words_to_take = int(total_words * 0.75)
    text_75 = ' '.join(words[:words_to_take])
    text_25 = text.replace(text_75, "").strip()
    return text_75, text_25


def run_transformer_tests():
    transformer_model = BaseTransformer()
    transformer_model.init_model(0, "distilgpt2")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    res_dict_json = []
    val_rouge_scores = []

    texts = pd.read_csv("./data/test.csv")['data'].tolist()
    texts = [i for i in texts if isinstance(i, str)][:1000]
    texts = [i for i in texts if len(i.split()) >= 4]
    for text in texts:
        source, expected_text = get_75_percent_words(text)
        generated_text = transformer_model.run(source)
        expected_generated_text = generated_text.replace(source, "").strip()

        # Вычисляем ROUGE между сгенерированным и ожидаемым продолжением
        if generated_text:  # Проверяем, что есть что сравнивать
            scores = scorer.score(expected_text, expected_generated_text)
            rouge1_scores.append(round(scores['rouge1'].fmeasure, 2))
            rouge2_scores.append(round(scores['rouge2'].fmeasure, 2))
            
            res_dict_json.append({
                "rouge1": round(scores['rouge1'].fmeasure, 2),
                "rouge2": round(scores['rouge2'].fmeasure, 2),
                "source": text,
                "input_text": source,
                "full_generated_text": generated_text,
                "expected_text": expected_text,
                "expected_generated_text": expected_generated_text,
            })
            val_rouge_scores.append({
                "rouge1": round(scores['rouge1'].fmeasure, 2),
                "rouge2": round(scores['rouge2'].fmeasure, 2),
            })
    dir_for_res_save = "./transformer_tests_results" 
    save_training_curves(val_rouge_scores=val_rouge_scores, save_dir=dir_for_res_save)
    save_json(res_dict_json, dir_for_res_save)
    avg_rouge_1 = [i["rouge1"] for i in val_rouge_scores]
    avg_rouge_1 = round(sum(avg_rouge_1) / len(avg_rouge_1), 2)
    avg_rouge_2 = [i["rouge2"] for i in val_rouge_scores]
    avg_rouge_2 = round(sum(avg_rouge_2) / len(avg_rouge_2), 2)
    return {"avg_rouge_1": avg_rouge_1, "avg_rouge_2": avg_rouge_2} 
