import requests
from io import StringIO
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from next_token_dataset import tokenizer


# функция для "чистки" текстов
def clean_string(text):
    text = str(text)
    # приведение к нижнему регистру
    text = text.lower()
    # удаление тегов и URL
    text = re.sub(r'@\w+|https?://\S+|http?://\S+', '', text)
    # удаление всего, кроме латинских букв, цифр и пробелов
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # удаление дублирующихся пробелов, удаление пробелов по краям
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Скачивание
url = 'https://code.s3.yandex.net/deep-learning/tweets.txt'
r = requests.get(url)

# Сохранение
lines = r.text.splitlines()
df = pd.DataFrame(lines, columns=['data'])
df[['data']].to_csv('data/raw_dataset.csv', index=False)

# Очистка текста
df['data'] = df['data'].apply(clean_string)

# Сохранение обработанного датасета
df[['data']].to_csv('data/dataset_processed.csv', index=False)

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=True))

token_lengths = df['data'].apply(count_tokens)
print(f"Минимальная длина в токенах: {token_lengths.min()}")
print(f"Средняя длина в токенах: {token_lengths.mean():.2f}")
print(f"Максимальная длина в токенах: {token_lengths.max()}")

# Разбиваем датасет на обучающую, валидационную и тестовую выборки:
# трейн: 80%,
# валидация: 10%,
# тест: 10%.
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Сохранение разделенных датасетов
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
