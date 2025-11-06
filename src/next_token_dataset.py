import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class PostsDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=64):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = self.data['data'].tolist()
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # токенизируем
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).squeeze(0)
        
        # X: все токены кроме последнего
        # Y: все токены кроме первого (сдвиг на 1 вправо)
        input_ids = tokens[:-1]  # ["я", "собираюсь", "купить"]
        target_ids = tokens[1:]  # ["собираюсь", "купить", "продукты"]
        
        # Маска внимания (игнорируем padding токены)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_ids
        }

# Инициализация токенизатора
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Добавляем pad token если его нет
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Создание датасетов
train_dataset = PostsDataset('data/train.csv', tokenizer)
val_dataset = PostsDataset('data/val.csv', tokenizer)
test_dataset = PostsDataset('data/test.csv', tokenizer)

# Создание DataLoader
batch_size = 128

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)
