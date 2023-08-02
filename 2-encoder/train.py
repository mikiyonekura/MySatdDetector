from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch


# モデルとトークナイザーの初期化
model_name = 'microsoft/codebert-base'
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# 学習データの準備
def prepare_data(data):
    # ここでは、コメントとコードを連結して1つのテキストとして扱います
    texts = [f"{x['comment']} {x['code']}" for x in data]
    labels = [x['label'] for x in data]  # ラベルは0（非SATD）と1（SATD）の二値
    encodings = tokenizer(texts, truncation=True, padding=True)
    return encodings, labels

# データセットの作成
class SATDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 学習と評価データの準備
# ダミーデータの作成
train_data = [
    {'comment': 'This is a hack, need to fix in future', 'code': 'int x = y / 0;', 'label': 1},
    {'comment': 'Calculate the area', 'code': 'int area = width * height;', 'label': 0},
    # 他のデータも同様に
]

valid_data = [
    {'comment': 'Temporary solution', 'code': 'int x = y / 0;', 'label': 1},
    {'comment': 'Calculate the volume', 'code': 'int volume = width * height * depth;', 'label': 0},
    # 他のデータも同様に
]

# train_data = ...  # ここに実際の学習データを指定します
# valid_data = ...  # ここに実際の評価データを指定します
train_encodings, train_labels = prepare_data(train_data)
valid_encodings, valid_labels = prepare_data(valid_data)
train_dataset = SATDDataset(train_encodings, train_labels)
valid_dataset = SATDDataset(valid_encodings, valid_labels)

# 学習設定
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,

    # per_device_train_batch_size=16,
    # per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# トレーナーの初期化と学習の開始
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)



if __name__ == '__main__':
    print("import train")