import train
import predict_project as predict
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments


# trained_model, tokenizer = train.train()
tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
#自作のtrained_model

#====================================================================
# trained_model = RobertaForSequenceClassification.from_pretrained("trained/trained_model-Argo-Hive-re")
#====================================================================
trained_model = RobertaForSequenceClassification.from_pretrained("trainedNew/trained_model-Merge--9")

# コメントを用意します

# ファイルからデータを読み込む
with open('dataset/data--Ant-Hivernate_reshaped.txt', 'r') as f:
    comments = [line.strip() for line in f.readlines()]  # 各行を読み込み、改行文字を取り除く

with open('dataset/label--Ant-Hivernate_reshaped.txt', 'r') as f:
    # "positive" を 1 に、"false" を 0 にマッピング
    labels = [1 if line.strip() == 'positive' else 0 for line in f.readlines()]  

data = [{'comment': c, 'label': l} for c, l in zip(comments, labels)]
for i in data:
    print(i)

# SATDかどうか判定します
predict.predict_satd(data, trained_model, tokenizer)

