import train_v1
import predict

train_v1.trainer.train()

# コメントとコードを用意します
comment = "This is a hack, need to fix in future"

# SATDかどうか判定します
result = predict.predict_satd(comment)

print(result)
