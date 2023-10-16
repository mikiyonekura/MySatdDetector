import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

def predict_satd(data, model, tokenizer):
    true_labels = []
    predicted_labels = []

    for d in data:
        # 単に文字列で連結した場合
        text = d["comment"]+d["code"]
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")

        # SEPを間に組み込みたい場合
        # inputs = tokenizer(d["comment"], d["code"], truncation=True, padding=True, return_tensors="pt")

        
        # If available use GPU
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            model.to('cuda')
        
        # Set model in evaluation mode and get output
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities from output
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get probability for non-SATD and SATD
        non_satd_prob = probabilities[0][0].item()
        satd_prob = probabilities[0][1].item()

        # Predicted label is the one with the highest probability
        predicted_label = 0 if non_satd_prob > satd_prob else 1

        # Save true and predicted labels
        true_labels.append(d["label"])
        predicted_labels.append(predicted_label)

        print("---------------------------------")
        print(f"comment&code: {d['comment']}{d['code']}")
        print(f"true label: {d['label']}")
        print(f"predicted label: {predicted_label}")
        print(f"non_satd_prob: {non_satd_prob}",f"satd_prob: {satd_prob}")

    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks([0.5,1.5], ['non-SATD', 'SATD'], rotation=0, va='center')
    plt.yticks([0.5,1.5], ['non-SATD', 'SATD'], rotation=90, va='center')
    plt.show()

if __name__ == "__main__":
    predict_satd(data, model, tokenizer)
