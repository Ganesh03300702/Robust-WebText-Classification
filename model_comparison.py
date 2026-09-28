import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

print("SCRIPT STARTED")

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Local model paths
MODELS = {
    "BERT": "./bert-agnews",
    "RoBERTa": "./roberta_model",
    "DistilBERT": "./distillbert_model",
    "DeBERTa": "./deberta_model",
}

# Load small test split
dataset = load_dataset("ag_news", split="test[:200]")
texts = dataset["text"]
labels = dataset["label"]


def evaluate(name, path):
    print(f"\nLoading {name}")

    # Load from local folder
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
    model.eval()

    preds = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits

        preds.append(torch.argmax(logits, dim=1).item())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    print(f"{name} DONE")
    return acc, f1


results = []

for name, path in MODELS.items():
    acc, f1 = evaluate(name, path)
    results.append([name, acc, f1])

df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1"])

print("\nRESULTS:")
print(df)

df.to_csv("model_comparison.csv", index=False)
print("\nCSV SAVED")