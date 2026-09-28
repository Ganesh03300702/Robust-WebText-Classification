import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

print("ENSEMBLE VOTING STARTED")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "BERT": "./bert-agnews",
    "RoBERTa": "./roberta_model",
    "DistilBERT": "./distillbert_model",
    "DeBERTa": "./deberta_model",
}

# Load small test split
dataset = load_dataset("ag_news", split="test[:200]")
texts = dataset["text"]
labels = np.array(dataset["label"])

# ----------------------------
# Step 1: Extract logits
# ----------------------------
def extract_logits(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()

    all_logits = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits

        all_logits.append(logits.cpu().numpy())

    return np.vstack(all_logits)


print("Extracting logits from all models...")
model_logits = []

for name, path in MODELS.items():
    print(f"Processing {name}")
    logits = extract_logits(path)
    model_logits.append(logits)

model_logits = np.array(model_logits)  # Shape: (4, samples, classes)

# ----------------------------
# Soft Voting (Average Logits)
# ----------------------------
def soft_voting(logits_array):
    avg_logits = np.mean(logits_array, axis=0)
    return np.argmax(avg_logits, axis=1)

# ----------------------------
# Hard Voting (Majority Vote)
# ----------------------------
def hard_voting(logits_array):
    predictions = np.argmax(logits_array, axis=2)  # shape (4, samples)
    final_preds = []

    for i in range(predictions.shape[1]):
        votes = predictions[:, i]
        majority = Counter(votes).most_common(1)[0][0]
        final_preds.append(majority)

    return np.array(final_preds)

# ----------------------------
# Evaluate
# ----------------------------
soft_preds = soft_voting(model_logits)
hard_preds = hard_voting(model_logits)

soft_acc = accuracy_score(labels, soft_preds)
soft_f1 = f1_score(labels, soft_preds, average="macro")

hard_acc = accuracy_score(labels, hard_preds)
hard_f1 = f1_score(labels, hard_preds, average="macro")

print("\nENSEMBLE RESULTS")
print("----------------------------")
print(f"Soft Voting  - Accuracy: {soft_acc:.4f}, F1: {soft_f1:.4f}")
print(f"Hard Voting  - Accuracy: {hard_acc:.4f}, F1: {hard_f1:.4f}")

if soft_f1 > hard_f1:
    print("\nSoft Voting performs better.")
elif hard_f1 > soft_f1:
    print("\nHard Voting performs better.")
else:
    print("\nBoth perform equally.")

# Save results
df = pd.DataFrame({
    "Method": ["Soft Voting", "Hard Voting"],
    "Accuracy": [soft_acc, hard_acc],
    "F1": [soft_f1, hard_f1]
})

df.to_csv("ensemble_results.csv", index=False)
print("\nResults saved to ensemble_results.csv")