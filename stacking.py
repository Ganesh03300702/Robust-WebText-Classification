import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

print("STACKING META-LEARNER STARTED")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "BERT": "./bert-agnews",
    "RoBERTa": "./roberta_model",
    "DistilBERT": "./distillbert_model",
    "DeBERTa": "./deberta_model",
}

# -----------------------------
# Load Train & Test Data
# -----------------------------
train_dataset = load_dataset("ag_news", split="train[:1000]")
test_dataset = load_dataset("ag_news", split="test[:500]")

X_train_texts = train_dataset["text"]
y_train = np.array(train_dataset["label"])

X_test_texts = test_dataset["text"]
y_test = np.array(test_dataset["label"])

# -----------------------------
# Extract logits function
# -----------------------------
def extract_logits(model_path, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()

    logits_list = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits

        logits_list.append(logits.cpu().numpy())

    return np.vstack(logits_list)

# -----------------------------
# Step 1: Get Logits for Train
# -----------------------------
print("Extracting TRAIN logits...")
train_logits_all = []

for name, path in MODELS.items():
    print(f"Processing {name}")
    logits = extract_logits(path, X_train_texts)
    train_logits_all.append(logits)

train_logits_all = np.hstack(train_logits_all)  # shape: (samples, 4*classes)

# -----------------------------
# Step 2: Train Meta-Learner
# -----------------------------
print("\nTraining Logistic Regression Meta-Learner...")
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(train_logits_all, y_train)

# -----------------------------
# Step 3: Get Test Logits
# -----------------------------
print("\nExtracting TEST logits...")
test_logits_all = []

for name, path in MODELS.items():
    print(f"Processing {name}")
    logits = extract_logits(path, X_test_texts)
    test_logits_all.append(logits)

test_logits_all = np.hstack(test_logits_all)

# -----------------------------
# Step 4: Predict & Evaluate
# -----------------------------
stack_preds = meta_model.predict(test_logits_all)

acc = accuracy_score(y_test, stack_preds)
f1 = f1_score(y_test, stack_preds, average="macro")

print("\nSTACKING RESULTS")
print("----------------------------")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save results
df = pd.DataFrame({
    "Method": ["Stacking (LogReg)"],
    "Accuracy": [acc],
    "F1": [f1]
})

df.to_csv("stacking_results.csv", index=False)
print("\nResults saved to stacking_results.csv")