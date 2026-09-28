import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix
from collections import Counter

print("ERROR ANALYSIS STARTED")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------
# CHANGE THIS to your best model folder
# ---------------------------------------
MODEL_PATH = "./bert-agnews"  

class_names = ["World", "Sports", "Business", "Sci/Tech"]

# ---------------------------------------
# Load Test Dataset
# ---------------------------------------
print("Loading test dataset...")
dataset = load_dataset("ag_news", split="test[:1000]")
texts = dataset["text"]
true_labels = dataset["label"]

# ---------------------------------------
# Load Model
# ---------------------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# ---------------------------------------
# Run Predictions
# ---------------------------------------
print("Running predictions...")
predicted_labels = []

for text in texts:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits

    prediction = torch.argmax(logits, dim=1).item()
    predicted_labels.append(prediction)

# ---------------------------------------
# Confusion Matrix
# ---------------------------------------
print("Generating Confusion Matrix...")
cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - AG News")
plt.tight_layout()
plt.show()

# ---------------------------------------
# Top 5 Most Confused Class Pairs
# ---------------------------------------
print("\nTop 5 Most Confused Class Pairs:")

confusions = []
for true, pred in zip(true_labels, predicted_labels):
    if true != pred:
        confusions.append((true, pred))

counter = Counter(confusions)

for (true, pred), count in counter.most_common(5):
    print(f"{class_names[true]} → {class_names[pred]} : {count} times")

print("\nDAY 16 COMPLETED")