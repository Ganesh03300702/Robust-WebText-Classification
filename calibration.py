import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
from torch.nn.functional import cross_entropy

print("TEMPERATURE SCALING STARTED")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./bert-agnews"

# -----------------------------
# Load Data
# -----------------------------
val_dataset = load_dataset("ag_news", split="train[1000:1500]")
test_dataset = load_dataset("ag_news", split="test[:500]")

val_texts = val_dataset["text"]
val_labels = torch.tensor(val_dataset["label"]).to(DEVICE)

test_texts = test_dataset["text"]
test_labels = np.array(test_dataset["label"])
test_labels_tensor = torch.tensor(test_labels).to(DEVICE)

# -----------------------------
# Load Model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# -----------------------------
# Extract Logits
# -----------------------------
def extract_logits(texts):
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

        all_logits.append(logits)

    return torch.cat(all_logits, dim=0)


print("Extracting validation logits...")
val_logits = extract_logits(val_texts)

print("Extracting test logits...")
test_logits = extract_logits(test_texts)

# -----------------------------
# Temperature Scaling Module
# -----------------------------
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        return logits / self.temperature


# -----------------------------
# Optimize Temperature
# -----------------------------
print("Optimizing temperature...")

scaler = TemperatureScaler().to(DEVICE)
optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=100)

def closure():
    optimizer.zero_grad()
    scaled_logits = scaler(val_logits)
    loss = cross_entropy(scaled_logits, val_labels)
    loss.backward()
    return loss

optimizer.step(closure)

optimal_T = scaler.temperature.item()
print(f"Optimal Temperature T = {optimal_T:.4f}")

# -----------------------------
# Accuracy Check
# -----------------------------
original_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
original_acc = accuracy_score(test_labels, original_preds)

scaled_test_logits = scaler(test_logits)
scaled_preds = torch.argmax(scaled_test_logits, dim=1).cpu().numpy()
scaled_acc = accuracy_score(test_labels, scaled_preds)

# -----------------------------
# NLL Comparison
# -----------------------------
nll_before = cross_entropy(test_logits, test_labels_tensor).item()
nll_after = cross_entropy(scaled_test_logits, test_labels_tensor).item()

# -----------------------------
# ECE Calculation
# -----------------------------
def compute_ece(logits, labels, n_bins=15):
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1).to(DEVICE)
    ece = torch.zeros(1, device=DEVICE)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

ece_before = compute_ece(test_logits, test_labels_tensor)
ece_after = compute_ece(scaled_test_logits, test_labels_tensor)

# -----------------------------
# Results
# -----------------------------
print("\nRESULTS")
print("----------------------------")
print(f"Accuracy BEFORE calibration: {original_acc:.4f}")
print(f"Accuracy AFTER calibration:  {scaled_acc:.4f}")
print(f"NLL BEFORE: {nll_before:.4f}")
print(f"NLL AFTER:  {nll_after:.4f}")
print(f"ECE BEFORE: {ece_before:.4f}")
print(f"ECE AFTER:  {ece_after:.4f}")

print("\nNote: Accuracy should remain same.")
print("If NLL and ECE decrease → Calibration improved.")