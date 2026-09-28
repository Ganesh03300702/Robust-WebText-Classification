import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# IMPORTANT: your model is inside bert-agnews/bert-agnews
MODEL_PATH = "bert-agnews/bert-agnews"
MAX_LENGTH = 128
BATCH_SIZE = 16

print("Loading AG News dataset...")
dataset = load_dataset("ag_news")
test_ds = dataset["test"]

print("Loading trained BERT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

logits_list = []
labels_list = []

print("Extracting logits from test set...")

for i in tqdm(range(0, len(test_ds), BATCH_SIZE)):
    batch = test_ds[i:i+BATCH_SIZE]
    texts = batch["text"]
    labels = batch["label"]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()

    logits_list.append(logits)
    labels_list.append(np.array(labels))

logits_all = np.vstack(logits_list)
labels_all = np.concatenate(labels_list)

np.save("bert_logits.npy", logits_all)
np.save("bert_labels.npy", labels_all)

print("Logits saved to bert_logits.npy")
print("Labels saved to bert_labels.npy")
print("DAY 5 COMPLETED")
