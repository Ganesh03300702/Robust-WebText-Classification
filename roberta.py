from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "roberta_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

text = "Stock market falls due to inflation"

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

pred = torch.argmax(outputs.logits, dim=1)

print("Prediction:", pred.item())
