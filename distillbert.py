from transformers import DistilBertTokenizerFast
from transformers import AutoModelForSequenceClassification
import torch

print("Loading tokenizer from HF (compatible)...")

# Load original tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

print("Loading YOUR trained model locally...")

MODEL_PATH = r"C:\Users\GANESH\web-text-classification-transformers\distillbert_model"

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.eval()

labels = ["World", "Sports", "Business", "Sci/Tech"]

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()

    print("\nText:", text)
    print("Prediction:", labels[pred])
    print("Confidence:", probs[0][pred].item())


predict("Apple launches new AI chip")
predict("India won the cricket match")
