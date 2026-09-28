from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print("Starting script...")

MODEL_PATH = "deberta_model"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("Model loaded ✅")

labels = ["World", "Sports", "Business", "Sci/Tech"]

def predict(text):
    print("Running prediction...")
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits).item()
    print("Prediction:", labels[pred])


predict("Apple releases new AI chip")
