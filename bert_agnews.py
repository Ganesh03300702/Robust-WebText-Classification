"""
Day 4: Fine-tune BERT on AG-News (4-class classification)
Used for training in Google Colab (GPU recommended)
"""

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
)

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "bert-agnews"
NUM_LABELS = 4
MAX_LENGTH = 128

# Label mapping
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
label2id = {v: k for k, v in id2label.items()}


# ---------- Metrics ----------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


# ---------- Training ----------
def main():

    print("Loading dataset...")
    dataset = load_dataset("ag_news")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DefaultDataCollator()

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(tokenize, batched=True)

    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_ds = dataset["train"].select(range(20000))   # smaller subset (fast + good accuracy)
    eval_ds = train_ds.select(range(2000))
    train_ds = train_ds.select(range(2000, len(train_ds)))

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training started...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete ✔")


if __name__ == "__main__":
    main()
