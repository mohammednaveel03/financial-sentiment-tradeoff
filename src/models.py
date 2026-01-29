import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, sentences, labels):
        self.enc = tokenizer(list(sentences), truncation=True, padding=False)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def measure_latency_ms(model, tokenizer, texts, device, batch_size=1, repeats=8):
    model.eval()
    model.to(device)

    batches = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        enc = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        batches.append(enc)

    # warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(**batches[0])

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            for b in batches:
                _ = model(**b)
            t1 = time.perf_counter()
            times.append(t1 - t0)

    avg = float(np.mean(times))
    return (avg / len(texts)) * 1000.0


def train_eval(model_name, train_df, val_df, test_df, epochs=2, batch_size=16, seed=42):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    train_ds = SimpleDataset(tokenizer, train_df["sentence"], train_df["label"])
    val_ds = SimpleDataset(tokenizer, val_df["sentence"], val_df["label"])
    test_ds = SimpleDataset(tokenizer, test_df["sentence"], test_df["label"])

    args = TrainingArguments(
        output_dir="tmp_out",
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=25,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    t0 = time.perf_counter()
    trainer.train()
    t1 = time.perf_counter()

    metrics = trainer.evaluate(test_ds)
    acc = float(metrics["eval_accuracy"])
    f1m = float(metrics["eval_f1_macro"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_texts = list(test_df["sentence"][:256])
    latency = measure_latency_ms(trainer.model, tokenizer, sample_texts, device=device, batch_size=1)

    return {
        "model": model_name,
        "accuracy": acc,
        "f1_macro": f1m,
        "latency_ms": latency,
        "train_seconds": float(t1 - t0),
        "device": device,
    }
