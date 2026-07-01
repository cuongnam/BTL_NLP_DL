from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "preprocessed" / "event_type"
LABEL_MAP_PATH = ROOT_DIR / "data" / "preprocessed" / "label_maps.json"


class BKEEEventTypeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label2id, tokenizer=None, tokenizer_name="vinai/phobert-base", max_len=256):
        with open(data_path, "r", encoding="utf8") as f:
            self.data = json.load(f)

        self.label2id = label2id.copy()
        if "O" not in self.label2id:
            self.label2id["O"] = 33

        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            add_prefix_space=True,
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item["tokens"]
        labels = ["O"] * len(words)

        if "trigger" in item and "event_type" in item:
            start_idx = item["trigger"].get("start", 0)
            end_idx = item["trigger"].get("end", start_idx + 1)
            event_str = item["event_type"]
            for i in range(start_idx, min(end_idx, len(labels))):
                labels[i] = event_str

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        word_ids = encoding.word_ids()
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_str = labels[word_idx]
                label_ids.append(self.label2id.get(label_str, self.label2id["O"]))

        encoding["labels"] = label_ids
        return {k: torch.tensor(v, dtype=torch.long) for k, v in encoding.items()}


def compute_metrics(p, id2label):
    predictions, labels = p
    prediction_ids = np.argmax(predictions, axis=2)

    flat_predictions = []
    flat_labels = []

    for prediction, label in zip(prediction_ids, labels):
        for p_id, l_id in zip(prediction, label):
            if l_id != -100:
                flat_predictions.append(int(p_id))
                flat_labels.append(int(l_id))

    true_preds_str = [id2label.get(str(p_id), "O") for p_id in flat_predictions]
    true_labels_str = [id2label.get(str(l_id), "O") for l_id in flat_labels]

    labels_to_eval = sorted(set(true_labels_str) - {"O"})
    if not labels_to_eval:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels_str,
        true_preds_str,
        average="macro",
        labels=labels_to_eval,
        zero_division=0,
    )

    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    if wandb is not None:
        wandb.init(project="bkee-event-extraction", name="phobert_event_type_baseline")

    with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
        maps = json.load(f)

    event_maps = maps["event_type"]
    label2id = event_maps["label2id"].copy()
    if "O" not in label2id:
        label2id["O"] = 33

    id2label = {str(v): k for k, v in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=True, add_prefix_space=True)
    train_dataset = BKEEEventTypeDataset(DATA_DIR / "train.json", label2id, tokenizer=tokenizer)
    dev_dataset = BKEEEventTypeDataset(DATA_DIR / "dev.json", label2id, tokenizer=tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        "vinai/phobert-base",
        num_labels=len(label2id),
    )
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=(ROOT_DIR / "models" / "best_phobert_event_type").as_posix(),
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs_event_type",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb" if wandb is not None else "none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )

    trainer.train()
    trainer.save_model(ROOT_DIR / "models" / "best_phobert_event_type")
    print("🎉 Mô hình Event Type đã được lưu tại:", ROOT_DIR / "models" / "best_phobert_event_type")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
