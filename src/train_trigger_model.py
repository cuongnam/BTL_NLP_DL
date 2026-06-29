"""
train_trigger_model.py

Fine-tuning PhoBERT/XLM-R cho bài toán Joint Trigger Detection 
và Event Type Classification (Token Classification).
"""

from pathlib import Path
import json
import torch
import numpy as np
import wandb
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# Cấu hình đường dẫn
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "preprocessed" / "trigger"
LABEL_MAP_PATH = ROOT_DIR / "data" / "preprocessed" / "label_maps.json"

class BKEETriggerDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label2id, tokenizer_name="vinai/phobert-base", max_len=256):
        with open(data_path, "r", encoding="utf8") as f:
            self.data = json.load(f)
        self.label2id = label2id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     item = self.data[idx]
    #     tokens = item["tokens"]
    #     labels = item["trigger_labels"]

    #     # Tokenize toàn bộ câu xử lý từ ghép tiếng Việt đã nối bằng gạch dưới hoặc khoảng trắng
    #     encoding = self.tokenizer(
    #         tokens,
    #         is_split_into_words=True,
    #         return_offsets_mapping=True,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=self.max_len
    #     )

    #     # Căn chỉnh nhãn BIO theo subwords (Subword Alignment)
    #     labels_ids = []
    #     word_ids = encoding.word_ids()
    #     previous_word_idx = None
        
    #     for word_idx in word_ids:
    #         if word_idx is None:
    #             labels_ids.append(-100) # Bỏ qua các token đặc biệt khi tính Loss
    #         elif word_idx != previous_word_idx:
    #             # Token đầu tiên của từ gốc
    #             labels_ids.append(self.label2id.get(labels[word_idx], 0))
    #         else:
    #             # Các subword phía sau của từ gốc gán nhãn tương tự hoặc chuyển sang nhãn I-
    #             labels_ids.append(self.label2id.get(labels[word_idx], 0))
    #         previous_word_idx = word_idx

    #     encoding["labels"] = labels_ids
    #     # Chuyển đổi thành Tensor torch
    #     return {k: torch.tensor(v) for k, v in encoding.items() if k != "offset_mapping"}
    def __getitem__(self, idx):
        item = self.data[idx]
        pieces = item["pieces"]
        token_lens = item["token_lens"]
        labels = item["trigger_labels"]

        # 1. Chuyển đổi trực tiếp các subwords (pieces) có sẵn thành ID của PhoBERT
        # Thêm token đặc biệt đầu <s> và cuối </s> theo chuẩn PhoBERT
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(pieces) + [self.tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)

        # 2. Tự căn chỉnh nhãn theo trường token_lens có sẵn của bạn
        # Token đầu <s> nhận nhãn -100 để bỏ qua khi tính Loss
        labels_ids = [-100] 
        
        for word_idx, length in enumerate(token_lens):
            word_label = labels[word_idx]
            label_id = self.label2id.get(word_label, 0)
            
            # Gán nhãn cho subword đầu tiên của từ
            labels_ids.append(label_id)
            
            # Gán nhãn cho các subword tiếp theo của từ đó (nếu từ bị cắt nhỏ)
            # Có thể gán nhãn tương tự hoặc gán nhãn -100 tùy chiến lược, ở đây gán -100 để tránh nhiễu mô hình
            for _ in range(length - 1):
                labels_ids.append(-100) 

        # Token cuối </s> nhận nhãn -100
        labels_ids.append(-100)

        # 3. Padding thủ công để đảm bảo độ dài max_len
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            labels_ids += [-100] * pad_len
        else:
            # Truncate nếu vượt quá max_len
            input_ids = input_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
            labels_ids = labels_ids[:self.max_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels_ids, dtype=torch.long)
        }
    
# def compute_metrics(p, id2label):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)

#     true_predictions = [
#         [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     true_labels = [
#         [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]

#     return {
#         "precision": precision_score(true_labels, true_predictions),
#         "recall": recall_score(true_labels, true_predictions),
#         "f1": f1_score(true_labels, true_predictions)
#     }

def compute_metrics(p, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        pred_list = []
        label_list = []
        for p_id, l_id in zip(prediction, label):
            if l_id != -100:  # Bỏ qua các token padding/subwords
                # Ép khóa tìm kiếm về dạng chuỗi string để khớp với label_maps.json
                p_str = str(p_id)
                l_str = str(l_id)
                
                pred_list.append(id2label.get(p_str, "O"))
                label_list.append(id2label.get(l_str, "O"))
                
        true_predictions.append(pred_list)
        true_labels.append(label_list)

    # Đề phòng trường hợp không dự đoán được nhãn nào, tránh lỗi chia cho 0
    try:
        p_score = precision_score(true_labels, true_predictions)
        r_score = recall_score(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)
    except Exception:
        p_score, r_score, f1 = 0.0, 0.0, 0.0

    return {
        "precision": p_score,
        "recall": r_score,
        "f1": f1
    }

def main():
    # Khởi tạo W&B cho Run 03/Run 04
    wandb.init(project="bkee-event-extraction", name="run_03_phobert_trigger")

    with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
        maps = json.load(f)
    
    trigger_maps = maps["trigger"]
    label2id = trigger_maps["label2id"]
    # id2label = {int(k): v for k, v in trigger_maps["id2label"].items()}
    # THAY BẰNG DÒNG NÀY (Giữ nguyên string key từ JSON):
    id2label = trigger_maps["id2label"]

    # Tải bộ dữ liệu mẫu
    train_dataset = BKEETriggerDataset(DATA_DIR / "train.json", label2id)
    dev_dataset = BKEETriggerDataset(DATA_DIR / "dev.json", label2id)

    # Khởi tạo mô hình PhoBERT
    model = AutoModelForTokenClassification.from_pretrained(
        "vinai/phobert-base", 
        num_labels=len(label2id)
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        eval_strategy="epoch",        #  Đã sửa thành eval_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb" # Tự động đẩy toàn bộ log lên W&B
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForTokenClassification(train_dataset.tokenizer),
        compute_metrics=lambda p: compute_metrics(p, id2label)
    )

    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    main()