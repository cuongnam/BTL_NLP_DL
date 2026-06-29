"""
train_argument_model.py

Fine-tuning XLM-RoBERTa-base cho bài toán Argument Extraction / Role Labeling (Token Classification).
Đã tích hợp cơ chế Mồi đặc trưng Trigger bằng thẻ Marker <tg>...</tg> và mở rộng từ điển XLM-R.
"""

from pathlib import Path
import json
import torch
import numpy as np
import wandb
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# Định nghĩa đường dẫn hệ thống
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "preprocessed" / "argument"
LABEL_MAP_PATH = ROOT_DIR / "data" / "preprocessed" / "label_maps.json"

class BKEEArgumentDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label2id, tokenizer_name="xlm-roberta-base", max_len=256):
        with open(data_path, "r", encoding="utf8") as f:
            self.data = json.load(f)
        self.label2id = label2id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # ĐĂNG KÝ TOKEN ĐẶC BIỆT: Yêu cầu XLM-R Tokenizer không cắt nhỏ thẻ marker
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<tg>', '</tg>']})
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        pieces = item["pieces"]
        token_lens = item["token_lens"]
        labels = item["argument_labels"]
        
        trigger_info = item.get("trigger", {})
        trigger_start = trigger_info.get("start", -1)
        trigger_end = trigger_info.get("end", -1)

        # 1. TẠO CHUỖI PIECES VÀ LABELS MỚI CÓ CHÈN MARKER TRỰC TIẾP
        marked_pieces = []
        marked_labels_ids = []
        
        # Thêm nhãn phòng vệ cho token mở đầu câu <s>
        marked_labels_ids.append(-100) 
        
        for word_idx, length in enumerate(token_lens):
            if word_idx >= len(labels):
                break
                
            # Chèn thẻ mở <tg> ngay trước từ bắt đầu trigger
            if word_idx == trigger_start:
                marked_pieces.append("<tg>")
                marked_labels_ids.append(-100) # Thẻ đặc biệt bỏ qua tính loss
                
            # Lấy thông tin nhãn thực tế của từ
            word_label = labels[word_idx]
            label_id = self.label2id.get(word_label, self.label2id.get("O", 0))
            
            # Tính toán vị trí trích xuất subwords tương ứng
            start_p = sum(token_lens[:word_idx])
            end_p = start_p + length
            word_subwords = pieces[start_p:end_p]
            
            # Nạp subword đầu tiên kèm nhãn thực tế
            if len(word_subwords) > 0:
                marked_pieces.append(word_subwords[0])
                marked_labels_ids.append(label_id)
                
                # Nạp các subword dôi ra phía sau kèm nhãn -100 (Subword alignment)
                for subword in word_subwords[1:]:
                    marked_pieces.append(subword)
                    marked_labels_ids.append(-100)
                
            # Chèn thẻ đóng </tg> ngay sau khi kết thúc từ trigger
            if word_idx == (trigger_end - 1):
                marked_pieces.append("</tg>")
                marked_labels_ids.append(-100)

        # Thêm nhãn phòng vệ cho token kết thúc câu </s>
        marked_labels_ids.append(-100)

        # 2. MÃ HÓA CHUỖI TOKEN THÀNH TENSOR CỦA XLM-ROBERTA
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(marked_pieces) + [self.tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)

        # Đồng bộ ranh giới độ dài mảng
        if len(input_ids) != len(marked_labels_ids):
            min_len = min(len(input_ids), len(marked_labels_ids))
            input_ids = input_ids[:min_len]
            attention_mask = attention_mask[:min_len]
            marked_labels_ids = marked_labels_ids[:min_len]

        # 3. Padding hoặc Truncate về max_len=256
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            marked_labels_ids += [-100] * pad_len
        else:
            input_ids = input_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
            marked_labels_ids = marked_labels_ids[:self.max_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(marked_labels_ids, dtype=torch.long)
        }

def compute_metrics(p, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions)
    }

def main():
    print("MỞ ĐẦU HUẤN LUYỆN ARGUMENT EXTRACTION VỚI XLM-ROBERTA")
    
    # Đọc bản đồ nhãn hệ thống
    with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
        maps = json.load(f)
    
    argument_maps = maps["argument"]
    label2id = argument_maps["label2id"]
    id2label = {int(k): v for k, v in argument_maps["id2label"].items()}

    # Khởi tạo Tokenizer XLM-R phía ngoài và đăng ký Special Token
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<tg>', '</tg>']})

    # Tải các phân mục dữ liệu Train/Dev Dataset
    train_dataset = BKEEArgumentDataset(DATA_DIR / "train.json", label2id, tokenizer_name="xlm-roberta-base")
    dev_dataset = BKEEArgumentDataset(DATA_DIR / "dev.json", label2id, tokenizer_name="xlm-roberta-base")

    # Khởi tạo mô hình XLM-RoBERTa-base
    model = AutoModelForTokenClassification.from_pretrained(
        "xlm-roberta-base", 
        num_labels=len(label2id)
    )
    
    # CHÌA KHÓA: Mở rộng không gian tầng Embedding để mô hình học vector cho cặp thẻ mới
    model.resize_token_embeddings(len(tokenizer))

    # Cấu hình siêu tham số huấn luyện (Chạy 5 Epoch)
    training_args = TrainingArguments(
        # output_dir="./results_argument_xlmr",
        num_train_epochs=5,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        learning_rate=2e-5,             
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs_argument_xlmr",
        logging_steps=10,
        eval_strategy="epoch",        
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForTokenClassification(train_dataset.tokenizer),
        compute_metrics=lambda p: compute_metrics(p, id2label)
    )

    # Kích hoạt tiến trình huấn luyện
    wandb.init(project="bkee-event-extraction", name="run_06_xlmr_argument_markers")
    trainer.train()
    
    # Đóng gói và lưu lại trọng số tốt nhất
    output_model_path = Path(ROOT_DIR) / "models" / "best_xlmr_argument"
    output_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    print(f"Mô hình tốt nhất đã được lưu thành công tại: {output_model_path}")
    
    wandb.finish()

if __name__ == "__main__":
    main()