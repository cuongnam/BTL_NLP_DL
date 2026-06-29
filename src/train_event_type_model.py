"""
train_event_type_model.py

Fine-tuning PhoBERT cho bài toán Event Type Classification (Token Classification).
Kế thừa cấu trúc chuẩn hóa token từ bài toán Trigger Detection.
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
DATA_DIR = ROOT_DIR / "data" / "preprocessed" / "trigger" # Sử dụng chung luồng data đã có trường pieces
LABEL_MAP_PATH = ROOT_DIR / "data" / "preprocessed" / "label_maps.json"

# class BKEEEventTypeDataset(torch.utils.data.Dataset):
#     def __init__(self, data_path, label2id, tokenizer_name="vinai/phobert-base", max_len=256):
#         with open(data_path, "r", encoding="utf8") as f:
#             self.data = json.load(f)
#         self.label2id = label2id
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         pieces = item["pieces"]
#         token_lens = item["token_lens"]
        
#         # LƯU Ý BÀI TOÁN EVENT TYPE: 
#         # Nếu cấu trúc file data của bạn lưu trường nhãn loại sự kiện là "event_type_labels", hãy đổi tên trường dưới đây.
#         # Nếu khâu tiền xử lý của bạn lưu chung nhãn chuỗi BIO dạng B-Appeal, I-Appeal vào "trigger_labels", giữ nguyên "trigger_labels".
#         labels = item.get("event_type_labels", item["trigger_labels"])

#         # Làm sạch ký tự đặc biệt "▁" tương tự như Run 03
#         cleaned_pieces = [p.replace("▁", "") for p in pieces]

#         # 1. Đưa về ID chuẩn hóa của PhoBERT kèm token đặc biệt <s> và </s>
#         input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(cleaned_pieces) + [self.tokenizer.eos_token_id]
#         attention_mask = [1] * len(input_ids)

#         # 2. Căn chỉnh nhãn Subword Alignment
#         labels_ids = [-100] # Đầu <s> nhận -100
        
#         for word_idx, length in enumerate(token_lens):
#             word_label = labels[word_idx]
            
#             # Khớp nhãn dựa trên label2id của danh sách 33 sự kiện
#             label_id = self.label2id.get(word_label, self.label2id.get("O", 0))
            
#             # Gán nhãn cho subword đầu tiên của từ gốc
#             labels_ids.append(label_id)
            
#             # Gán nhãn -100 cho phần đuôi từ bị cắt nhỏ để tránh nhiễu Loss
#             for _ in range(length - 1):
#                 labels_ids.append(-100) 

#         labels_ids.append(-100) # Cuối </s> nhận -100

#         # Phòng vệ lệch độ dài dữ liệu cấu trúc
#         if len(input_ids) != len(labels_ids):
#             min_len = min(len(input_ids), len(labels_ids))
#             input_ids = input_ids[:min_len]
#             attention_mask = attention_mask[:min_len]
#             labels_ids = labels_ids[:min_len]

#         # 3. Padding hoặc Truncate về max_len=256
#         pad_len = self.max_len - len(input_ids)
#         if pad_len > 0:
#             input_ids += [self.tokenizer.pad_token_id] * pad_len
#             attention_mask += [0] * pad_len
#             labels_ids += [-100] * pad_len
#         else:
#             input_ids = input_ids[:self.max_len]
#             attention_mask = attention_mask[:self.max_len]
#             labels_ids = labels_ids[:self.max_len]

#         return {
#             "input_ids": torch.tensor(input_ids, dtype=torch.long),
#             "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
#             "labels": torch.tensor(labels_ids, dtype=torch.long)
#         }
class BKEEEventTypeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label2id, tokenizer_name="vinai/phobert-base", max_len=256):
        with open(data_path, "r", encoding="utf8") as f:
            self.data = json.load(f)
        self.label2id = label2id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        pieces = item["pieces"]
        token_lens = item["token_lens"]
        
        # === DỰNG LẠI CHUỖI NHÃN DẠNG THÔ THEO SAMPLE ===
        num_words = len(token_lens)
        # Sử dụng nhãn "O" đại diện cho từ thường. Do label2id không có "O", 
        # lát nữa chúng ta sẽ map "O" thành -100 để bỏ qua khi tính Loss.
        word_labels = ["O"] * num_words  
        
        trigger_info = item.get("trigger", {})
        event_type = item.get("event_type", "O")
        
        if trigger_info and event_type != "O":
            start_idx = trigger_info.get("start")
            end_idx = trigger_info.get("end")
            
            # Gán trực tiếp tên sự kiện thô (Raw) vào đúng vị trí của trigger trong câu
            if start_idx is not None and start_idx < num_words:
                word_labels[start_idx] = event_type
                
            if end_idx is not None:
                for i in range(start_idx + 1, end_idx):
                    if i < num_words:
                        word_labels[i] = event_type

        # Làm sạch ký tự đặc biệt "▁" tương tự Run 03
        cleaned_pieces = [p.replace("▁", "") for p in pieces]

        # 1. Mã hóa chuỗi subword thành ID của PhoBERT kèm <s> và </s>
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(cleaned_pieces) + [self.tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)

        # 2. Căn chỉnh nhãn Subword Alignment khớp với label_maps.json thô
        labels_ids = [-100] # Đầu <s> nhận -100
        
        for word_idx, length in enumerate(token_lens):
            word_label = word_labels[word_idx]
            
            if word_label == "O":
                # VÌ TRONG FILE JSON KHÔNG CÓ NHÃN "O", ta gán -100 để mô hình bỏ qua, không tính loss tại các từ thường này.
                # Cách này giúp PhoBERT tập trung học cực tốt các từ kích hoạt sự kiện thực tế.
                label_id = -100 
            else:
                # Khớp tên sự kiện thô (ví dụ: "Start-org") với ID tương ứng (26)
                label_id = self.label2id.get(word_label, -100)
            
            # Gán nhãn cho subword đầu tiên của từ
            labels_ids.append(label_id)
            
            # Các subword phía sau gán -100 để tránh làm nhiễu mô hình
            for _ in range(length - 1):
                labels_ids.append(-100) 

        labels_ids.append(-100) # Cuối </s> nhận -100

        # Phòng vệ lệch độ dài mảng
        if len(input_ids) != len(labels_ids):
            min_len = min(len(input_ids), len(labels_ids))
            input_ids = input_ids[:min_len]
            attention_mask = attention_mask[:min_len]
            labels_ids = labels_ids[:min_len]

        # 3. Padding hoặc Truncate về max_len=256
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            labels_ids += [-100] * pad_len
        else:
            input_ids = input_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
            labels_ids = labels_ids[:self.max_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels_ids, dtype=torch.long)
        }

def compute_metrics(p, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        pred_list = []
        label_list = []
        for p_id, l_id in zip(prediction, label):
            if l_id != -100:  
                p_str = str(p_id)
                l_str = str(l_id)
                
                # Trích xuất nhãn chuỗi (String) từ ánh xạ id2label của event_type
                pred_list.append(id2label.get(p_str, "O"))
                label_list.append(id2label.get(l_str, "O"))
                
        true_predictions.append(pred_list)
        true_labels.append(label_list)

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
    # Khởi tạo tiến trình lưu trữ đồ thị mới trên W&B cho Run 04
    wandb.init(project="bkee-event-extraction", name="run_04_phobert_event_type")

    with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
        maps = json.load(f)
    
    # === THAY ĐỔI CỐT LÕI: LẤY NHÁNH EVENT_TYPE CHỨ KHÔNG LẤY TRIGGER ===
    event_maps = maps["event_type"]
    label2id = event_maps["label2id"]
    id2label = event_maps["id2label"] 

    # Tải bộ dữ liệu áp dụng cho lớp Event Type Dataset mới
    train_dataset = BKEEEventTypeDataset(DATA_DIR / "train.json", label2id)
    dev_dataset = BKEEEventTypeDataset(DATA_DIR / "dev.json", label2id)

    # Khởi tạo mô hình PhoBERT với num_labels tương ứng số loại sự kiện (33 nhãn)
    model = AutoModelForTokenClassification.from_pretrained(
        "vinai/phobert-base", 
        num_labels=len(label2id)
    )

    # Thiết lập siêu tham số tối ưu (Tăng nhẹ học vị và cấu hình Batch Size)
    training_args = TrainingArguments(
        output_dir="./results_event_type",
        num_train_epochs=5,
        per_device_train_batch_size=16, # Tăng lên 16 giúp việc gom cụm 33 nhãn ổn định hơn
        per_device_eval_batch_size=16,
        learning_rate=2e-5,             # Đặt LR tiêu chuẩn của PhoBERT để tránh tụt Loss chậm
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs_event_type",
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

    # Thực hiện quá trình huấn luyện Run 04
    trainer.train()
    
    # Lưu trọng số mô hình phân loại sự kiện tối ưu nhất về thư mục riêng
    best_model_path = ROOT_DIR / "models" / "best_phobert_event_type"
    trainer.save_model(best_model_path)
    print(f"🎉 Mô hình Event Type tốt nhất đã được lưu tại: {best_model_path}")
    
    wandb.finish()

if __name__ == "__main__":
    main()