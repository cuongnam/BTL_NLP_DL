"""
train_argument_model.py

Fine-tuning PhoBERT cho bài toán Argument Extraction / Role Labeling (Token Classification).
Đã đồng bộ chính xác theo cấu trúc BIO từ mảng argument_labels.
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

# class BKEEArgumentDataset(torch.utils.data.Dataset):
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
#         labels = item["argument_labels"] # Khớp chuẩn xác với sample thực tế

#         # Làm sạch ký tự đặc biệt "▁" tương tự các run trước
#         cleaned_pieces = [p.replace("▁", "") for p in pieces]

#         # 1. Mã hóa chuỗi subword về ID của PhoBERT kèm token đặc biệt <s> và </s>
#         input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(cleaned_pieces) + [self.tokenizer.eos_token_id]
#         attention_mask = [1] * len(input_ids)

#         # 2. Căn chỉnh nhãn Subword Alignment
#         labels_ids = [-100]  # Token đầu <s> nhận -100 để bỏ qua khi tính Loss
        
#         for word_idx, length in enumerate(token_lens):
#             word_label = labels[word_idx]
            
#             # Tra cứu ID nhãn BIO thực tế từ nhánh "argument" trong file label_maps.json
#             label_id = self.label2id.get(word_label, self.label2id.get("O", 0))
            
#             # Gán nhãn cho subword đầu tiên cấu thành nên từ gốc
#             labels_ids.append(label_id)
            
#             # Gán nhãn -100 cho phần đuôi từ bị cắt nhỏ để tránh làm nhiễu mô hình
#             for _ in range(length - 1):
#                 labels_ids.append(-100) 

#         labels_ids.append(-100)  # Token cuối </s> nhận -100

#         # Phòng vệ lệch độ dài mảng cấu trúc dữ liệu
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
class BKEEArgumentDataset(torch.utils.data.Dataset):
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
        labels = item["argument_labels"]
        
        # Lấy thông tin vị trí từ kích hoạt gốc ban đầu
        trigger_info = item.get("trigger", {})
        trigger_start = trigger_info.get("start", -1)
        trigger_end = trigger_info.get("end", -1)

        # Làm sạch ký tự đặc biệt "▁"
        cleaned_pieces = [p.replace("▁", "") for p in pieces]

        # 1. Mã hóa chuỗi subword thành ID kèm token đặc biệt <s> và </s>
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(cleaned_pieces) + [self.tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)

        # 2. Căn chỉnh nhãn Subword Alignment bằng cách theo dõi index từ gốc
        labels_ids = [-100]  # Đầu câu <s> nhận -100
        
        original_word_idx = 0 
        
        # Sử dụng biến cờ để xác định xem marker đã xuất hiện hay chưa dựa trên logic nạp của file preprocess
        for length in token_lens:
            
            # KIỂM TRA ĐIỀU KIỆN RANH GIỚI:
            # Nếu original_word_idx đã chạy hết mảng nhãn gốc, mọi token dôi ra phía sau đều là marker hoặc ký tự đặc biệt bổ sung
            if original_word_idx >= len(labels):
                for _ in range(length):
                    labels_ids.append(-100)
                continue
                
            # Kiểm tra xem từ hiện tại có phải là marker dựa trên vị trí từ thực tế
            # File preprocess chèn <tg> ngay TẠI vị trí trigger_start, và chèn </tg> SAU vị trí (trigger_end - 1)
            # Nhận diện marker vì độ dài token_lens của marker luôn bằng 1 và không khớp với vị trí từ thông thường.
            
            # Logic phòng vệ chuẩn: Nếu độ dài là 1 và vị trí lặp hiện tại tương ứng với điểm chèn marker
            if length == 1 and (original_word_idx == trigger_start or original_word_idx == trigger_end):
                # Đây là token marker đặc biệt (<tg> hoặc </tg>), gán nhãn -100 (Bỏ qua không tính loss)
                labels_ids.append(-100)
                # KHÔNG tăng original_word_idx vì đây không phải từ gốc trong câu
            else:
                # Đây là từ nội dung gốc trong câu văn
                word_label = labels[original_word_idx]
                label_id = self.label2id.get(word_label, self.label2id.get("O", 0))
                
                # Gán nhãn thực tế cho subword đầu tiên của từ
                labels_ids.append(label_id)
                
                # Gán nhãn -100 cho các subwords thừa (nếu từ bị phân rã thành nhiều mảnh nhỏ)
                for _ in range(length - 1):
                    labels_ids.append(-100)
                
                # Hoàn thành 1 từ gốc thành công -> tăng index để chuyển sang từ tiếp theo
                original_word_idx += 1

        labels_ids.append(-100)  # Cuối câu </s> nhận -100

        # Phòng vệ lệch độ dài mảng cấu trúc dữ liệu do rút gọn
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
                
                # Trích xuất nhãn BIO dạng chuỗi hợp lệ cho thư viện seqeval
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
    # Khởi tạo Run 05 trên Weights & Biases
    wandb.init(project="bkee-event-extraction", name="run_05_phobert_argument")

    with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
        maps = json.load(f)
    
    # Nạp cấu trúc nhánh "argument" từ label_maps.json (Bao gồm các nhãn BIO thực tế)
    argument_maps = maps["argument"]
    label2id = argument_maps["label2id"]
    id2label = argument_maps["id2label"] 

    # Khởi tạo bộ Dataset tập Train và tập Dev tương ứng
    train_dataset = BKEEArgumentDataset(DATA_DIR / "train.json", label2id)
    dev_dataset = BKEEArgumentDataset(DATA_DIR / "dev.json", label2id)

    # Khởi tạo mô hình PhoBERT khớp chính xác với số lượng nhãn BIO của tham thể (không cần +1)
    model = AutoModelForTokenClassification.from_pretrained(
        "vinai/phobert-base", 
        num_labels=len(label2id)
    )

    # Siêu tham số tối ưu hóa cho bài toán trích xuất thực thể/vai trò
    training_args = TrainingArguments(
        output_dir="./results_argument",
        num_train_epochs=5,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        learning_rate=2e-5,             
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs_argument",
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

    # Thực hiện quá trình huấn luyện Run 05
    trainer.train()
    
    # Lưu trọng số mô hình tham thể tối ưu nhất
    best_model_path = ROOT_DIR / "models" / "best_phobert_argument"
    trainer.save_model(best_model_path)
    print(f"🎉 Mô hình Argument Extraction tốt nhất đã được lưu tại: {best_model_path}")
    
    wandb.finish()

if __name__ == "__main__":
    main()