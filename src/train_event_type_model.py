# """
# train_event_type_model.py

# Fine-tuning PhoBERT cho bài toán Event Type Classification (Token Classification).
# Đã sửa lỗi đường dẫn DATA_DIR và bổ sung nhãn "O" (ID 33) tường minh.
# """

# from pathlib import Path
# import json
# import torch
# import numpy as np
# import wandb
# from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
# from transformers import DataCollatorForTokenClassification
# from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# # ========================================================
# # 1. CẤU HÌNH ĐƯỜNG DẪN CHUẨN XÁC ĐẾN THƯ MỤC EVENT_TYPE
# # ========================================================
# ROOT_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR = ROOT_DIR / "data" / "preprocessed" / "event_type"
# LABEL_MAP_PATH = ROOT_DIR / "data" / "preprocessed" / "label_maps.json"

# class BKEEEventTypeDataset(torch.utils.data.Dataset):
#     def __init__(self, data_path, label2id, tokenizer_name="vinai/phobert-base", max_len=256):
#         with open(data_path, "r", encoding="utf8") as f:
#             self.data = json.load(f)
        
#         # Sao chép và bổ sung nhãn "O" vào ID 33 (do file label_maps.json gốc thiếu nhãn này)
#         self.label2id = label2id.copy()
#         if "O" not in self.label2id:
#             new_id = max(self.label2id.values()) + 1 if self.label2id else 0
#             self.label2id["O"] = new_id
            
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         pieces = item["pieces"]
#         token_lens = item["token_lens"]
        
#         num_words = len(token_lens)
#         word_labels = ["O"] * num_words  # Mặc định tất cả các từ trong câu ban đầu mang nhãn "O"
        
#         trigger_info = item.get("trigger", {})
#         event_type = item.get("event_type", "O")
        
#         # Đè tên sự kiện thô (ví dụ: "Start-org") vào đúng vị trí của trigger dựa vào start và end
#         if trigger_info and event_type != "O":
#             start_idx = trigger_info.get("start")
#             end_idx = trigger_info.get("end")
            
#             if start_idx is not None and start_idx < num_words:
#                 word_labels[start_idx] = event_type
                
#             if end_idx is not None:
#                 for i in range(start_idx + 1, end_idx):
#                     if i < num_words:
#                         word_labels[i] = event_type

#         # Làm sạch ký tự đặc biệt "▁" để tránh biến thành token <unk>
#         cleaned_pieces = [p.replace("▁", "") for p in pieces]

#         # 1. Mã hóa chuỗi subword thành ID của PhoBERT kèm token đặc biệt <s> và </s>
#         input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(cleaned_pieces) + [self.tokenizer.eos_token_id]
#         attention_mask = [1] * len(input_ids)

#         # 2. Căn chỉnh nhãn Subword Alignment
#         labels_ids = [-100] # Đầu <s> nhận -100
        
#         for word_idx, length in enumerate(token_lens):
#             word_label = word_labels[word_idx]
            
#             # Khớp nhãn lấy ID thực tế (Nếu từ thường "O" -> ID 33, giúp mô hình tính toán Loss ổn định)
#             label_id = self.label2id.get(word_label, self.label2id["O"])
            
#             # Gán nhãn cho subword đầu tiên của từ
#             labels_ids.append(label_id)
            
#             # Các subword phía sau nhận -100 để không tính loss lặp lại
#             for _ in range(length - 1):
#                 labels_ids.append(-100) 

#         labels_ids.append(-100) # Cuối </s> nhận -100

#         # Phòng vệ lệch độ dài mảng
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
    
# def compute_metrics(p, id2label):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)

#     true_predictions = []
#     true_labels = []

#     # Cập nhật bản sao id2label nội bộ để chứa ánh xạ "33": "O" nhằm phục vụ hàm eval
#     local_id2label = id2label.copy()
#     if "33" not in local_id2label:
#         local_id2label["33"] = "O"

#     for prediction, label in zip(predictions, labels):
#         pred_list = []
#         label_list = []
#         for p_id, l_id in zip(prediction, label):
#             if l_id != -100:  
#                 p_str = str(p_id)
#                 l_str = str(l_id)
                
#                 pred_list.append(local_id2label.get(p_str, "O"))
#                 label_list.append(local_id2label.get(l_str, "O"))
                
#         true_predictions.append(pred_list)
#         true_labels.append(label_list)

#     try:
#         p_score = precision_score(true_labels, true_predictions)
#         r_score = recall_score(true_labels, true_predictions)
#         f1 = f1_score(true_labels, true_predictions)
#     except Exception:
#         p_score, r_score, f1 = 0.0, 0.0, 0.0

#     return {
#         "precision": p_score,
#         "recall": r_score,
#         "f1": f1
#     }

# def main():
#     # Khởi tạo Run mới trên W&B
#     wandb.init(project="bkee-event-extraction", name="run_04_phobert_event_type")

#     with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
#         maps = json.load(f)
    
#     event_maps = maps["event_type"]
#     label2id = event_maps["label2id"]
#     id2label = event_maps["id2label"] 

#     # Tải bộ dữ liệu
#     train_dataset = BKEEEventTypeDataset(DATA_DIR / "train.json", label2id)
#     dev_dataset = BKEEEventTypeDataset(DATA_DIR / "dev.json", label2id)

#     # Khởi tạo mô hình PhoBERT với số lượng nhãn đầu ra lớn hơn 1 (để chứa nhãn "O")
#     model = AutoModelForTokenClassification.from_pretrained(
#         "vinai/phobert-base", 
#         num_labels=len(label2id) + 1  # Cộng thêm 1 đại diện cho nhãn "O"
#     )

#     # Thiết lập siêu tham số tối ưu cho không gian 34 nhãn
#     training_args = TrainingArguments(
#         # output_dir="./results_event_type",
#         num_train_epochs=5,
#         per_device_train_batch_size=16, 
#         per_device_eval_batch_size=16,
#         learning_rate=2e-5,             
#         warmup_steps=100,
#         weight_decay=0.01,
#         logging_dir="./logs_event_type",
#         logging_steps=10,
#         eval_strategy="epoch",        
#         save_strategy="epoch",
#         load_best_model_at_end=True,
#         report_to="wandb"
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=dev_dataset,
#         data_collator=DataCollatorForTokenClassification(train_dataset.tokenizer),
#         compute_metrics=lambda p: compute_metrics(p, id2label)
#     )

#     # Chạy huấn luyện lại Run 04 chuẩn hóa
#     trainer.train()
    
#     # Lưu trọng số mô hình tốt nhất
#     best_model_path = ROOT_DIR / "models" / "best_phobert_event_type"
#     trainer.save_model(best_model_path)
#     print(f"🎉 Mô hình Event Type tốt nhất đã được lưu tại: {best_model_path}")
    
#     wandb.finish()

# if __name__ == "__main__":
#     main()
"""
train_event_type_model.py

Fine-tuning PhoBERT phân loại Event Type 
tích hợp kỹ thuật Quantization-Aware Training (QAT) INT8.
"""

from pathlib import Path
import json
import torch
import torch.ao.quantization as quantization
import numpy as np
import wandb
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
# THÊM IMPORT NÀY Ở ĐẦU FILE
from transformers import RobertaTokenizerFast


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "preprocessed" / "event_type"
LABEL_MAP_PATH = ROOT_DIR / "data" / "preprocessed" / "label_maps.json"

class BKEEEventTypeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label2id, tokenizer_name="vinai/phobert-base", max_len=256):
        with open(data_path, "r", encoding="utf8") as f:    
            self.data = json.load(f)
        
        self.label2id = label2id.copy()
        if "O" not in self.label2id:
            self.label2id["O"] = 33

        # SỬA DÒNG NÀY: Ép buộc sử dụng phiên bản Fast để có hàm word_ids()
        # SỬA DÒNG NÀY: Bổ sung tham số add_prefix_space=True
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            tokenizer_name, 
            add_prefix_space=True
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item["tokens"]
        
        # Khởi tạo danh sách nhãn nền "O"
        labels = ["O"] * len(words)
        
        # Đọc từ trường trigger.start và event_type của data thực tế chặng Event
        if "trigger" in item and "event_type" in item:
            start_idx = item["trigger"]["start"]
            event_str = item["event_type"]
            if start_idx < len(labels):
                labels[start_idx] = event_str

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None
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
        return {k: torch.tensor(v) for k, v in encoding.items()}

def compute_metrics(p, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label.get(p_id, "O") for (p_id, l_id) in zip(prediction, label) if l_id != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label.get(l_id, "O") for (p_id, l_id) in zip(prediction, label) if l_id != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions, zero_division=0),
        "recall": recall_score(true_labels, true_predictions, zero_division=0),
        "f1": f1_score(true_labels, true_predictions, zero_division=0)
    }

def main():
    with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
        maps = json.load(f)
    event_maps = maps["event_type"]
    label2id = event_maps["label2id"]
    
    if "O" not in label2id:
        label2id["O"] = 33
        
    id2label = {int(k): v for k, v in event_maps["id2label"].items()}
    id2label[33] = "O"

    train_dataset = BKEEEventTypeDataset(DATA_DIR / "train.json", label2id)
    dev_dataset = BKEEEventTypeDataset(DATA_DIR / "dev.json", label2id)

    model = AutoModelForTokenClassification.from_pretrained(
        "vinai/phobert-base", 
        num_labels=len(label2id)
    )

    # ========================================================
    # ========================================================
    # KÍCH HOẠT CONFIG QUANTIZATION-AWARE TRAINING (QAT)
    # ========================================================
    print("--- [QAT] Cấu hình mô hình sang trạng thái Nhận thức Lượng tử hóa ---")
    model.train()
    
    # 1. Cấu hình lớp Tuyến tính
    qconfig_linear = quantization.get_default_qat_qconfig('fbgemm')
    model.qconfig = qconfig_linear
    
    # 2. Cấu hình lớp Embedding nhằm tránh lỗi AssertionError khi convert
    qconfig_embedding = quantization.float_qparams_weight_only_qconfig
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.qconfig = qconfig_embedding
            print(f"-> Đã áp dụng qconfig bảo vệ thành công cho lớp Embedding: {name}")

    # 3. Khởi tạo Fake Quantization
    model = quantization.prepare_qat(model, inplace=True)
    print("--- [QAT] Khởi tạo phân cấp Fake Quantization Nodes thành công! ---")

    training_args = TrainingArguments(
        output_dir=(ROOT_DIR / "models" / "best_phobert_event_type").as_posix(),
        num_train_epochs=5,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        learning_rate=2e-5,             
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

    wandb.init(project="bkee-event-extraction", name="run_phobert_event_qat")
    trainer.train()
    
    # ========================================================
    # CHUYỂN ĐỔI (CONVERT) SANG MÔ HÌNH INT8 SAU TRAIN
    # ========================================================
    print("--- [QAT] Đang convert đóng gói đồ thị sang INT8 thực thụ ---")
    model.eval()
    model.to('cpu')
    quantized_model = quantization.convert(model, inplace=False)

    output_dir = ROOT_DIR / "models" / "best_phobert_event_type"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model.state_dict(), output_dir / "pytorch_model_qat_int8.pt")
    train_dataset.tokenizer.save_pretrained(output_dir)
    print("🎉 Hoàn thành lưu mô hình Event Type QAT thành công!")

if __name__ == "__main__":
    main()