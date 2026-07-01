# """
# train_trigger_model.py

# Fine-tuning PhoBERT/XLM-R cho bài toán Joint Trigger Detection 
# và Event Type Classification (Token Classification).
# """

# from pathlib import Path
# import json
# import torch
# import numpy as np
# import wandb
# from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
# from transformers import DataCollatorForTokenClassification
# from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


# # Cấu hình đường dẫn
# ROOT_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR = ROOT_DIR / "data" / "preprocessed" / "trigger"
# LABEL_MAP_PATH = ROOT_DIR / "data" / "preprocessed" / "label_maps.json"

# class BKEETriggerDataset(torch.utils.data.Dataset):
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
#         labels = item["trigger_labels"]

#         # === FIX LỖI KÝ TỰ LẠ ▁ TẠI ĐÂY ===
#         # Loại bỏ ký tự đặc biệt "▁" xuất hiện trong quá trình tiền xử lý dữ liệu đầu vào
#         cleaned_pieces = [p.replace("▁", "") for p in pieces]

#         # 1. Chuyển đổi các subwords đã làm sạch thành ID chuẩn của PhoBERT
#         # Thêm token đặc biệt đầu <s> và cuối </s> theo đúng chuẩn PhoBERT
#         input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(cleaned_pieces) + [self.tokenizer.eos_token_id]
#         attention_mask = [1] * len(input_ids)

#         # 2. Tự căn chỉnh nhãn theo trường token_lens có sẵn
#         # Token đầu <s> nhận nhãn -100 để bỏ qua khi tính Loss
#         labels_ids = [-100] 
        
#         for word_idx, length in enumerate(token_lens):
#             word_label = labels[word_idx]
#             label_id = self.label2id.get(word_label, 0)
            
#             # Gán nhãn cho subword đầu tiên của từ
#             labels_ids.append(label_id)
            
#             # Gán nhãn -100 cho các subword tiếp theo của từ đó để tránh làm nhiễu mô hình
#             for _ in range(length - 1):
#                 labels_ids.append(-100) 

#         # Token cuối </s> nhận nhãn -100
#         labels_ids.append(-100)

#         # Kiểm tra tính đồng bộ độ dài mảng dữ liệu cấu trúc
#         if len(input_ids) != len(labels_ids):
#             # Fallback phòng vệ nếu độ dài lệch nhau do khâu tiền xử lý cũ
#             min_len = min(len(input_ids), len(labels_ids))
#             input_ids = input_ids[:min_len]
#             attention_mask = attention_mask[:min_len]
#             labels_ids = labels_ids[:min_len]

#         # 3. Padding hoặc Truncate để đảm bảo độ dài cố định max_len
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

#     for prediction, label in zip(predictions, labels):
#         pred_list = []
#         label_list = []
#         for p_id, l_id in zip(prediction, label):
#             if l_id != -100:  # Bỏ qua các token đặc biệt và token subword thứ 2 trở đi
#                 # Ép khóa tìm kiếm sang dạng chuỗi (string) để khớp tuyệt đối với nhãn của label_maps.json
#                 p_str = str(p_id)
#                 l_str = str(l_id)
                
#                 pred_list.append(id2label.get(p_str, "O"))
#                 label_list.append(id2label.get(l_str, "O"))
                
#         true_predictions.append(pred_list)
#         true_labels.append(label_list)

#     # Đề phòng trường hợp chưa hội tụ ở Epoch đầu, tránh lỗi chia cho 0
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
#     # Khởi tạo Weights & Biases
#     wandb.init(project="bkee-event-extraction", name="run_03_phobert_trigger")

#     with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
#         maps = json.load(f)
    
#     trigger_maps = maps["trigger"]
#     label2id = trigger_maps["label2id"]
#     id2label = trigger_maps["id2label"] # Giữ nguyên string key từ JSON

#     # Tải bộ dữ liệu
#     train_dataset = BKEETriggerDataset(DATA_DIR / "train.json", label2id)
#     dev_dataset = BKEETriggerDataset(DATA_DIR / "dev.json", label2id)

#     # Khởi tạo mô hình PhoBERT
#     model = AutoModelForTokenClassification.from_pretrained(
#         "vinai/phobert-base", 
#         num_labels=len(label2id)
#     )

#     training_args = TrainingArguments(
#         # output_dir="./results",
#         num_train_epochs=5,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         warmup_steps=100,
#         weight_decay=0.01,
#         logging_dir="./logs",
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

#     # Chạy huấn luyện
#     trainer.train()
    
#     # Lưu chính thức mô hình tốt nhất sau khi train xong
#     best_model_path = ROOT_DIR / "models" / "best_phobert_trigger"
#     trainer.save_model(best_model_path)
#     print(f"🎉 Mô hình tốt nhất đã được lưu tại: {best_model_path}")
    
#     wandb.finish()

# if __name__ == "__main__":
#     main()
"""
train_trigger_model.py

Fine-tuning PhoBERT cho bài toán Joint Trigger Detection 
tích hợp kỹ thuật Quantization-Aware Training (QAT) INT8.
"""

from pathlib import Path
import json
import torch
import torch.ao.quantization as quantization  # Thư viện QAT PyTorch FX
import numpy as np
import wandb
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import RobertaTokenizerFast

# Cấu hình đường dẫn
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "preprocessed" / "trigger"
LABEL_MAP_PATH = ROOT_DIR / "data" / "preprocessed" / "label_maps.json"

class BKEETriggerDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label2id, tokenizer=None, tokenizer_name="vinai/phobert-base", max_len=256):
        with open(data_path, "r", encoding="utf8") as f:
            self.data = json.load(f)
        self.label2id = label2id
        self.tokenizer = tokenizer if tokenizer is not None else RobertaTokenizerFast.from_pretrained(
            tokenizer_name,
            add_prefix_space=True
        )
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item["tokens"]
        
        # Trỏ đích xác vào key trigger_labels theo mẫu data thực tế của bạn
        labels = item["trigger_labels"]

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
                label_ids.append(self.label2id.get(label_str, 0))

        encoding["labels"] = label_ids
        return {k: torch.tensor(v) for k, v in encoding.items()}

def compute_metrics(p, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        pred_list = []
        label_list = []
        for p_id, l_id in zip(prediction, label):
            if l_id != -100:  # Bỏ qua các token đặc biệt ([CLS], [SEP], padding)
                # Chuyển đổi ID sang String để khớp chính xác với key từ file JSON bản đồ nhãn
                p_str = str(p_id)
                l_str = str(l_id)
                
                pred_list.append(id2label.get(p_str, "O"))
                label_list.append(id2label.get(l_str, "O"))
                
        true_predictions.append(pred_list)
        true_labels.append(label_list)

    # Tính toán chính xác thông số F1, Precision, Recall cho chuỗi BIO
    p_score = precision_score(true_labels, true_predictions, zero_division=0)
    r_score = recall_score(true_labels, true_predictions, zero_division=0)
    f1 = f1_score(true_labels, true_predictions, zero_division=0)

    return {
        "precision": p_score,
        "recall": r_score,
        "f1": f1
    }

def main():
    with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
        maps = json.load(f)
    trigger_maps = maps["trigger"]
    label2id = trigger_maps["label2id"]
    id2label = {str(k): v for k, v in trigger_maps["id2label"].items()}

    tokenizer = RobertaTokenizerFast.from_pretrained("vinai/phobert-base", add_prefix_space=True)
    train_dataset = BKEETriggerDataset(DATA_DIR / "train.json", label2id, tokenizer=tokenizer)
    dev_dataset = BKEETriggerDataset(DATA_DIR / "dev.json", label2id, tokenizer=tokenizer)
    # THÊM DÒNG NÀY: Khởi tạo tập kiểm thử độc lập
    # test_dataset = BKEETriggerDataset(DATA_DIR / "test.json", label2id, tokenizer=tokenizer)
    # 1. Khởi tạo mô hình nền gốc FP32
    model = AutoModelForTokenClassification.from_pretrained(
        "vinai/phobert-base", 
        num_labels=len(label2id)
    )
    model.resize_token_embeddings(len(tokenizer))

    # ========================================================
    # ========================================================
    # KÍCH HOẠT CONFIG QUANTIZATION-AWARE TRAINING (QAT)
    # ========================================================
    print("--- [QAT] Cấu hình mô hình sang trạng thái Nhận thức Lượng tử hóa ---")
    model.train()
    
    # 1. Áp dụng cấu hình mặc định cho các lớp tuyến tính
    qconfig_linear = quantization.get_default_qat_qconfig('fbgemm')
    model.qconfig = qconfig_linear
    
    # 2. Định nghĩa cấu hình QAT bảo vệ riêng cho lớp nhúng Embedding
    qconfig_embedding = quantization.float_qparams_weight_only_qconfig
    
    # Duyệt qua các modules để gán cấu hình riêng cho Embedding của PhoBERT
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.qconfig = qconfig_embedding
            print(f"-> Đã áp dụng qconfig bảo vệ thành công cho lớp Embedding: {name}")

    # 3. Chuẩn bị mạng đồ thị FakeQuantize
    model = quantization.prepare_qat(model, inplace=True)
    print("--- [QAT] Khởi tạo phân cấp Fake Quantization Nodes thành công! ---")

    training_args = TrainingArguments(
        output_dir=(ROOT_DIR / "models" / "best_phobert_trigger").as_posix(),
        num_train_epochs=10,              # Gợi ý: Tăng lên 10
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,               # Gợi ý: Tăng lên 5e-5 để đẩy Recall
        warmup_ratio=0.1,                 # Gợi ý: Dùng tỉ lệ thay cho cứng 100 steps
        weight_decay=0.01,
        logging_dir="./logs",
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

    wandb.init(project="bkee-event-extraction", name="run_phobert_trigger_qat")
    trainer.train()
    
    # print("\n--- [ĐÁNH GIÁ] Đang quét qua tập dữ liệu TEST.JSON khách quan ---")
    # test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    # print("\n📊 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST CHUẨN:")
    # print(f" -> Test Loss:      {test_metrics.get('test_loss', 'N/A')}")
    # print(f" -> Test Precision: {test_metrics.get('test_precision', 'N/A')}")
    # print(f" -> Test Recall:    {test_metrics.get('test_recall', 'N/A')}")
    # print(f" -> Test F1-Score:  {test_metrics.get('test_f1', 'N/A')}")
    # print("-" * 50)

    # ========================================================
    # CHUYỂN ĐỔI (CONVERT) SANG MÔ HÌNH INT8 THỰC TẾ SAU TRAIN
    # ========================================================
    print("--- [QAT] Đang convert đóng gói đồ thị sang INT8 thực thụ ---")
    model.eval()
    model.to('cpu')  # Thao tác ép kiểu toán tử diễn ra trên CPU an toàn
    quantized_model = quantization.convert(model, inplace=False)
    
    # Lưu mô hình QAT hoàn chỉnh dưới dạng PyTorch Script/Weights
    output_dir = ROOT_DIR / "models" / "best_phobert_trigger"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model.state_dict(), output_dir / "pytorch_model_qat_int8.pt")
    train_dataset.tokenizer.save_pretrained(output_dir)
    print("🎉 Hoàn thành lưu mô hình Trigger QAT thành công!")

if __name__ == "__main__":
    main()