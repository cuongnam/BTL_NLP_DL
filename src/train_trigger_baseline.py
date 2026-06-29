"""
train_trigger_baseline.py

Mô hình Baseline cho Trigger Detection:
- Run 01: Rule-based (Dictionary Matching)
- Run 02: CRF (Conditional Random Fields)

Tích hợp theo dõi hiệu năng qua Weights & Biases (W&B).
"""

from pathlib import Path
import json
import logging
import wandb
from collections import Counter
from sklearn_crfsuite import CRF
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# Cấu hình log
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cấu hình đường dẫn
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "preprocessed" / "trigger"

# ============================================================
# 1. ĐỌC DỮ LIỆU
# ============================================================
def load_json_data(path: Path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

# ============================================================
# RUN 01: RULE-BASED TRIGGER DETECTION
# ============================================================
def run_01_rule_based(train_data, test_data):
    logger.info("--- BẮT ĐẦU RUN 01: RULE-BASED TRIGGER ---")
    
    # Khởi tạo W&B Run
    wandb.init(project="bkee-event-extraction", name="run_01_rule_based_trigger")
    
    # Bước 1: Học từ điển trigger từ tập Train
    trigger_lexicon = set()
    for sample in train_data:
        for token, label in zip(sample["tokens"], sample["trigger_labels"]):
            if label != "O":
                trigger_lexicon.add(token.lower())
                
    logger.info(f"Khai phá được {len(trigger_lexicon)} từ khóa kích hoạt từ tập Train.")
    
    # Bước 2: Dự đoán trên tập Test/Val dựa trên từ điển
    y_true = []
    y_pred = []
    
    for sample in test_data:
        true_labels = sample["trigger_labels"]
        pred_labels = []
        
        for token in sample["tokens"]:
            # Nếu từ nằm trong từ điển, gán nhãn B-TRIGGER (luật đơn giản)
            if token.lower() in trigger_lexicon:
                pred_labels.append("B-TRIGGER")
            else:
                pred_labels.append("O")
                
        y_true.append(true_labels)
        y_pred.append(pred_labels)
        
    # Bước 3: Đánh giá và Log lên W&B
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    logger.info(f"Run 01 Kết quả - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    wandb.log({"precision": precision, "recall": recall, "f1": f1})
    wandb.finish()

# ============================================================
# RUN 02: CRF TRIGGER DETECTION (HỌC MÁY TRUYỀN THỐNG)
# ============================================================
def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True # Khởi đầu câu

    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True # Kết thúc câu
        
    return features

def sent2features(sent):
    return [word2features(sent, i)  for i in range(len(sent))]

def run_02_crf(train_data, test_data):
    logger.info("--- BẮT ĐẦU RUN 02: CRF TRIGGER ---")
    
    # Khởi tạo W&B Run
    wandb.init(project="bkee-event-extraction", name="run_02_bilstm_crf_trigger")
    
    # Chuẩn bị dữ liệu đặc trưng
    X_train = [sent2features(s["tokens"]) for s in train_data]
    y_train = [s["trigger_labels"] for s in train_data]
    
    X_test = [sent2features(s["tokens"]) for s in test_data]
    y_test = [s["trigger_labels"] for s in test_data]
    
    # Huấn luyện mô hình CRF
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    logger.info("Đang huấn luyện mô hình CRF...")
    crf.fit(X_train, y_train)
    
    # Dự đoán
    y_pred = crf.predict(X_test)
    
    # Đánh giá và Log lên W&B
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    logger.info(f"Run 02 Kết quả - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    wandb.log({"precision": precision, "recall": recall, "f1": f1})
    wandb.finish()

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    # Tải dữ liệu preprocessed
    train_data = load_json_data(DATA_DIR / "train.json")
    test_data = load_json_data(DATA_DIR / "test.json") # Có thể đổi thành dev.json tùy mục đích kiểm thử
    
    # Chạy Run 01
    run_01_rule_based(train_data, test_data)
    
    # Chạy Run 02
    run_02_crf(train_data, test_data)

if __name__ == "__main__":
    main()