"""
timeline_generator.py

Pipeline kết nối trọn bộ 3 mô hình ONNX INT8 + Bộ chuẩn hóa thời gian.
Trích xuất sự kiện thô từ văn bản đầu vào thành bảng Timeline hoàn chỉnh.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import AutoTokenizer
from time_normalizer import normalize_vietnamese_time

# Cấu hình đường dẫn hệ thống
ROOT_DIR = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT_DIR / "models" / "onnx_optimized"
LABEL_MAP_PATH = ROOT_DIR / "data" / "preprocessed" / "label_maps.json"

class BKEEEventPipeline:
    def __init__(self):
        print("[Pipeline] Đang nạp các bản đồ nhãn và bộ Tokenizer...")
        # 1. Đọc bản đồ nhãn hệ thống
        with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
            maps = json.load(f)
            
        self.trigger_id2label = {int(k): v for k, v in maps["trigger"]["id2label"].items()}
        self.event_id2label = {int(k): v for k, v in maps["event_type"]["id2label"].items()}
        self.argument_id2label = {int(k): v for k, v in maps["argument"]["id2label"].items()}
        
        # 2. Khởi tạo các bộ Tokenizer gốc
        self.phobert_tok = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.xlmr_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.xlmr_tok.add_special_tokens({'additional_special_tokens': ['<tg>', '</tg>']})

        print("[Pipeline] Đang khởi tạo các phiên Session ONNX Runtime INT8...")
        # 3. Khởi tạo Session ONNX tốc độ cao chạy trên CPU (Tiết kiệm RAM)
        self.sess_trigger = ort.InferenceSession(str(ONNX_DIR / "phobert_trigger_int8.onnx"))
        self.sess_event = ort.InferenceSession(str(ONNX_DIR / "phobert_event_type_int8.onnx"))
        self.sess_argument = ort.InferenceSession(str(ONNX_DIR / "xlmr_argument_int8.onnx"))
        print("[Pipeline] Hệ thống Pipeline ONNX đã sẵn sàng hoạt động!")

    def _predict_onnx(self, session, input_ids, attention_mask):
        """Hàm trợ giúp thực thi Inference trên ONNX Runtime"""
        ort_inputs = {
            'input_ids': np.array(input_ids, dtype=np.int64),
            'attention_mask': np.array(attention_mask, dtype=np.int64)
        }
        logits = session.run(None, ort_inputs)[0]
        return logits[0] # Lấy mẫu đầu tiên của batch

    def extract_events(self, raw_text: str, anchor_date: str = "2026-06-30"):
        """
        Hàm xử lý trung tâm: Trích xuất toàn bộ cấu trúc sự kiện từ một câu văn thô.
        """
        # Tiền xử lý tách từ nhẹ (Giả định văn bản đã được nối từ ghép bằng _ từ chặng 1)
        words = raw_text.strip().split()
        if not words:
            return []

        # --- CHẶNG 1: TRÍCH XUẤT TRIGGER (PhoBERT) ---
        # Tokenize căn chỉnh theo từ ghép tiếng Việt
        encoded_phobert = self.phobert_tok(words, is_split_into_words=True, max_length=256, padding="max_length", truncation=True)
        trigger_logits = self._predict_onnx(self.sess_trigger, [encoded_phobert["input_ids"]], [encoded_phobert["attention_mask"]])
        trigger_preds = np.argmax(trigger_logits, axis=-1)

        # Trích xuất danh sách vị trí Trigger dựa trên nhãn BIO
        word_ids = encoded_phobert.word_ids()
        detected_triggers = []
        current_trigger = None

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx >= len(words):
                continue
            pred_id = trigger_preds[idx]
            label = self.trigger_id2label.get(pred_id, "O")
            
            if label.startswith("B-"):
                if current_trigger:
                    detected_triggers.append(current_trigger)
                current_trigger = {"start": word_idx, "end": word_idx + 1, "text": words[word_idx]}
            elif label.startswith("I-") and current_trigger:
                current_trigger["end"] = word_idx + 1
                current_trigger["text"] += " " + words[word_idx]
            else:
                if current_trigger:
                    detected_triggers.append(current_trigger)
                    current_trigger = None
        if current_trigger:
            detected_triggers.append(current_trigger)

        # Lọc bỏ trùng lặp vị trí từ kích hoạt
        unique_triggers = []
        seen_ranges = set()
        for tg in detected_triggers:
            rng = (tg["start"], tg["end"])
            if rng not in seen_ranges:
                seen_ranges.add(rng)
                unique_triggers.append(tg)

        results = []
        # Duyệt qua từng Trigger tìm được để xử lý chặng 2 và chặng 3
        for tg in unique_triggers:
            # --- CHẶNG 2: PHÂN LOẠI LOẠI SỰ KIỆN (PhoBERT Event) ---
            # Sử dụng chung tokenized của PhoBERT ở trên để lấy dự đoán loại sự kiện
            event_logits = self._predict_onnx(self.sess_event, [encoded_phobert["input_ids"]], [encoded_phobert["attention_mask"]])
            event_pred_id = np.argmax(event_logits[word_ids.index(tg["start"])]) # Lấy logits tại vị trí từ kích hoạt
            event_type = self.event_id2label.get(event_pred_id, "O")
            
            if event_type == "O": 
                continue # Bỏ qua nếu mô hình đánh giá đây không phải sự kiện hợp lệ

            # --- CHẶNG 3: TRÍCH XUẤT ARGUMENTS VỚI THẺ MỒI (XLM-RoBERTa) ---
            t_start, t_end = tg["start"], tg["end"]
            marked_words = words[:t_start] + ["<tg>"] + words[t_start:t_end] + ["</tg>"] + words[t_end:]
            
            encoded_xlmr = self.xlmr_tok(marked_words, is_split_into_words=True, max_length=256, padding="max_length", truncation=True)
            arg_logits = self._predict_onnx(self.sess_argument, [encoded_xlmr["input_ids"]], [encoded_xlmr["attention_mask"]])
            arg_preds = np.argmax(arg_logits, axis=-1)

            xlmr_word_ids = encoded_xlmr.word_ids()
            arguments = {}
            curr_arg_type = None
            curr_arg_tokens = []

            for idx, w_idx in enumerate(xlmr_word_ids):
                if w_idx is None or w_idx >= len(marked_words):
                    continue
                # Bỏ qua không trích xuất nhãn trên chính thẻ marker đặc biệt
                if marked_words[w_idx] in ["<tg>", "</tg>"]:
                    continue

                pred_id = arg_preds[idx]
                label = self.argument_id2label.get(pred_id, "O")

                if label.startswith("B-"):
                    if curr_arg_type and curr_arg_tokens:
                        arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")
                    curr_arg_type = label.split("-")[1]
                    curr_arg_tokens = [marked_words[w_idx]]
                elif label.startswith("I-") and curr_arg_type == label.split("-")[1]:
                    curr_arg_tokens.append(marked_words[w_idx])
                else:
                    if curr_arg_type and curr_arg_tokens:
                        arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")
                        curr_arg_type = None
                        curr_arg_tokens = []
            if curr_arg_type and curr_arg_tokens:
                arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")

            # --- CHẶNG CHUẨN HÓA THỜI GIAN NÂNG CAO ---
            # Nếu tìm thấy thành phần thời gian (thường lưu dưới dạng nhãn 'Time' hoặc 'Date' tùy dataset)
            for time_key in ["Time", "Date", "Thời_gian"]:
                if time_key in arguments:
                    raw_time = arguments[time_key]
                    arguments[f"{time_key}_Chuẩn_Hóa"] = normalize_vietnamese_time(raw_time, anchor_date)

            results.append({
                "Trigger": tg["text"].replace("_", " "),
                "Loại Sự Kiện": event_type,
                "Các Tham Thể Trích Xuất": arguments
            })

        return results

# Đoạn mã chạy thử nghiệm sinh dữ liệu Timeline trực tiếp
if __name__ == "__main__":
    pipeline = BKEEEventPipeline()
    
    # Văn bản tin tức giả định đã được tiền xử lý token
    sample_news = "Hôm_qua , Công_an thành_phố Hà_Nội đã khởi_tố đối_tượng Nguyễn_Văn_A về hành_vi lừa_đảo chiếm_đoạt tài_sản ."
    pub_date = "2026-06-30"
    
    print(f"\n[Test] Văn bản đầu vào: {sample_news}")
    events = pipeline.extract_events(sample_news, anchor_date=pub_date)
    
    # Chuyển đổi cấu trúc thành DataFrame dạng bảng Timeline dữ liệu doanh nghiệp
    timeline_rows = []
    for ev in events:
        args = ev["Các Tham Thể Trích Xuất"]
        # Thử tìm trường thời gian đã chuẩn hóa, nếu không có lấy ngày đăng bài báo làm mặc định
        time_display = args.get("Time_Chuẩn_Hóa", args.get("Date_Chuẩn_Hóa", pub_date))
        
        timeline_rows.append({
            "Mốc Thời Gian": time_display,
            "Sự Kiện": ev["Loại Sự Kiện"],
            "Từ Kích Hoạt": ev["Trigger"],
            "Chi Tiết Tham Thể": json.dumps(args, ensure_ascii=False)
        })
        
    df = pd.DataFrame(timeline_rows)
    print("\n=== BẢNG TIMELINE SỰ KIỆN DOANH NGHIỆP TRÍCH XUẤT THÀNH CÔNG ===")
    print(df.to_string(index=False))
    
    # Lưu ra file CSV mẫu
    output_csv = ROOT_DIR / "data" / "timeline_examples.csv"
    df.to_csv(output_csv, index=False, encoding="utf8")
    print(f"\n[Thành công] Đã xuất bảng dòng thời gian ra file: {output_csv}")