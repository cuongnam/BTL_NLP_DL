# """
# timeline_generator.py

# Pipeline kết nối trọn bộ 3 mô hình ONNX INT8 + Bộ chuẩn hóa thời gian.
# Trích xuất sự kiện thô từ văn bản đầu vào thành bảng Timeline hoàn chỉnh.
# """

# from pathlib import Path
# import json
# import numpy as np
# import pandas as pd
# import onnxruntime as ort
# from transformers import AutoTokenizer
# from time_normalizer import normalize_vietnamese_time
# from transformers import RobertaTokenizerFast

# # Cấu hình đường dẫn hệ thống
# ROOT_DIR = Path(__file__).resolve().parent.parent
# ONNX_DIR = ROOT_DIR / "models" / "onnx_optimized"
# LABEL_MAP_PATH = ROOT_DIR / "data" / "preprocessed" / "label_maps.json"

# class BKEEEventPipeline:
#     def __init__(self):
#         print("[Pipeline] Đang nạp các bản đồ nhãn và bộ Tokenizer...")
#         # 1. Đọc bản đồ nhãn hệ thống
#         with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
#             maps = json.load(f)
            
#         self.trigger_id2label = {int(k): v for k, v in maps["trigger"]["id2label"].items()}
#         self.event_id2label = {int(k): v for k, v in maps["event_type"]["id2label"].items()}
#         self.argument_id2label = {int(k): v for k, v in maps["argument"]["id2label"].items()}
        
#         # Ép hệ thống nạp phiên bản RobertaTokenizerFast cấu hình theo PhoBERT để hỗ trợ hàm word_ids()
#         # THÊM THAM SỐ add_prefix_space=True để xử lý dữ liệu dạng mảng từ (is_split_into_words)
#         self.phobert_tok = RobertaTokenizerFast.from_pretrained("vinai/phobert-base", add_prefix_space=True)

#         # XLM-RoBERTa mặc định đã hỗ trợ Fast Tokenizer nên giữ nguyên
#         self.xlmr_tok = AutoTokenizer.from_pretrained("xlm-roberta-base", add_prefix_space=True)
#         self.xlmr_tok.add_special_tokens({'additional_special_tokens': ['<tg>', '</tg>']})

#         print("[Pipeline] Đang khởi tạo các phiên Session ONNX Runtime INT8...")
#         # 3. Khởi tạo Session ONNX tốc độ cao chạy trên CPU (Tiết kiệm RAM)
#         self.sess_trigger = ort.InferenceSession(str(ONNX_DIR / "phobert_trigger_int8.onnx"))
#         self.sess_event = ort.InferenceSession(str(ONNX_DIR / "phobert_event_type_int8.onnx"))
#         self.sess_argument = ort.InferenceSession(str(ONNX_DIR / "xlmr_argument_int8.onnx"))
#         print("[Pipeline] Hệ thống Pipeline ONNX đã sẵn sàng hoạt động!")

#     def _predict_onnx(self, session, input_ids, attention_mask):
#         """Hàm trợ giúp thực thi Inference trên ONNX Runtime"""
#         ort_inputs = {
#             'input_ids': np.array(input_ids, dtype=np.int64),
#             'attention_mask': np.array(attention_mask, dtype=np.int64)
#         }
#         logits = session.run(None, ort_inputs)[0]
#         return logits[0] # Lấy mẫu đầu tiên của batch

#     # def extract_events(self, raw_text: str, anchor_date: str = "2026-06-30"):
#     #     """
#     #     Hàm xử lý trung tâm: Trích xuất toàn bộ cấu trúc sự kiện từ một câu văn thô.
#     #     """
#     #     # Tiền xử lý tách từ nhẹ (Giả định văn bản đã được nối từ ghép bằng _ từ chặng 1)
#     #     words = raw_text.strip().split()
#     #     if not words:
#     #         return []

#     #     # --- CHẶNG 1: TRÍCH XUẤT TRIGGER (PhoBERT) ---
#     #     # Tokenize căn chỉnh theo từ ghép tiếng Việt
#     #     encoded_phobert = self.phobert_tok(words, is_split_into_words=True, max_length=256, padding="max_length", truncation=True)
#     #     trigger_logits = self._predict_onnx(self.sess_trigger, [encoded_phobert["input_ids"]], [encoded_phobert["attention_mask"]])
#     #     trigger_preds = np.argmax(trigger_logits, axis=-1)

#     #     # Trích xuất danh sách vị trí Trigger dựa trên nhãn BIO
#     #     word_ids = encoded_phobert.word_ids()
#     #     detected_triggers = []
#     #     current_trigger = None

#     #     for idx, word_idx in enumerate(word_ids):
#     #         if word_idx is None or word_idx >= len(words):
#     #             continue
#     #         pred_id = trigger_preds[idx]
#     #         label = self.trigger_id2label.get(pred_id, "O")
            
#     #         if label.startswith("B-"):
#     #             if current_trigger:
#     #                 detected_triggers.append(current_trigger)
#     #             current_trigger = {"start": word_idx, "end": word_idx + 1, "text": words[word_idx]}
#     #         elif label.startswith("I-") and current_trigger:
#     #             current_trigger["end"] = word_idx + 1
#     #             current_trigger["text"] += " " + words[word_idx]
#     #         else:
#     #             if current_trigger:
#     #                 detected_triggers.append(current_trigger)
#     #                 current_trigger = None
#     #     if current_trigger:
#     #         detected_triggers.append(current_trigger)

#     #     # Lọc bỏ trùng lặp vị trí từ kích hoạt
#     #     unique_triggers = []
#     #     seen_ranges = set()
#     #     for tg in detected_triggers:
#     #         rng = (tg["start"], tg["end"])
#     #         if rng not in seen_ranges:
#     #             seen_ranges.add(rng)
#     #             unique_triggers.append(tg)

#     #     results = []
#     #     # Duyệt qua từng Trigger tìm được để xử lý chặng 2 và chặng 3
#     #     for tg in unique_triggers:
#     #         # --- CHẶNG 2: PHÂN LOẠI LOẠI SỰ KIỆN (PhoBERT Event) ---
#     #         # Sử dụng chung tokenized của PhoBERT ở trên để lấy dự đoán loại sự kiện
#     #         event_logits = self._predict_onnx(self.sess_event, [encoded_phobert["input_ids"]], [encoded_phobert["attention_mask"]])
#     #         event_pred_id = np.argmax(event_logits[word_ids.index(tg["start"])]) # Lấy logits tại vị trí từ kích hoạt
#     #         event_type = self.event_id2label.get(event_pred_id, "O")
            
#     #         if event_type == "O": 
#     #             continue # Bỏ qua nếu mô hình đánh giá đây không phải sự kiện hợp lệ

#     #         # --- CHẶNG 3: TRÍCH XUẤT ARGUMENTS VỚI THẺ MỒI (XLM-RoBERTa) ---
#     #         t_start, t_end = tg["start"], tg["end"]
#     #         marked_words = words[:t_start] + ["<tg>"] + words[t_start:t_end] + ["</tg>"] + words[t_end:]
            
#     #         encoded_xlmr = self.xlmr_tok(marked_words, is_split_into_words=True, max_length=256, padding="max_length", truncation=True)
#     #         arg_logits = self._predict_onnx(self.sess_argument, [encoded_xlmr["input_ids"]], [encoded_xlmr["attention_mask"]])
#     #         arg_preds = np.argmax(arg_logits, axis=-1)

#     #         xlmr_word_ids = encoded_xlmr.word_ids()
#     #         arguments = {}
#     #         curr_arg_type = None
#     #         curr_arg_tokens = []

#     #         for idx, w_idx in enumerate(xlmr_word_ids):
#     #             if w_idx is None or w_idx >= len(marked_words):
#     #                 continue
#     #             # Bỏ qua không trích xuất nhãn trên chính thẻ marker đặc biệt
#     #             if marked_words[w_idx] in ["<tg>", "</tg>"]:
#     #                 continue

#     #             pred_id = arg_preds[idx]
#     #             label = self.argument_id2label.get(pred_id, "O")

#     #             if label.startswith("B-"):
#     #                 if curr_arg_type and curr_arg_tokens:
#     #                     arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")
#     #                 curr_arg_type = label.split("-")[1]
#     #                 curr_arg_tokens = [marked_words[w_idx]]
#     #             elif label.startswith("I-") and curr_arg_type == label.split("-")[1]:
#     #                 curr_arg_tokens.append(marked_words[w_idx])
#     #             else:
#     #                 if curr_arg_type and curr_arg_tokens:
#     #                     arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")
#     #                     curr_arg_type = None
#     #                     curr_arg_tokens = []
#     #         if curr_arg_type and curr_arg_tokens:
#     #             arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")

#     #         # --- CHẶNG CHUẨN HÓA THỜI GIAN NÂNG CAO ---
#     #         # Nếu tìm thấy thành phần thời gian (thường lưu dưới dạng nhãn 'Time' hoặc 'Date' tùy dataset)
#     #         for time_key in ["Time", "Date", "Thời_gian"]:
#     #             if time_key in arguments:
#     #                 raw_time = arguments[time_key]
#     #                 arguments[f"{time_key}_Chuẩn_Hóa"] = normalize_vietnamese_time(raw_time, anchor_date)

#     #         results.append({
#     #             "Trigger": tg["text"].replace("_", " "),
#     #             "Loại Sự Kiện": event_type,
#     #             "Các Tham Thể Trích Xuất": arguments
#     #         })

#     #     return results
#     def extract_events(self, raw_text: str, anchor_date: str = "2026-06-30"):
#         """
#         Phiên bản Hybrid Pipeline: Tự động sửa lỗi lệch Tokenizer 
#         và bổ sung bộ cứu nguy bằng từ khóa nếu mô hình INT8 dự đoán sót.
#         """
#         words = raw_text.strip().split()
#         if not words:
#             return []

#         # --- CHẶNG 1: TRÍCH XUẤT TRIGGER (PhoBERT ONNX INT8) ---
#         encoded_phobert = self.phobert_tok(words, is_split_into_words=True, max_length=256, padding="max_length", truncation=True)
#         trigger_logits = self._predict_onnx(self.sess_trigger, [encoded_phobert["input_ids"]], [encoded_phobert["attention_mask"]])
#         trigger_preds = np.argmax(trigger_logits, axis=-1)

#         word_ids = encoded_phobert.word_ids()
#         detected_triggers = []

#         # 1. Quét từ mô hình AI
#         for idx, word_idx in enumerate(word_ids):
#             if word_idx is None or word_idx >= len(words):
#                 continue
#             pred_id = trigger_preds[idx]
#             label = self.trigger_id2label.get(pred_id, "O")
            
#             if label.upper() != "O" and label != "":
#                 detected_triggers.append({
#                     "start": word_idx,
#                     "end": word_idx + 1,
#                     "text": words[word_idx],
#                     "predicted_label": label
#                 })

#         # 2. CƠ CHẾ CỨU NGUY (Fallback): Nếu AI bị triệt tiêu trọng số, dùng luật để ép bắt từ khóa cốt lõi
#         keywords_rescue = {
#             "bổ_nhiệm": "Thay_đổi_nhân_sự", 
#             "khởi_tố": "Pháp_lý", 
#             "đầu_tư": "Đầu_tư", 
#             "thành_lập": "Thành_lập"
#         }
        
#         if not detected_triggers:
#             for w_idx, w in enumerate(words):
#                 w_clean = w.lower()
#                 if w_clean in keywords_rescue:
#                     detected_triggers.append({
#                         "start": w_idx,
#                         "end": w_idx + 1,
#                         "text": words[w_idx],
#                         "predicted_label": keywords_rescue[w_clean]
#                     })

#         # Khử trùng lặp vị trí
#         unique_triggers = []
#         seen_words = set()
#         for tg in detected_triggers:
#             if tg["start"] not in seen_words:
#                 seen_words.add(tg["start"])
#                 unique_triggers.append(tg)

#         results = []
#         # Duyệt qua từng Trigger để chạy tiếp Chặng 2 & Chặng 3
#         for tg in unique_triggers:
#             # --- CHẶNG 2: PHÂN LOẠI LOẠI SỰ KIỆN ---
#             event_logits = self._predict_onnx(self.sess_event, [encoded_phobert["input_ids"]], [encoded_phobert["attention_mask"]])
#             try:
#                 trigger_token_idx = word_ids.index(tg["start"])
#                 event_pred_id = np.argmax(event_logits[trigger_token_idx])
#                 event_type = self.event_id2label.get(event_pred_id, "O")
#             except ValueError:
#                 event_type = "O"
            
#             if event_type.upper() == "O":
#                 event_type = tg["predicted_label"]

#             # --- CHẶNG 3: TRÍCH XUẤT ARGUMENTS VỚI THÈ MỒI (XLM-RoBERTa ONNX INT8) ---
#             t_start, t_end = tg["start"], tg["end"]
#             marked_words = words[:t_start] + ["<tg>"] + words[t_start:t_end] + ["</tg>"] + words[t_end:]
            
#             encoded_xlmr = self.xlmr_tok(marked_words, is_split_into_words=True, max_length=256, padding="max_length", truncation=True)
#             arg_logits = self._predict_onnx(self.sess_argument, [encoded_xlmr["input_ids"]], [encoded_xlmr["attention_mask"]])
#             arg_preds = np.argmax(arg_logits, axis=-1)

#             xlmr_word_ids = encoded_xlmr.word_ids()
#             arguments = {}
#             curr_arg_type = None
#             curr_arg_tokens = []

#             for idx, w_idx in enumerate(xlmr_word_ids):
#                 if w_idx is None or w_idx >= len(marked_words):
#                     continue
#                 if marked_words[w_idx] in ["<tg>", "</tg>"]:
#                     continue

#                 pred_id = arg_preds[idx]
#                 label = self.argument_id2label.get(pred_id, "O")
#                 clean_label = label.replace("B-", "").replace("I-", "")
                
#                 # Ép lấy nhãn nếu mô hình trích xuất ra thông tin thực tế
#                 if label.startswith("B-") or (label.upper() != "O" and clean_label != curr_arg_type):
#                     if curr_arg_type and curr_arg_tokens:
#                         arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")
#                     curr_arg_type = clean_label
#                     curr_arg_tokens = [marked_words[w_idx]]
#                 elif label.startswith("I-") or (label.upper() != "O" and clean_label == curr_arg_type):
#                     if curr_arg_type:
#                         curr_arg_tokens.append(marked_words[w_idx])
#                 else:
#                     if curr_arg_type and curr_arg_tokens:
#                         arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")
#                         curr_arg_type = None
#                         curr_arg_tokens = []
                        
#             if curr_arg_type and curr_arg_tokens:
#                 arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")

#             # --- CHẶNG CHUẨN HÓA THỜI GIAN NÂNG CAO ---
#             for k in list(arguments.keys()):
#                 if k.lower() in ["time", "date", "thời_gian", "dat"]:
#                     raw_time = arguments[k]
#                     arguments[f"{k}_Chuẩn_Hóa"] = normalize_vietnamese_time(raw_time, anchor_date)

#             results.append({
#                 "Trigger": tg["text"].replace("_", " "),
#                 "Loại Sự Kiện": event_type,
#                 "Các Tham Thể Trích Xuất": arguments
#             })

#         return results

# # Đoạn mã chạy thử nghiệm sinh dữ liệu Timeline trực tiếp
# if __name__ == "__main__":
#     pipeline = BKEEEventPipeline()
    
#     # Định nghĩa danh sách câu test sát với miền dữ liệu Doanh nghiệp của BKEE
#     test_sentences = [
#         # Câu doanh nghiệp 1: Sự kiện bổ nhiệm nhân sự
#         "Ngày 15/6 , Hội_đồng_quản_trị tập_đoàn VinGroup đã bổ_nhiệm ông Nguyễn_Văn_B làm Tổng_giám_đốc mới .",
        
#         # Câu doanh nghiệp 2: Sự kiện đầu tư / mua bán sáp nhập
#         "Công_ty FPT vừa đầu_tư 50 triệu USD vào một doanh_nghiệp công_nghệ tại Mỹ ."
#     ]
    
#     pub_date = "2026-06-30"
#     all_rows = []
    
#     for idx, sample_news in enumerate(test_sentences, 1):
#         print(f"\n======================================")
#         print(f"[Test {idx}] Văn bản đầu vào: {sample_news}")
#         print(f"======================================")
        
#         events = pipeline.extract_events(sample_news, anchor_date=pub_date)
        
#         for ev in events:
#             args = ev["Các Tham Thể Trích Xuất"]
#             time_display = args.get("Time_Chuẩn_Hóa", args.get("Date_Chuẩn_Hóa", pub_date))
#             all_rows.append({
#                 "Mốc Thời Gian": time_display,
#                 "Sự Kiện": ev["Loại Sự Kiện"],
#                 "Từ Kích Hoạt": ev["Trigger"],
#                 "Chi Tiết Tham Thể": json.dumps(args, ensure_ascii=False)
#             })
        
#     df = pd.DataFrame(all_rows)
#     print("\n\n=== BẢNG TIMELINE SỰ KIỆN DOANH NGHIỆP TRÍCH XUẤT THÀNH CÔNG ===")
#     if not df.empty:
#         print(df.to_string(index=False))
#     else:
#         print("🚨 Bảng vẫn trống. Lý do: Mô hình ONNX lượng tử hóa của bạn có thể đã bị triệt tiêu trọng số quá mức trong chặng nén INT8, hoặc câu test vẫn chưa trúng từ khóa trong tập train của bạn.")
"""
timeline_generator_pytorch.py

Pipeline kết nối trọn bộ 3 mô hình PyTorch gốc (FP32) + Bộ chuẩn hóa thời gian.
Dùng để đối chứng và kiểm tra độ chính xác nguyên bản trước khi lượng tử hóa.
"""

from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import RobertaTokenizerFast
from time_normalizer import normalize_vietnamese_time

# Cấu hình đường dẫn hệ thống
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models" / "onnx_optimized"
LABEL_MAP_PATH = ROOT_DIR / "data" / "preprocessed" / "label_maps.json"

class BKEEEventPyTorchPipeline:
    def __init__(self):
        print("[Pipeline PyTorch] Đang nạp các bản đồ nhãn và bộ Tokenizer...")
        # 1. Đọc bản đồ nhãn hệ thống
        with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
            maps = json.load(f)
            
        self.trigger_id2label = {int(k): v for k, v in maps["trigger"]["id2label"].items()}
        self.event_id2label = {int(k): v for k, v in maps["event_type"]["id2label"].items()}
        self.argument_id2label = {int(k): v for k, v in maps["argument"]["id2label"].items()}

        # 2. Khởi tạo Tokenizer kép (Giữ nguyên cấu hình mồi khoảng trắng ẩn)
        self.phobert_tok = RobertaTokenizerFast.from_pretrained("vinai/phobert-base", add_prefix_space=True)
        self.xlmr_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.xlmr_tok.add_special_tokens({'additional_special_tokens': ['<tg>', '</tg>']})

        # 3. Tự động kiểm tra thiết bị phần cứng (Ưu tiên GPU CUDA nếu có trên Kaggle)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pipeline PyTorch] Đang chạy trên thiết bị: {self.device}")

        # 4. Nạp trực tiếp 3 mô hình PyTorch gốc từ thư mục models của bạn
        print("[Pipeline PyTorch] Đang tải các checkpoint mô hình gốc FP32...")
        
        self.model_trigger = AutoModelForTokenClassification.from_pretrained(MODELS_DIR / "best_phobert_trigger").to(self.device)
        self.model_event = AutoModelForTokenClassification.from_pretrained(MODELS_DIR / "best_phobert_event_type").to(self.device)
        self.model_argument = AutoModelForTokenClassification.from_pretrained(MODELS_DIR / "best_xlmr_argument").to(self.device)
        
        # Đồng bộ kích thước embedding của XLM-R vì có thêm thẻ marker <tg>
        self.model_argument.resize_token_embeddings(len(self.xlmr_tok))
        
        # Chuyển tất cả mô hình sang chế độ đánh giá (Tắt Dropout)
        self.model_trigger.eval()
        self.model_event.eval()
        self.model_argument.eval()
        print("[Pipeline PyTorch] Hệ thống mô hình gốc đã sẵn sàng hoạt động 100%!")

    def extract_events(self, raw_text: str, anchor_date: str = "2026-06-30"):
        words = raw_text.strip().split()
        if not words:
            return []

        # --- CHẶNG 1: TRÍCH XUẤT TRIGGER VỚI PHOBERT GỐC ---
        encoded_phobert = self.phobert_tok(words, is_split_into_words=True, return_tensors="pt", max_length=256, padding="max_length", truncation=True)
        # Đẩy dữ liệu vào đúng thiết bị CPU/GPU
        input_ids = encoded_phobert["input_ids"].to(self.device)
        attention_mask = encoded_phobert["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs_trigger = self.model_trigger(input_ids=input_ids, attention_mask=attention_mask)
            trigger_preds = torch.argmax(outputs_trigger.logits, dim=-1).squeeze(0).cpu().numpy()

        word_ids = encoded_phobert.word_ids()
        detected_triggers = []

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx >= len(words):
                continue
            pred_id = trigger_preds[idx]
            label = self.trigger_id2label.get(pred_id, "O")
            
            if label.upper() != "O" and label != "":
                detected_triggers.append({
                    "start": word_idx,
                    "end": word_idx + 1,
                    "text": words[word_idx],
                    "predicted_label": label
                })

        # --- LOẠI BỎ CƠ CHẾ FALLBACK TỪ KHÓA ĐỂ KIỂM TRA ĐỘ THÔNG MINH THỰC CỦA AI ---
        # Khử trùng lặp vị trí từ kích hoạt
        unique_triggers = []
        seen_words = set()
        for tg in detected_triggers:
            if tg["start"] not in seen_words:
                seen_words.add(tg["start"])
                unique_triggers.append(tg)

        results = []
        
        # Duyệt qua từng Trigger tìm được từ mô hình gốc
        for tg in unique_triggers:
            # --- CHẶNG 2: PHAN LOẠI SỰ KIỆN VỚI PHOBERT EVENT GỐC ---
            with torch.no_grad():
                outputs_event = self.model_event(input_ids=input_ids, attention_mask=attention_mask)
                event_logits = outputs_event.logits.squeeze(0).cpu().numpy()
                
            try:
                trigger_token_idx = word_ids.index(tg["start"])
                event_pred_id = np.argmax(event_logits[trigger_token_idx])
                event_type = self.event_id2label.get(event_pred_id, tg["predicted_label"])
            except ValueError:
                event_type = tg["predicted_label"]

            # --- CHẶNG 3: TRÍCH XUẤT ARGUMENTS VỚI THÈ MỒI (XLM-R GỐC) ---
            t_start, t_end = tg["start"], tg["end"]
            marked_words = words[:t_start] + ["<tg>"] + words[t_start:t_end] + ["</tg>"] + words[t_end:]
            
            encoded_xlmr = self.xlmr_tok(marked_words, is_split_into_words=True, return_tensors="pt", max_length=256, padding="max_length", truncation=True)
            xlmr_input_ids = encoded_xlmr["input_ids"].to(self.device)
            xlmr_attention_mask = encoded_xlmr["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs_arg = self.model_argument(input_ids=xlmr_input_ids, attention_mask=xlmr_attention_mask)
                arg_preds = torch.argmax(outputs_arg.logits, dim=-1).squeeze(0).cpu().numpy()

            xlmr_word_ids = encoded_xlmr.word_ids()
            arguments = {}
            curr_arg_type = None
            curr_arg_tokens = []

            for idx, w_idx in enumerate(xlmr_word_ids):
                if w_idx is None or w_idx >= len(marked_words):
                    continue
                if marked_words[w_idx] in ["<tg>", "</tg>"]:
                    continue

                pred_id = arg_preds[idx]
                label = self.argument_id2label.get(pred_id, "O")
                clean_label = label.replace("B-", "").replace("I-", "")
                
                if label.startswith("B-") or (label.upper() != "O" and clean_label != curr_arg_type):
                    if curr_arg_type and curr_arg_tokens:
                        arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")
                    curr_arg_type = clean_label
                    curr_arg_tokens = [marked_words[w_idx]]
                elif label.startswith("I-") or (label.upper() != "O" and clean_label == curr_arg_type):
                    if curr_arg_type:
                        curr_arg_tokens.append(marked_words[w_idx])
                else:
                    if curr_arg_type and curr_arg_tokens:
                        arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")
                        curr_arg_type = None
                        curr_arg_tokens = []
                        
            if curr_arg_type and curr_arg_tokens:
                arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")

            # --- CHẶNG 4: CHUẨN HÓA THỜI GIAN ---
            for k in list(arguments.keys()):
                if k.lower() in ["time", "date", "thời_gian", "dat"]:
                    raw_time = arguments[k]
                    arguments[f"{k}_Chuẩn_Hóa"] = normalize_vietnamese_time(raw_time, anchor_date)

            results.append({
                "Trigger": tg["text"].replace("_", " "),
                "Loại Sự Kiện": event_type,
                "Các Tham Thể Trích Xuất": arguments
            })

        return results

# Đoạn mã chạy thử độc lập (Sanity Test)
if __name__ == "__main__":
    pipeline = BKEEEventPyTorchPipeline()
    # Thử nghiệm câu phức tạp hình sự xem AI gốc có nhận diện được không
    test_text = "Tháng 8/2018 , thẩm_phán Malaysia tuyên_bố đã khởi_tố hai nghi_phạm ."
    print(pipeline.extract_events(test_text))