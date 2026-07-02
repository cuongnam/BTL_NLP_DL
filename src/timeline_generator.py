"""
timeline_generator.py

Pipeline kết nối mô hình ONNX INT8 đã được xuất từ export_onnx.py,
trích xuất trigger, event type và arguments rồi chuẩn hóa thời gian.
"""

from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, RobertaTokenizerFast
from time_normalizer import normalize_vietnamese_time

ROOT_DIR = Path(__file__).resolve().parent.parent
ONNX_DIR = ROOT_DIR / "models" / "onnx_optimized"
LABEL_MAP_PATH = ROOT_DIR / "data" / "preprocessed" / "label_maps.json"


class BKEEEventPipeline:
    def __init__(self):
        print("[Pipeline ONNX] Loading label maps and tokenizers...")
        with open(LABEL_MAP_PATH, "r", encoding="utf8") as f:
            maps = json.load(f)

        self.trigger_id2label = {int(k): v for k, v in maps["trigger"]["id2label"].items()}
        self.event_id2label = {int(k): v for k, v in maps["event_type"]["id2label"].items()}
        self.argument_id2label = {int(k): v for k, v in maps["argument"]["id2label"].items()}

        self.phobert_tok = RobertaTokenizerFast.from_pretrained("vinai/phobert-base", add_prefix_space=True)
        self.xlmr_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.xlmr_tok.add_special_tokens({"additional_special_tokens": ["<tg>", "</tg>"]})

        print("[Pipeline ONNX] Initializing ONNX Runtime sessions...")
        self.sess_trigger = ort.InferenceSession(str(ONNX_DIR / "phobert_trigger_int8.onnx"))
        self.sess_event = ort.InferenceSession(str(ONNX_DIR / "phobert_event_type_int8.onnx"))
        self.sess_argument = ort.InferenceSession(str(ONNX_DIR / "xlmr_argument_int8.onnx"))
        print("[Pipeline ONNX] Ready.")

    def _predict_onnx(self, session, input_ids, attention_mask):
        ort_inputs = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64),
        }
        logits = session.run(None, ort_inputs)[0]
        return logits[0]

    def extract_events(self, raw_text: str, anchor_date: str = "2026-06-30"):
        words = raw_text.strip().split()
        if not words:
            return []

        encoded_phobert = self.phobert_tok(
            words,
            is_split_into_words=True,
            max_length=256,
            padding="max_length",
            truncation=True,
        )

        trigger_logits = self._predict_onnx(
            self.sess_trigger,
            [encoded_phobert["input_ids"]],
            [encoded_phobert["attention_mask"]],
        )
        trigger_preds = np.argmax(trigger_logits, axis=-1)
        word_ids = encoded_phobert.word_ids()

        detected_triggers = []
        current_trigger = None

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx >= len(words):
                continue
            pred_id = int(trigger_preds[idx])
            label = self.trigger_id2label.get(pred_id, "O")
            if label.startswith("B-"):
                if current_trigger:
                    detected_triggers.append(current_trigger)
                current_trigger = {
                    "start": word_idx,
                    "end": word_idx + 1,
                    "text": words[word_idx],
                    "predicted_label": label,
                }
            elif label.startswith("I-") and current_trigger:
                current_trigger["end"] = word_idx + 1
                current_trigger["text"] += " " + words[word_idx]
            else:
                if current_trigger:
                    detected_triggers.append(current_trigger)
                    current_trigger = None

        if current_trigger:
            detected_triggers.append(current_trigger)

        # if not detected_triggers:
        #     keywords_rescue = {
        #         "bổ_nhiệm": "Thay_đổi_nhân_sự",
        #         "khởi_tố": "Pháp_lý",
        #         "đầu_tư": "Đầu_tư",
        #         "thành_lập": "Thành_lập",
        #     }
        #     for w_idx, w in enumerate(words):
        #         w_clean = w.lower()
        #         if w_clean in keywords_rescue:
        #             detected_triggers.append({
        #                 "start": w_idx,
        #                 "end": w_idx + 1,
        #                 "text": words[w_idx],
        #                 "predicted_label": keywords_rescue[w_clean],
        #             })

        unique_triggers = []
        seen_ranges = set()
        for tg in detected_triggers:
            rng = (tg["start"], tg["end"])
            if rng not in seen_ranges:
                seen_ranges.add(rng)
                unique_triggers.append(tg)

        results = []
        for tg in unique_triggers:
            event_logits = self._predict_onnx(
                self.sess_event,
                [encoded_phobert["input_ids"]],
                [encoded_phobert["attention_mask"]],
            )

            try:
                trigger_token_idx = word_ids.index(tg["start"])
                event_pred_id = int(np.argmax(event_logits[trigger_token_idx]))
                event_type = self.event_id2label.get(event_pred_id, tg["predicted_label"])
            except ValueError:
                event_type = tg["predicted_label"]

            if event_type.upper() == "O":
                event_type = tg["predicted_label"]

            t_start, t_end = tg["start"], tg["end"]
            marked_words = words[:t_start] + ["<tg>"] + words[t_start:t_end] + ["</tg>"] + words[t_end:]

            encoded_xlmr = self.xlmr_tok(
                marked_words,
                is_split_into_words=True,
                max_length=256,
                padding="max_length",
                truncation=True,
            )
            arg_logits = self._predict_onnx(
                self.sess_argument,
                [encoded_xlmr["input_ids"]],
                [encoded_xlmr["attention_mask"]],
            )
            arg_preds = np.argmax(arg_logits, axis=-1)
            xlmr_word_ids = encoded_xlmr.word_ids()

            arguments = {}
            curr_arg_type = None
            curr_arg_tokens = []

            for idx, w_idx in enumerate(xlmr_word_ids):
                if w_idx is None or w_idx >= len(marked_words):
                    continue
                if marked_words[w_idx] in ["<tg>", "</tg>"]:
                    continue

                pred_id = int(arg_preds[idx])
                label = self.argument_id2label.get(pred_id, "O")
                clean_label = label.replace("B-", "").replace("I-", "")

                if label.startswith("B-") or (
                    label.upper() != "O" and clean_label != curr_arg_type
                ):
                    if curr_arg_type and curr_arg_tokens:
                        arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")
                    curr_arg_type = clean_label
                    curr_arg_tokens = [marked_words[w_idx]]
                elif label.startswith("I-") and curr_arg_type == clean_label:
                    curr_arg_tokens.append(marked_words[w_idx])
                else:
                    if curr_arg_type and curr_arg_tokens:
                        arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")
                        curr_arg_type = None
                        curr_arg_tokens = []

            if curr_arg_type and curr_arg_tokens:
                arguments[curr_arg_type] = " ".join(curr_arg_tokens).replace("_", " ")

            for k in list(arguments.keys()):
                if k.lower() in ["time", "date", "thời_gian", "dat"]:
                    raw_time = arguments[k]
                    arguments[f"{k}_Chuẩn_Hóa"] = normalize_vietnamese_time(raw_time, anchor_date)

            results.append({
                "Trigger": tg["text"].replace("_", " "),
                "Loại Sự Kiện": event_type,
                "Các Tham Thể Trích Xuất": arguments,
            })

        return results


BKEEEventPyTorchPipeline = BKEEEventPipeline

