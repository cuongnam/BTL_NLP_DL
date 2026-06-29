"""
export_onnx.py

Chuyển đổi và lượng tử hóa (Quantization INT8) 3 mô hình Deep Learning 
(PhoBERT-Trigger, PhoBERT-Event, XLM-R-Argument) sang định dạng ONNX 
để tối ưu hóa bộ nhớ và tăng tốc độ Pipeline Giai đoạn 5.
"""

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Cấu hình đường dẫn hệ thống
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
ONNX_OUTPUT_DIR = ROOT_DIR / "models" / "onnx_optimized"
ONNX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def export_and_quantize_model(model_folder_name, output_name, is_xlmr=False, num_labels=3):
    """
    Hàm đóng gói: Tải mô hình PyTorch -> Xuất sang ONNX FP32 -> Lượng tử hóa sang ONNX INT8.
    """
    model_path = MODELS_DIR / model_folder_name
    if not model_path.exists():
        print(f"[CẢNH BÁO] Không tìm thấy thư mục mô hình tại: {model_path}. Bỏ qua...")
        return

    print(f"\n--- BẮT ĐẦU XỬ LÝ MÔ HÌNH: {output_name} ---")
    
    # 1. Tải Tokenizer và Mô hình PyTorch gốc
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()  # Chuyển sang chế độ dự đoán (Inference)

    # 2. Tạo dữ liệu giả lập (Dummy Input) để định hình khung đồ thị tính toán
    text_dummy = "Học_viên Bách_Khoa làm đồ_án xử_lý ngôn_ngữ tự_nhiên."
    inputs = tokenizer(text_dummy, return_tensors="pt", max_length=256, padding="max_length", truncation=True)
    
    dummy_input_ids = inputs["input_ids"]
    dummy_attention_mask = inputs["attention_mask"]

    # Thiết lập đường dẫn file xuất ra
    onnx_fp32_path = ONNX_OUTPUT_DIR / f"{output_name}_fp32.onnx"
    onnx_int8_path = ONNX_OUTPUT_DIR / f"{output_name}_int8.onnx"

    # 3. Tiến hành xuất sang ONNX định dạng gốc (FP32)
    print(f"[1/3] Đang xuất sang cấu trúc đồ thị ONNX FP32...")
    
    # Cấu hình các trục động (Dynamic axes) để xử lý độ dài câu linh hoạt khi chạy thực tế
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'}
    }

    torch.onnx.export(
        model=model,
        args=(dummy_input_ids, dummy_attention_mask),
        f=str(onnx_fp32_path),
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
        opset_version=14,  # Phiên bản toán tử khuyên dùng cho Transformer
        do_constant_folding=True
    )
    print(f"-> Đã lưu file gốc FP32 tại: {onnx_fp32_path}")

    # Kiểm tra tính hợp lệ của file ONNX vừa tạo
    onnx_model = onnx.load(str(onnx_fp32_path))
    onnx.checker.check_model(onnx_model)

    # 4. Áp dụng kỹ thuật Lượng tử hóa động (Dynamic Quantization INT8)
    print(f"[2/3] Đang tiến hành ép nén trọng số về số nguyên INT8...")
    quantize_dynamic(
        model_input=str(onnx_fp32_path),
        model_output=str(onnx_int8_path),
        weight_type=QuantType.QUInt8  # Ép trọng số tuyến tính về dạng Unsigned INT8
    )
    print(f"-> Đã tối ưu nén xong file INT8 tại: {onnx_int8_path}")

    # 5. Xóa file FP32 trung gian để dọn dẹp bộ nhớ lưu trữ
    print(f"[3/3] Dọn dẹp dán nhãn file lưu trữ...")
    if onnx_fp32_path.exists():
        onnx_fp32_path.unlink()
        
    print(f"✔️ Hoàn thành tối ưu mô hình {output_name} thành công!")

def main():
    print("=== CHƯƠNG TRÌNH ĐÓNG GÓI NÉN MÔ HÌNH SANG ONNX CHUẨN CÔNG NGHIỆP ===")
    
    # 1. Đóng gói mô hình Trigger Detection (PhoBERT)
    export_and_quantize_model(
        model_folder_name="best_phobert_trigger", 
        output_name="phobert_trigger"
    )

    # 2. Đóng gói mô hình Event Type Classification (PhoBERT)
    # Lưu ý: Mô hình này có cấu hình đầu ra 34 nhãn do có nhãn "O" (ID 33)
    export_and_quantize_model(
        model_folder_name="best_phobert_event_type", 
        output_name="phobert_event_type"
    )

    # 3. Đóng gói mô hình Argument Extraction (XLM-RoBERTa)
    # Đã bao gồm ma trận nhúng mở rộng cho cặp thẻ <tg>...</tg>
    export_and_quantize_model(
        model_folder_name="best_xlmr_argument", 
        output_name="xlmr_argument"
    )

    print("\n[THÀNH CÔNG] Tất cả mô hình nén định dạng đuôi '_int8.onnx' đã sẵn sàng tại thư mục: models/onnx_optimized/")

if __name__ == "__main__":
    main()