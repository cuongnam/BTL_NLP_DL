"""
demo_timeline_app.py

Giao diện Web tương tác (Streamlit) cho phép nhập một câu hoặc một đoạn văn tin tức,
tự động tách câu, chạy qua hệ thống Pipeline ONNX INT8 và hiển thị kết quả trích xuất sự kiện cấu trúc.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime
from timeline_generator import BKEEEventPipeline

# 1. Cấu hình trang Streamlit diện mạo hiện đại
st.set_page_config(page_title="BKEE - Event Extraction Demo", layout="wide", page_icon="📝")

@st.cache_resource
def load_pipeline():
    """Tải và cấu hình bộ Pipeline duy nhất một lần để tiết kiệm RAM"""
    return BKEEEventPipeline()

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"❌ Không thể nạp hệ thống Pipeline mô hình: {e}")
    st.stop()

# --- TIÊU ĐỀ GIAO DIỆN ---
st.title("📝 Hệ thống Trích xuất Sự kiện & Sinh Dòng thời gian Doanh nghiệp")
st.markdown("Ứng dụng sử dụng mô hình mã hóa ngôn ngữ kết hợp luật ngữ nghĩa nhằm bóc tách tự động: *Loại sự kiện, Từ kích hoạt, Doanh nghiệp (Org), Đối tượng tác động (Agent), Số tiền (Money), và Thời gian (Time)*.")
st.markdown("---")

# Bố cục giao diện: Cột trái nhập câu xử lý - Cột phải xem kết quả phân tích trực quan
col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("📥 Nhập văn bản phân tích")
    
    # Cho phép người dùng linh hoạt nhập 1 câu hoặc một chuỗi câu ngắn
    default_sentence = "Ngày 15/6 , Hội_đồng_quản_trị tập_đoàn VinGroup đã bổ_nhiệm ông Nguyễn_Văn_B làm Tổng_giám_đốc mới ."
    
    user_input = st.text_area(
        "Nhập câu tin tức của bạn tại đây (Từ ghép tiếng Việt có thể nối bằng dấu _ để mô hình nhận diện tốt hơn):",
        value=default_sentence,
        height=150,
        placeholder="Ví dụ: Công_ty FPT vừa đầu_tư 50 triệu USD vào doanh_nghiệp công_nghệ tại Mỹ ."
    )
    
    # Cấu hình Ngày neo ngữ cảnh phục vụ chặng chuẩn hóa thời gian máy đọc (Chặng 4)
    anchor_date = st.date_input("Ngày neo ngữ cảnh (Anchor Date phục vụ chuẩn hóa):", datetime(2026, 6, 30))
    anchor_date_str = anchor_date.strftime("%Y-%m-%d")
    
    # Nút bấm kích hoạt tiến trình trích xuất
    submit_btn = st.button("🚀 Trích xuất Sự kiện ngay", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Kết quả Trích xuất Cấu trúc")
    
    if submit_btn and user_input.strip():
        # Hỗ trợ tách văn bản nếu người dùng nhập nhiều câu bằng dấu chấm hoặc xuống dòng
        raw_sentences = [s.strip() for s in user_input.replace("\n", ".").split(".") if s.strip()]
        
        all_extracted_events = []
        
        with st.spinner("Hệ thống ONNX INT8 đang thực hiện suy luận siêu tốc..."):
            for sentence in raw_sentences:
                # Gọi hàm xử lý trung tâm của Pipeline
                extracted_res = pipeline.extract_events(sentence, anchor_date=anchor_date_str)
                all_extracted_events.extend(extracted_res)
                
        if not all_extracted_events:
            st.warning("⚠️ Không tìm thấy từ kích hoạt hoặc mô hình đánh giá không có sự kiện doanh nghiệp nào trong câu trên.")
        else:
            st.success(f"🎉 Tìm thấy {len(all_extracted_events)} sự kiện cấu trúc!")
            
            # Đóng gói kết quả hiển thị dạng bảng và đồ thị
            table_data = []
            for ev in all_extracted_events:
                args = ev["Các Tham Thể Trích Xuất"]
                
                # Sửa lỗi hiển thị mốc thời gian nếu chặng chuẩn hóa trả về chữ "Ngày" chung chung
                time_normalized = args.get("Time_Chuẩn_Hóa", args.get("Date_Chuẩn_Hóa", anchor_date_str))
                if time_normalized == "Ngày" or not time_normalized:
                    time_normalized = anchor_date_str
                
                table_data.append({
                    "Mốc Thời Gian": time_normalized,
                    "Loại Sự Kiện": ev["Loại Sự Kiện"],
                    "Từ Kích Hoạt (Trigger)": ev["Trigger"],
                    "Thông Tin Chi Tiết (Arguments)": json.dumps(args, ensure_ascii=False)
                })
                
            df_events = pd.DataFrame(table_data).sort_values(by="Mốc Thời Gian")
            
            # --- TRỰC QUAN HÓA BẰNG ĐỒ THỊ BONG BÓNG TIMELINE ---
            fig = px.scatter(
                df_events, 
                x="Mốc Thời Gian", 
                y="Loại Sự Kiện", 
                text="Từ Kích Hoạt (Trigger)",
                color="Loại Sự Kiện",
                hover_data=["Thông Tin Chi Tiết (Arguments)"],
                title="Sơ đồ Tiến trình Sự kiện bóc tách được từ câu",
                size_max=35
            )
            fig.update_traces(
                textposition='top center', 
                marker=dict(size=18, line=dict(width=1.5, color='DarkSlateGrey'))
            )
            fig.update_layout(xaxis_type='category', height=300)
            
            # Đổ đồ thị ra giao diện web
            st.plotly_chart(fig, use_container_width=True)
            
            # --- HIỂN THỊ DẠNG BẢNG DỮ LIỆU CẤU TRÚC ---
            st.markdown("#### 📋 Chi tiết các thực thể bóc tách diện JSON")
            st.dataframe(df_events, use_container_width=True)
            
            # Thêm box hiển thị định dạng JSON nguyên bản cho sinh viên dễ nộp báo cáo
            with st.expander("🔍 Xem định dạng JSON thô trả về từ mô hình"):
                st.json(all_extracted_events)