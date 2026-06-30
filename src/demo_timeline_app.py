"""
demo_timeline_app.py

Giao diện Web tương tác (Streamlit) cho phép nhập văn bản tin tức,
chạy Pipeline ONNX INT8 và trực quan hóa chuỗi sự kiện doanh nghiệp trên trục thời gian Plotly.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
from datetime import datetime
from timeline_generator import BKEEEventPipeline

# Cấu hình trang Streamlit
st.set_page_config(page_title="BKEE - Enterprise Event Timeline", layout="wide", page_icon="📈")

@st.cache_resource
def load_pipeline():
    # Cache lại đối tượng pipeline để tránh việc nạp lại mô hình ONNX mỗi lần bấm nút
    return BKEEEventPipeline()

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Không thể nạp hệ thống Pipeline: {e}")
    st.stop()

st.title("📈 Hệ thống Trích xuất & Trực quan hóa Dòng thời gian Sự kiện Doanh nghiệp")
st.markdown("---")

# Bố cục giao diện chia làm 2 cột: Cột trái nhập liệu - Cột phải hiển thị kết quả
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📝 Dữ liệu Tin tức Đầu vào")
    
    # Textarea nhận văn bản tin tức từ người dùng
    default_text = (
        "Ngày 15/06/2026 , Hội_đồng_quản_trị tập_đoàn VinGroup đã bổ_nhiệm ông Nguyễn_Văn_B làm Tổng_giám_đốc mới .\n"
        "Công_ty FPT vừa đầu_tư 50 triệu USD vào một doanh_nghiệp công_nghệ tại Mỹ ."
    )
    
    raw_input = st.text_area(
        "Nhập các câu tin tức (Mỗi câu một dòng, từ ghép có thể nối bằng dấu _ ):",
        value=default_text,
        height=200
    )
    
    anchor_date = st.date_input("Ngày neo ngữ cảnh (Anchor Date):", datetime(2026, 6, 30))
    anchor_date_str = anchor_date.strftime("%Y-%m-%d")
    
    run_btn = st.button("🚀 Trích xuất Sự kiện & Sinh Timeline", type="primary")

with col2:
    st.subheader("📊 Trục Thời gian Sự kiện (Interactive Timeline)")
    
    if run_btn and raw_input.strip():
        lines = [line.strip() for line in raw_input.split("\n") if line.strip()]
        
        all_events = []
        with st.spinner("Hệ thống ONNX INT8 đang xử lý siêu tốc..."):
            for line in lines:
                extracted = pipeline.extract_events(line, anchor_date=anchor_date_str)
                all_events.extend(extracted)
        
        if not all_events:
            st.warning("⚠️ Không tìm thấy hoặc không trích xuất được sự kiện nào từ văn bản trên.")
        else:
            # Chuyển đổi dữ liệu sang DataFrame để vẽ đồ thị
            plot_data = []
            for ev in all_events:
                args = ev["Các Tham Thể Trích Xuất"]
                
                # Tìm mốc thời gian chuẩn hóa, nếu lỗi/thiếu thì lấy ngày anchor làm mặc định
                time_val = args.get("Time_Chuẩn_Hóa", args.get("Date_Chuẩn_Hóa", anchor_date_str))
                if time_val == "Ngày" or not time_val:
                    time_val = anchor_date_str
                    
                plot_data.append({
                    "Mốc Thời Gian": time_val,
                    "Sự Kiện": ev["Loại Sự Kiện"],
                    "Từ Kích Hoạt": ev["Trigger"],
                    "Chi Tiết": json.dumps(args, ensure_ascii=False)
                })
            
            df = pd.DataFrame(plot_data)
            
            # Sắp xếp theo thứ tự thời gian tăng dần
            df = df.sort_values(by="Mốc Thời Gian")
            
            # Vẽ đồ thị dòng thời gian bằng Plotly Scatter
            fig = px.scatter(
                df, 
                x="Mốc Thời Gian", 
                y="Sự Kiện", 
                text="Từ Kích Hoạt",
                color="Sự Kiện",
                hover_data=["Chi Tiết"],
                title="Chuỗi Tiến Trình Sự Kiện Doanh Nghiệp Trích Xuất Được",
                size_max=40
            )
            fig.update_traces(textposition='top center', marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')))
            fig.update_layout(xaxis_type='category', height=350)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Hiển thị bảng dữ liệu chi tiết ở dưới đồ thị
            st.subheader("📋 Bảng Dữ liệu Cấu trúc")
            st.dataframe(df, use_container_width=True)