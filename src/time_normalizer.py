"""
time_normalizer.py

Yêu cầu nâng cao Giai đoạn 4: Chuẩn hóa thực thể thời gian tương đối
từ văn bản tiếng Việt về định dạng chuẩn YYYY-MM-DD dựa trên ngày xuất bản báo.
"""

import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def normalize_vietnamese_time(time_text: str, anchor_date_str: str = None) -> str:
    """
    Chuẩn hóa các cụm từ thời gian tiếng Việt về định dạng YYYY-MM-DD.
    
    Parameters
    ----------
    time_text : str
        Cụm từ thời gian trích xuất được (ví dụ: "hôm qua", "ngày 25/12/2023")
    anchor_date_str : str, optional
        Ngày xuất bản của bài báo dạng "YYYY-MM-DD". Nếu không có, mặc định là ngày hiện tại.
        
    Returns
    -------
    str
        Chuỗi ngày đã chuẩn hóa dạng YYYY-MM-DD hoặc giữ nguyên nếu không xử lý được.
    """
    # 1. Xác định ngày neo (Anchor Date - Ngày xuất bản báo)
    if anchor_date_str:
        try:
            anchor_date = datetime.strptime(anchor_date_str, "%Y-%m-%d")
        except ValueError:
            anchor_date = datetime.now()
    else:
        anchor_date = datetime.now()

    # Làm sạch văn bản chữ thường
    text = time_text.lower().strip()
    
    # ======= TRƯỜNG HỢP 1: THỜI GIAN TƯƠNG ĐỐI TRONG NGÀY/TUẦN =======
    if text in ["hôm nay", "sáng nay", "chiều nay", "tối nay", "nay"]:
        return anchor_date.strftime("%Y-%m-%d")
        
    if text in ["hôm qua", "sáng qua", "chiều qua", "tối qua"]:
        target_date = anchor_date - timedelta(days=1)
        return target_date.strftime("%Y-%m-%d")
        
    if text in ["hôm kia", "ngày kia"]:
        target_date = anchor_date - timedelta(days=2)
        return target_date.strftime("%Y-%m-%d")
        
    if text in ["ngày mai", "sáng mai"]:
        target_date = anchor_date + timedelta(days=1)
        return target_date.strftime("%Y-%m-%d")

    if text in ["tuần trước", "vài tuần trước"]:
        target_date = anchor_date - timedelta(weeks=1)
        return target_date.strftime("%Y-%m-%d")
        
    if text in ["tháng trước"]:
        target_date = anchor_date - relativedelta(months=1)
        return target_date.strftime("%Y-%m-%d")

    # ======= TRƯỜNG HỢP 2: THỨ TRONG TUẦN (Ví dụ: "thứ ba vừa qua", "thứ 5 tuần trước") =======
    weekday_map = {
        "hai": 0, "thứ hai": 0, "thứ 2": 0,
        "ba": 1, "thứ ba": 1, "thứ 3": 1,
        "tư": 2, "thứ tư": 2, "thứ 4": 2,
        "năm": 3, "thứ năm": 3, "thứ 5": 3,
        "sáu": 4, "thứ sáu": 4, "thứ 6": 4,
        "bảy": 5, "thứ bảy": 5, "thứ 7": 5,
        "chủ nhật": 6, "cn": 6
    }
    
    for day_name, day_code in weekday_map.items():
        if day_name in text:
            # Tính khoảng cách thứ của anchor_date so với thứ trong text
            current_weekday = anchor_date.weekday()
            days_diff = current_weekday - day_code
            
            if "tuần trước" in text:
                target_date = anchor_date - timedelta(days=days_diff + 7)
            else:
                # Nếu chỉ nói "thứ ba vừa qua" hoặc "thứ ba" -> lùi về thứ ba gần nhất
                if days_diff <= 0:
                    days_diff += 7
                target_date = anchor_date - timedelta(days=days_diff)
            return target_date.strftime("%Y-%m-%d")

    # ======= TRƯỜNG HỢP 3: ĐỊNH DẠNG SỐ TUYỆT ĐỐI (Ví dụ: "25/12/2023", "ngày 15 tháng 8") =======
    # Khớp định dạng: DD/MM/YYYY hoặc DD-MM-YYYY
    match_full_date = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', text)
    if match_full_date:
        d, m, y = match_full_date.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

    # Khớp dạng chữ: "ngày 15 tháng 8 năm 2022" hoặc "ngày 15/8"
    match_text_date = re.search(r'ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})(?:\s+năm\s+(\d{4}))?', text)
    if match_text_date:
        d, m, y = match_text_date.groups()
        year = int(y) if y else anchor_date.year  # Nếu không nói năm, lấy năm của bài báo
        return f"{year:04d}-{int(m):02d}-{int(d):02d}"

    # Nếu không khớp bất kỳ luật nào, giữ nguyên text gốc để bảo toàn dữ liệu
    return time_text

# Đoạn code kiểm thử chạy thử nghiệm nhanh (Sanity Check)
if __name__ == "__main__":
    # Giả định bài báo xuất bản ngày 25 tháng 12 năm 2023 (Thứ Hai)
    paper_date = "2023-12-25"
    print(f"Ngày xuất bản báo làm gốc: {paper_date}\n" + "-"*40)
    
    test_cases = [
        "Hôm qua", 
        "Sáng nay", 
        "thứ Ba vừa qua", 
        "thứ 5 tuần trước", 
        "ngày 15 tháng 8", 
        "30/04/1975"
    ]
    
    for case in test_cases:
        norm = normalize_vietnamese_time(case, anchor_date_str=paper_date)
        print(f"Gốc: '{case}' ---> Chuẩn hóa: {norm}")