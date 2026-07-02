# Data Card: Bộ dữ liệu BKEE – Bách Khoa Event Extraction

## 1.Dataset Summary

**BKEE (Bách Khoa Event Extraction Dataset)** là bộ dữ liệu chuẩn dành cho bài toán **Trích xuất sự kiện (Event Extraction)** trong tiếng Việt, được công bố tại hội nghị **LREC-COLING 2024**. Bộ dữ liệu được xây dựng từ các bài báo điện tử tiếng Việt và được gán nhãn thủ công theo các thành phần của sự kiện.

Trong đề tài này, bộ dữ liệu BKEE được sử dụng để xây dựng hệ thống trích xuất sự kiện và dựng Timeline, bao gồm các tác vụ:

- Nhận diện Trigger (Trigger Detection)
- Phân loại loại sự kiện (Event Type Classification)
- Trích xuất thành phần sự kiện (Argument Extraction)
- Trích xuất thông tin thời gian
- Xây dựng Timeline sự kiện

---

## 2.Motivation and Intended Use

BKEE được xây dựng nhằm cung cấp một bộ dữ liệu chuẩn phục vụ nghiên cứu các bài toán Trích xuất sự kiện trong tiếng Việt.

Bộ dữ liệu phù hợp cho các nhiệm vụ:

- Trích xuất sự kiện (Event Extraction)
- Nhận diện Trigger
- Phân loại loại sự kiện
- Trích xuất Argument
- Trích xuất thông tin
- Xây dựng Timeline
- Xây dựng đồ thị tri thức (Knowledge Graph)
- Nghiên cứu và đánh giá các mô hình NLP tiếng Việt

---

## 3.Dataset Sources

**Tên bộ dữ liệu**

BKEE – Bách Khoa Event Extraction Dataset

**Nguồn công bố**

LREC-COLING 2024

**Tác giả**

Nguyễn Thị Nhung và cộng sự.

**Kho mã nguồn**

https://github.com/nhungnt7/BKEE

**Ngôn ngữ**

Tiếng Việt

**Nguồn văn bản**

Các bài báo điện tử tiếng Việt thuộc nhiều lĩnh vực như chính trị, kinh tế, xã hội, y tế và giáo dục.

---

## 4.Dataset Composition

Mỗi mẫu dữ liệu bao gồm:

- Mã tài liệu (Document ID)
- Câu văn (Sentence)
- Danh sách token
- Nhãn Trigger
- Loại sự kiện (Event Type)
- Thành phần sự kiện (Arguments)
- Thực thể (Named Entity)
- Thông tin thời gian (Temporal Information)

Ví dụ:

**Câu văn**

```
Bộ Y tế công bố thêm 5 ca mắc mới.
```

**Kết quả gán nhãn**

```
Trigger:
công bố

Event Type:
Announcement

Arguments

Agent:
Bộ Y tế

Patient:
5 ca mắc mới
```

---

## 5.Data Collection Process

Dữ liệu được thu thập từ các bài báo điện tử tiếng Việt công khai.

Quy trình xây dựng dữ liệu gồm các bước:

1. Thu thập bài báo.
2. Làm sạch dữ liệu.
3. Phân tách câu.
4. Tách từ.
5. Gán nhãn thủ công.
6. Kiểm tra và hiệu chỉnh dữ liệu.

---

## 6.Annotation Process

Bộ dữ liệu được gán nhãn thủ công theo schema của BKEE.

Các thành phần được chú thích gồm:

- Trigger
- Event Type
- Argument
- Named Entity
- Temporal Expression

Trigger được gán nhãn theo chuẩn BIO.

Ví dụ:

| Token | Nhãn |
|--------|------|
| Bộ | O |
| Y_tế | O |
| công_bố | B-Trigger |
| thêm | O |
| 5 | O |
| ca_mắc_mới | O |

---

## 7.Preprocessing

Trong đề tài, dữ liệu được tiền xử lý theo pipeline sau:

```
Tài liệu
    ↓
Phân tách câu
    ↓
Tách từ tiếng Việt
    ↓
Tokenization bằng PhoBERT
    ↓
Căn chỉnh nhãn BIO
    ↓
Padding
    ↓
Lưu dưới dạng JSON
```

Các bước tiền xử lý bao gồm:

- Sentence Segmentation
- Vietnamese Word Segmentation
- PhoBERT Tokenization
- Label Alignment
- Padding
- Sinh Attention Mask

Độ dài tối đa của chuỗi đầu vào:

```
256 token
```

---

## 8.Train / Validation / Test Split

Trong quá trình thực nghiệm, bộ dữ liệu được chia theo tỷ lệ:

| Tập dữ liệu | Tỷ lệ |
|-------------|-------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Trong đó:

- **Train** dùng để huấn luyện mô hình.
- **Validation** dùng để lựa chọn mô hình và điều chỉnh siêu tham số.
- **Test** dùng để đánh giá kết quả cuối cùng.

---

## 9.Label Distribution

### Trigger

Trigger được gán nhãn theo chuẩn BIO:

- B-Trigger
- I-Trigger
- O

### Event Type

Bộ dữ liệu bao gồm nhiều loại sự kiện khác nhau như:

- Announcement
- Meeting
- Attack
- Movement
- Arrest
- Medical
- Disaster
- Transportation

### Argument

Các vai trò Argument phổ biến gồm:

- Agent
- Person
- Organization
- Victim
- Target
- Place
- Time
- Instrument

---

## 10.Data Quality Checks

Trong quá trình tiền xử lý và huấn luyện, dữ liệu được kiểm tra theo các tiêu chí:

- Kiểm tra tính hợp lệ của nhãn BIO.
- Kiểm tra sự tương ứng giữa Trigger và Event Type.
- Kiểm tra căn chỉnh giữa token và nhãn.
- Kiểm tra Argument thuộc đúng sự kiện.
- Kiểm tra độ dài chuỗi đầu vào.
- Loại bỏ các mẫu dữ liệu không hợp lệ.

---

## 11.Ethical Considerations

Bộ dữ liệu được xây dựng từ các bài báo công khai trên Internet.

Dataset chỉ phục vụ mục đích:

- Nghiên cứu khoa học.
- Học tập.
- Phát triển các mô hình NLP.

Người sử dụng cần tuân thủ giấy phép của bộ dữ liệu và trích dẫn nguồn khi sử dụng.

---

## 12.Biases and Limitations

Bộ dữ liệu vẫn còn một số hạn chế:

- Quy mô dữ liệu còn tương đối nhỏ.
- Phân bố các loại sự kiện chưa cân bằng.
- Chỉ bao gồm văn bản tiếng Việt.
- Chủ yếu được thu thập từ báo điện tử.
- Một số loại sự kiện có số lượng mẫu rất ít.

Do đó, hiệu năng của mô hình có thể khác nhau giữa các loại sự kiện.

---

## 13.License and Access

Kho dữ liệu:

https://github.com/nhungnt7/BKEE

Bài báo:

https://aclanthology.org/2024.lrec-main.217/

Người sử dụng cần trích dẫn bài báo gốc khi sử dụng bộ dữ liệu trong các nghiên cứu hoặc sản phẩm học thuật.

---

## 14.Recommended Uses

Bộ dữ liệu phù hợp cho các nghiên cứu:

- Trích xuất sự kiện.
- Nhận diện Trigger.
- Phân loại loại sự kiện.
- Trích xuất Argument.
- Xây dựng Timeline.
- Hệ thống hỏi đáp.
- Xây dựng Knowledge Graph.
- Fine-tuning các mô hình Transformer.
- Đánh giá các mô hình NLP tiếng Việt.

---

## 15.Prohibited or Risky Uses

Không nên sử dụng bộ dữ liệu cho các mục đích:

- Xâm phạm quyền riêng tư.
- Giám sát cá nhân.
- Hệ thống ra quyết định pháp lý hoặc y tế mà không có sự kiểm chứng của con người.
- Tạo hoặc lan truyền thông tin sai lệch.
- Các ứng dụng thương mại không tuân thủ giấy phép của bộ dữ liệu.
