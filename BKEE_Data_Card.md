# Data Card -- BKEE: Vietnamese Event Extraction Dataset

## 1. Dataset Summary

**Dataset Name:** BKEE (BK Event Extraction)

**Version:** Phiên bản được công bố trong bài báo LREC-COLING 2024

**Language:** Tiếng Việt

**Task:** Event Extraction (EE)

**Publication:** LREC-COLING 2024

**Dataset Description**

BKEE là bộ dữ liệu gán nhãn cho bài toán Event Extraction tiếng Việt. Bộ
dữ liệu hỗ trợ đầy đủ ba tác vụ chính của Event Extraction:

-   Entity Mention Detection (EMD)
-   Event Detection (ED)
-   Event Argument Extraction (EAE)

BKEE được xây dựng nhằm khắc phục sự thiếu hụt tài nguyên chuẩn cho
nghiên cứu Event Extraction trong tiếng Việt.

## 2. Motivation

### Why was this dataset created?

-   Cung cấp bộ dữ liệu chuẩn cho Event Extraction tiếng Việt.
-   Hỗ trợ nghiên cứu và phát triển các mô hình NLP.
-   Thiết lập benchmark cho các nghiên cứu sau này.
-   Thu hẹp khoảng cách tài nguyên giữa tiếng Việt và các ngôn ngữ giàu
    tài nguyên.

## 3. Dataset Composition

  |Thuộc tính            |Giá trị
  |--------------------- |---------------------
  |Number of documents   |1,066
  |Language              |Vietnamese
  |Domains               |11 lĩnh vực tin tức
  |Entity types          |12
  |Event types           |8
  |Event sub-types       |33
  |Argument roles        |28

## 4. Data Source

Nguồn dữ liệu được thu thập từ các bài báo và bản tin tiếng Việt thuộc
nhiều lĩnh vực khác nhau nhằm đảm bảo tính đa dạng của các loại sự kiện.

## 5. Annotation Process

Việc gán nhãn được thực hiện hoàn toàn bằng thủ công.

-   Entity Mention
-   Event Trigger
-   Event Arguments

## 6. Intended Use

-   Event Extraction
-   Event Detection
-   Trigger Detection
-   Argument Extraction
-   Information Extraction
-   Knowledge Graph Construction
-   Question Answering

## 7. Limitations

-   Chỉ hỗ trợ tiếng Việt.
-   Chủ yếu là dữ liệu tin tức.
-   Quy mô nhỏ hơn nhiều benchmark quốc tế.

## 8. Citation

Nguyen et al. (2024). *BKEE: Pioneering Event Extraction in the
Vietnamese Language*. LREC-COLING 2024.
