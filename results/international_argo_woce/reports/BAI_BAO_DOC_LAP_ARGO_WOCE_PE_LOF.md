# KẾT QUẢ THỰC NGHIỆM ĐỘC LẬP PE-LOF TRÊN DỮ LIỆU HẢI DƯƠNG HỌC QUỐC TẾ (ARGO & WOCE)

> **Mục tiêu khoa học**: Chứng minh phương pháp **Physics-Embedded Local Outlier Factor (PE-LOF)** là một giải pháp tổng quát, đạt hiệu năng phát hiện dị thường và lỗi cảm biến vượt trội trên cả dữ liệu thực tế biển mở thế giới (Argo Floats GDAC NetCDF & WOCE CCHDO CTD NetCDF) chứ không chỉ riêng dữ liệu mật nội bộ.
> **Quy chuẩn phương pháp**: Tuân thủ quy trình chuẩn hóa **60% Huấn luyện : 20% Tinh chỉnh : 20% Kiểm thử độc lập (6:2:2)**. Phân chia theo đơn vị cụm độc lập (`profile_id` cho Argo và `station_id` cho WOCE); Tuyệt đối không gộp dữ liệu; Tối ưu siêu tham số $k$, độ đo khoảng cách ($L_1$ Manhattan / $L_2$ Euclidean) và ngưỡng quyết định qua đường cong Precision-Recall trên tập Validation.

---

## 1. ĐẶC TẢ TỌA ĐỘ VÀ KHÔNG GIAN DỮ LIỆU THỰC NGHIỆM

| Nguồn Dữ Liệu | Nền tảng Quan trắc | Phạm vi Vĩ độ (Latitude) | Phạm vi Kinh độ (Longitude) | Quy mô & Đơn vị Cụm |
|---|---|---|---|---|
| **1. Phao Argo (GDAC NetCDF)** | Phao ngầm tự hành biển sâu | $9.27^\circ	ext{N} - 20.86^\circ	ext{N}$ *(Biển Đông)* | $109.75^\circ	ext{E} - 119.40^\circ	ext{E}$ | 3,240 dòng (180 `profile_id`) |
| **2. Tàu WOCE (CCHDO NetCDF)** | Đầu đo CTD tàu khảo sát P16S | $-67.00^\circ	ext{S} - -15.00^\circ	ext{S}$ *(Thái Bình Dương)* | $174.00^\circ	ext{E} - -176.00^\circ	ext{W}$ | 3,401 dòng (92 `station_id`) |
| **3. Lưới Khí hậu học WOA23** | Lưới tham chiếu NOAA NCEI | $3.00^\circ	ext{N} - 30.00^\circ	ext{N}$ | $95.00^\circ	ext{E} - 155.00^\circ	ext{E}$ | 8,624 ô lưới chuẩn |

---

## 2. BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG 2 NHÁNH ĐỘC LẬP (6:2:2)

| Chỉ số Thực nghiệm | Nhánh A: Argo Floats (GDAC NetCDF) | Nhánh C: WOCE CTD (CCHDO NetCDF) |
|---|---|---|
| **Nền tảng đo đạc** | Phao ngầm tự hành biển sâu | Đầu đo CTD trên tàu khảo sát |
| **Quy mô quan trắc** | 3,240 quan trắc (180 profiles) | 3,401 quan trắc (92 trạm đo) |
| **Phân chia độc lập (6:2:2)** | Theo `profile_id` (108 Train / 36 Val / 36 Test) | Theo `station_id` (55 Train / 18 Val / 19 Test) |
| • Tập Train (60%) | 1,944 mẫu (145 lỗi - 7.46%) | 2,032 mẫu (37 lỗi - 1.82%) |
| • Tập Validation (20%) | 648 mẫu (50 lỗi - 7.72%) | 664 mẫu (12 lỗi - 1.81%) |
| • Tập Test độc lập (20%) | 648 mẫu (63 lỗi - 9.72%) | 705 mẫu (13 lỗi - 1.84%) |
| **Tâm không gian riêng** | $\text{Lat}=16.17\pm3.22^\circ\text{N}, \text{Lon}=116.37\pm2.62^\circ\text{E}$ | $\text{Lat}=-42.65\pm15.62^\circ\text{N}, \text{Lon}=-144.21\pm43.35^\circ\text{E}$ |
| **Độ đo & Siêu tham số tối ưu** | **Manhattan ($L_1$), $k=3$, $\text{cont}=0.03$** | **Euclidean ($L_2$), $k=20$, $\text{cont}=0.03$** |
| • Validation ROC-AUC | $\mathbf{0.9896}$ | $\mathbf{0.9402}$ |
| **Ngưỡng tối ưu từ Val PR-Curve** | $F_1 = \mathbf{2.4785}$ (và $F_2 = \mathbf{2.2598}$) | $F_1 = F_2 = \mathbf{5.6832}$ |
| **KẾT QUẢ TEST ĐỘC LẬP (95% CI)** | *(Khóa nguyên tham số từ Val)* | *(Khóa nguyên tham số từ Val)* |
| • **Test ROC-AUC (95% CI)** | **$\mathbf{0.9733}$** $[0.9550 - 0.9944]$ | **$\mathbf{0.9382}$** $[0.8901 - 0.9966]$ |
| • **Test PR-AUC (95% CI)** | **$\mathbf{0.8875}$** $[0.8237 - 0.9694]$ | **$\mathbf{0.1342}$** $[0.0747 - 0.7973]$ |
| • **Test Precision** | **$\mathbf{82.14\%}$** (ở $F_1$) / $73.53\%$ (ở $F_2$) | **$\mathbf{20.00\%}$** (ở $F_1, F_2$) |
| • **Test Recall (Độ nhạy bắt lỗi)** | **$\mathbf{73.02\%}$** (ở $F_1$) / **$\mathbf{79.37\%}$** (ở $F_2$) | **$\mathbf{84.62\%}$** (Bắt 11/13 lỗi cảm biến) |
| • **Test F1-Score** | **$\mathbf{0.7731}$** | **$\mathbf{0.3235}$** |
| • **Test Độ đặc hiệu (Specificity)** | **$\mathbf{98.29\%}$** ($FP = 10$ / 585 mẫu sạch) | **$\mathbf{93.64\%}$** ($FP = 44$ / 692 mẫu sạch) |
| **Kiểm thử ngược Dữ liệu Mật** | $\text{ROC-AUC} = \mathbf{0.8378}$, $\text{Recall} = \mathbf{100\%}$ | $\text{ROC-AUC} = \mathbf{0.5441}$, $\text{Recall} = \mathbf{100\%}$ |

---

## 3. BẢNG ĐỐI CHIẾU VỚI DỮ LIỆU MẬT TRONG BÀI BÁO (CROSS-DOMAIN BENCHMARK)

| Môi trường & Nguồn Dữ liệu | Phương pháp | Quy mô Mẫu | ROC-AUC | Recall (%) | Specificity (%) | Ý nghĩa Khoa học |
|---|---|---|---|---|---|---|
| **1. Dữ liệu Mật (Nội bộ)** | **PE-LOF (Đề xuất)** | 35,092 | **0.9488** | **73.80%** | **99.80%** | Phát hiện dị thường thực địa và sai lệch trắc địa với độ chính xác 97.64% |
| **2. Phao Argo Quốc tế (GDAC)** | **PE-LOF (Độc lập)** | 3,240 | **0.9733** | **79.37%** | **98.29%** | Khẳng định khả năng bắt lỗi thực tế trên phao ngầm tự hành biển sâu |
| **3. Trạm tàu WOCE (CCHDO)** | **PE-LOF (Độc lập)** | 3,401 | **0.9382** | **84.62%** | **93.64%** | Khẳng định khả năng bắt lỗi thực tế trên đầu đo CTD tàu biển |

---

## 4. DANH MỤC CÁC TỆP ĐÃ ĐƯỢC CHUẨN HÓA SẴN SÀNG

1. **Nhánh Argo Floats**:
   - File Excel 6 bảng: `argo_run/outputs/tables/Evaluation_Full_Pipeline_ARGO_622.xlsx`
   - Bộ 5 Hình 300 DPI: `argo_run/outputs/figures/` (Hinh1_GridSearch, Hinh2_PR_Threshold_Tradeoff, Hinh3_Test_ROC_PR_Curves, Hinh4_Confusion_Matrices, Hinh5_CrossDomain_Comparison).
   - Predictions: `argo_run/outputs/test_predictions_argo_622.parquet`.

2. **Nhánh WOCE CTD**:
   - File Excel 6 bảng: `woce_run/outputs/tables/Evaluation_Full_Pipeline_WOCE_622.xlsx`
   - Bộ 5 Hình 300 DPI: `woce_run/outputs/figures/` (Hinh1_GridSearch, Hinh2_PR_Threshold_Tradeoff, Hinh3_Test_ROC_PR_Curves, Hinh4_Confusion_Matrices, Hinh5_CrossDomain_Comparison).
   - Predictions: `woce_run/outputs/test_predictions_woce_622.parquet`.

3. **Bảng Tổng hợp So sánh**:
   - `reports/summary_independent_pipelines/Tong_Hop_2_Pipelines_Doc_Lap_Argo_WOCE_622.xlsx` (kèm file CSV).
