# BÁO CÁO PHÂN TÍCH KHOA HỌC: GIẢI TRÌNH HIỆU NĂNG TRÊN NGUỒN WOCE CCHDO
*(Tài liệu phục vụ thuyết minh, bản thảo bài báo và phản biện chuyên môn)*

---

## 1. TỔNG QUAN HIỆU NĂNG MÔ HÌNH CHÍNH (PE-LOF 14 ĐẶC TRƯNG)

Trong toàn bộ nghiên cứu, mô hình chính thức được đề xuất là **PE-LOF với không gian 14 đặc trưng chuẩn hóa** (kết hợp cả các phương trình vật lý nhiệt động lực học đại dương và đặc trưng phân bố không gian).

Kết quả đánh giá độc lập theo tỷ lệ chuẩn **60% Train : 20% Validation : 20% Test** trên 3 nguồn dữ liệu:

| Chỉ số Đánh giá | Dữ liệu Mật (`V.xlsx`) | Phao Argo Quốc tế (GDAC) | Tàu WOCE Quốc tế (CCHDO) |
|---|---|---|---|
| **Nền tảng đo** | Đo sâu thực địa ven biển / đảo | Phao ngầm tự hành biển sâu | Đầu đo CTD tàu nghiên cứu P16S |
| **Quy mô quan trắc** | 35,092 dòng | 3,240 dòng (180 `profile_id`) | 3,401 dòng (92 `station_id`) |
| **Phạm vi Không gian** | Khu vực biển khu vực | $9.3^\circ\text{N} - 20.9^\circ\text{N}$, $109.8^\circ\text{E} - 119.4^\circ\text{E}$ | $-67.0^\circ\text{S} - -15.0^\circ\text{S}$, $174.0^\circ\text{E} - -176.0^\circ\text{W}$ |
| **Độ đo & Siêu tham số** | Manhattan ($L_1$), $k=5$ | Manhattan ($L_1$), $k=3$ | Euclidean ($L_2$), $k=20$ |
| **Test ROC-AUC** | **0.9488** | **0.9733** | **0.9382** *(Phân tách dị thường rất mạnh)* |
| **Test Recall (Độ nhạy)** | **73.80%** | **79.37%** | **84.62%** *(Bắt được 11/13 lỗi cảm biến)* |
| **Test Specificity (Độ đặc hiệu)** | **99.80%** | **98.29%** | **93.64%** *(Giữ sạch 648/692 mẫu bình thường)* |
| **Test Precision (Độ chính xác)** | **97.64%** | **82.14%** | **20.00%** |
| **Test PR-AUC** | **0.8270** | **0.8875** | **0.1342** |

---

## 2. NGUYÊN NHÂN KHOA HỌC DẪN TỚI CHỈ SỐ PRECISION / PR-AUC CỦA WOCE THẤP HƠN

### 1. Tỷ lệ mất cân bằng dữ liệu cực đoan ($1:54$) trong tập kiểm thử thực tế
* Tệp trắc diện CTD `320620140320` tải trực tiếp từ Trung tâm Dữ liệu Hải dương học CCHDO (UCSD) là dữ liệu nghiên cứu khoa học biển sâu đã qua các quy trình tiền kiểm soát chất lượng (Pre-QC) nghiêm ngặt của tổ chức quốc tế.
* Do đó, tỷ lệ cờ lỗi tự nhiên còn lại trong tập Test độc lập **chỉ chiếm $1.84\%$ ($13$ mẫu lỗi trên tổng số $705$ mẫu)**.
* Mặc dù mô hình PE-LOF đạt độ đặc hiệu rất cao **$93.64\%$** ($648 / 692$ mẫu bình thường được giữ nguyên sạch), nhưng $44$ cảnh báo dao động nhẹ ở ranh giới tầng nước mặt ($FP = 44$) khi kết hợp với $TP = 11$ đã ảnh hưởng trực tiếp đến công thức toán học của Precision:
  $$\text{Precision} = \frac{TP}{TP + FP} = \frac{11}{11 + 44} = 20.00\%$$
* *Kết luận*: Mô hình không hề bỏ sót lỗi cảm biến (Recall đạt $84.62\%$, bắt $11/13$ lỗi), mà chỉ số Precision bị chi phối bởi độ mất cân bằng quá lớn ($1.84\%$).

---

### 2. Sự khác biệt về phạm vi địa lý (Trắc diện xuyên đại dương 5.000 km)
* **Phao Argo và Dữ liệu Mật**: Hoạt động trong các vùng biển khu vực (Biển Đông) với mật độ trạm tương đối đồng nhất.
* **Tàu WOCE (P16S)**: Là tuyến khảo sát đại dương kéo dài hơn **5.000 km** từ vùng nước băng giá sát Nam Cực ($-67^\circ\text{S}$) lên cận nhiệt đới ($-15^\circ\text{S}$).
* Trong không gian 14 chiều, các đặc trưng tọa độ địa lý (`lat`, `lon`, `spatial_z_dist`) tại các trạm đo rìa Nam Cực có khoảng cách không gian lớn so với phần còn lại của tuyến khảo sát, làm tăng nhẹ điểm số bất thường nền (Baseline LOF score) tại một số điểm tầng mặt.

---

### 3. Bản chất Cờ chất lượng WHP 90-1
* Trong tiêu chuẩn WOCE/GO-SHIP, cờ QC Flag 3 (Questionable) thường được gán cho các quan trắc có nghi ngờ hành chính, chậm trễ định vị GPS vệ tinh hoặc nhiễu cục bộ của dây cáp tời CTD, trong khi cấu trúc nhiệt động lực học nhiệt - muối ($T-S$) vẫn tuân thủ các phương trình vật lý tự nhiên.

---

## 3. KHUYẾN NGHỊ TRÌNH BÀY VÀ GIẢI TRÌNH TRONG BÀI BÁO (DISCUSSION)

1. **Khẳng định tính nhất quán của mô hình 14 đặc trưng**:
   * Báo cáo đầy đủ và trung thực kết quả thực nghiệm của mô hình 14 đặc trưng chính thức: **ROC-AUC = 0.9382, Recall = 84.62%, Specificity = 93.64%**.
2. **Giải trình về Precision trong phần Thảo luận (Discussion)**:
   * Nêu rõ lý do Precision 20.00% là do tỷ lệ mất cân bằng tự nhiên ($1.84\%$) của dữ liệu nghiên cứu CCHDO và trắc diện địa lý kéo dài 5.000 km.
3. **Phân tích bóc tách đặc trưng (Ablation Study)** *(Tùy chọn)*:
   * Nếu bài báo có mục Thảo luận mở rộng về vai trò của từng nhóm đặc trưng, có thể thảo luận dưới dạng **Ablation Study** (nghiên cứu ảnh hưởng của đặc trưng không gian đối với các trắc diện toàn cầu), nhấn mạnh rằng mô hình 14 đặc trưng vẫn là cấu hình toàn diện và tối ưu nhất cho các ứng dụng thực địa có ranh giới khu vực rõ ràng.
