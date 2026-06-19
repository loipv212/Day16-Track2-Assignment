# LAB 16 — Báo cáo phương án CPU (LightGBM) — Phần 7

## Lý do dùng CPU thay GPU
Tài khoản AWS không được duyệt quota GPU (dòng G/VT mặc định = 0 vCPU, yêu cầu tăng bị từ chối/chưa duyệt). Theo Phần 7 của lab, chuyển sang triển khai bài toán ML thực tế (LightGBM — phát hiện gian lận thẻ tín dụng) trên instance CPU.

## Hạ tầng
- Terraform IaC: VPC private + NAT Gateway + ALB + Bastion + node (giữ nguyên kiến trúc).
- Node: **`r5.xlarge`** (4 vCPU / 32 GB RAM), Amazon Linux 2023.
  - Lưu ý: dùng `r5.xlarge` thay vì `r5.2xlarge` (đề xuất trong README) vì tài khoản giới hạn **Standard On-Demand = 8 vCPU**; `r5.2xlarge` (8 vCPU) + Bastion (2 vCPU) vượt hạn mức.

## Kết quả benchmark (Credit Card Fraud, 284,807 dòng, 0.173% gian lận)
| Metric | Kết quả |
|---|---|
| Thời gian load data | 1.68 s |
| Thời gian training | 1.79 s |
| Best iteration | 1 |
| AUC-ROC | 0.9316 |
| Accuracy | 0.9989 |
| F1-Score | 0.7249 |
| Precision | 0.6336 |
| Recall | 0.8469 |
| Inference latency (1 dòng) | 0.386 ms |
| Throughput (1000 dòng) | ~1,691,934 dòng/s |

## Nhận xét
- **Training rất nhanh (~1.8s)** trên 4 vCPU nhờ LightGBM tối ưu cho CPU — với dataset dạng bảng cỡ vài trăm nghìn dòng, CPU hoàn toàn đủ, không cần GPU. Đây là bài học chọn hạ tầng đúng workload: GPU chỉ đáng tiền cho deep learning / LLM, không cho gradient boosting trên dữ liệu bảng.
- **Best iteration = 1** là hợp lệ: tập validation chỉ có ~79 ca gian lận nên 1 cây đã đạt AUC đỉnh; early-stopping (kiên nhẫn 100 vòng) xác nhận. Mô hình vẫn đạt AUC 0.93, Recall 0.85 — bắt được phần lớn giao dịch gian lận.
- **Inference cực nhanh**: 0.39 ms/dòng, ~1.7 triệu dòng/giây — thừa sức phục vụ real-time.
- So sánh chi phí: `r5.xlarge` (~$0.25/giờ) rẻ hơn nhiều so với `g4dn.xlarge` GPU (~$0.53/giờ) cho bài toán này.
