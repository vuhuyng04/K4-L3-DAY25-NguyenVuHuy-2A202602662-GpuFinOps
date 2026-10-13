# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $13,705  
**Projected savings:** $13,428  (**49%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,200 |
| Purchasing (spot/reserved) | $9,792 |
| Right-size (util-lie + MBU) | $1,836 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Best region (cleanest grid): europe-north1

## Baseline vs. Optimized theo $/1M-token

Sản lượng: 7,533,027 token/ngày ≈ 226M token/tháng. Tổng chi giảm từ **$27,133 → $13,705/tháng (−49%)**.

**Chi phí blended (toàn bộ GPU + API) chia cho token phục vụ, bật lần lượt từng lever:**

| Bước | Chi/tháng | $/1M-token | % đóng góp vào tổng tiết kiệm |
|---|---|---|---|
| Baseline | $27,133 | $120.06 | — |
| + Inference (cascade/cache/batch) | $25,933 | $114.75 | 9% |
| + Purchasing (spot/reserved) | $16,141 | $71.42 | 73% |
| + Right-size (util-lie + MBU) | $14,305 | $63.30 | 14% |
| + Kill idle GPUs | $13,705 | $60.64 | 4% |

**Riêng inference API (M2): từng lever giảm $/1M-token bao nhiêu:**

| Bước | Chi/tháng | $/1M-token |
|---|---|---|
| Baseline (all large, no cache, no batch) | $1,466 | $6.488 |
| + Cascade (route_tier) | $344 | $1.523 |
| + Prompt caching (gated) | $321 | $1.419 |
| + Batch API | $266 | $1.178 |

Cascade đóng góp lớn nhất ở inference ($6.49 → $1.52/1M) vì 79% request là câu dễ và model nhỏ rẻ hơn 15× (input) đến 37.5× (output). Cache và batch là chiết khấu nhân lên trên phần còn lại, nên tác động tuyệt đối nhỏ hơn sau khi đã cascade.

_Lưu ý phạm vi: inference + purchasing tính trên hoá đơn baseline; right-size và idle đo trên fleet telemetry (M1). Các lever có thể chồng lấn một phần khi triển khai thật._

## Phân tích: "GPU-Util lie" là gì và tốn bao nhiêu tiền

| GPU | Loại | Workload | GPU-Util | MFU | MBU | Hiệu quả thực | Tiền thuê/tháng | Trả cho FLOPs/BW không nhận được |
|---|---|---|---|---|---|---|---|---|
| gpu-h100-4 | H100 | train | 98.2% | 0.194 | 0.207 | MFU 0.19 vs mục tiêu 0.40 | $1,800 | $927 |
| gpu-a10g-1 | A10G | infer | 96.9% | 0.268 | 0.302 | MBU 0.30 vs mục tiêu 0.60 | $720 | $358 |

**Tổng tiền "trả mà không nhận": $1,285/tháng.**

**Cơ chế.** `GPU-Util` của `nvidia-smi` là tỷ lệ *thời gian* có ít nhất một kernel đang chạy trong cửa sổ lấy mẫu, không phải tỷ lệ SM/Tensor Core thực sự làm việc. Một kernel nhỏ chiếm 1/132 SM, hoặc một kernel đang đứng chờ HBM (memory stall), vẫn được tính là GPU "bận" 100%. Vì vậy `gpu-h100-4` báo 98.2% util nhưng chỉ đạt MFU 0.19 (các H100 cùng cụm đạt 0.40–0.43). Nguyên nhân gốc rễ thường gặp: (1) batch/micro-batch quá nhỏ nên kernel không lấp đầy SM; (2) kernel launch overhead và đồng bộ CPU↔GPU (không dùng CUDA Graphs/fused kernels); (3) dataloader/I-O chậm làm GPU quay vòng chờ; (4) chờ all-reduce/NCCL giữa các node; (5) phép toán memory-bound (layernorm, attention không dùng FlashAttention) chiếm phần lớn thời gian. `gpu-a10g-1` là trường hợp decode memory-bound: util 97% nhưng MBU chỉ ~0.30, nghĩa là băng thông HBM cũng không được dùng hết.

**Ý nghĩa chi phí.** Hoá đơn tính theo giờ GPU, không theo FLOPs. Ở MFU 0.19, chi phí thực trên mỗi FLOP hữu ích cao gấp ~2.1× so với các H100 khoẻ cùng cụm, tức là $/1M-token training cũng đắt gấp chừng đó. Dashboard dựa trên GPU-Util sẽ báo "đã tận dụng hết, cần mua thêm GPU", dẫn tới quyết định mua sai. Hãy đo MFU/MBU (DCGM `SM_ACTIVE`, `TENSOR_ACTIVE`, `DRAM_ACTIVE`) và sửa nguyên nhân *trước* khi hạ cấp hay mua thêm GPU.

## Hành động đề xuất (xếp theo ROI = tiết kiệm / công sức & rủi ro)

| # | Hành động | Tiết kiệm/tháng | Công sức / rủi ro | Vì sao ở vị trí này |
|---|---|---|---|---|
| 1 | Tắt GPU idle (auto-shutdown khi util < 10% trong 30 phút) | $600 | Vài giờ, không rủi ro | Tiền mất trắng, sửa bằng một policy |
| 2 | Chuyển 5 job interruptible sang spot + checkpoint | $5,926 | Thấp: job đã checkpoint được | Lever lớn nhất không cần cam kết dài hạn |
| 3 | Cascade + batch + cache có điều kiện cho inference | $1,200 | Thấp: đổi routing/config | Giảm $/1M-token API 82% |
| 4 | Sửa util-lie & right-size theo MBU (gộp 2×A100 → 1×MI300X, 2×A10G → 1×A100) | $1,836 | Trung bình: cần benchmark, MI300X đổi sang ROCm | Phải làm **trước** bước 5 |
| 5 | Reserved 3yr cho inference 24/7 sau khi đã right-size | $3,866 | Cao: cam kết 36 tháng | Lớn nhưng khoá loại GPU; reserve sai GPU là khoá lãng phí 3 năm |
| 6 | Cap reasoning ≤5% traffic + dời job train sang vùng sạch | $7 + điện | Thấp | $ nhỏ nhưng giảm 236 kWh/tháng và 92% CO2 của job train |

Nguyên tắc: làm việc rẻ & không hối tiếc trước (idle, spot, routing), đo lại MFU/MBU, rồi mới right-size, và chỉ cam kết reserved khi fleet đã đúng kích thước.

## Sustainability: carbon gắn với chi phí

- Năng lượng/truy vấn chuẩn (800 token): 0.24 Wh; truy vấn reasoning trung bình 148 Wh (173×).
- Job interruptible tiêu thụ 2,147 kWh/tháng (TDP × GPU × giờ × PUE 1.2).

| Vùng | $/kWh | gCO2/kWh | Tiền điện | CO2 (kg) | Điện + CO2 @ $100/t | Độ trễ tới user |
|---|---|---|---|---|---|---|
| europe-north1 | 0.090 | 30 | $193 | 64 | $200 | transatlantic |
| us-east-wa | 0.055 | 90 | $118 | 193 | $137 | near |
| us-west-2 | 0.070 | 120 | $150 | 258 | $176 | cross-US |
| us-east-1 | 0.120 | 380 | $258 | 816 | $339 | local |
| europe-central2 | 0.180 | 660 | $386 | 1,417 | $528 | transatlantic |

- **Sạch nhất:** europe-north1 (thuỷ điện Na Uy): dời job train từ us-east-1 giảm 751 kg CO2e (−92%) và $64 tiền điện.
- **Rẻ nhất / cân bằng nhất:** us-east-wa ($0.055/kWh, 90 gCO2/kWh): rẻ hơn us-east-1 $140 và vẫn giảm 76% CO2; cũng thắng khi định giá carbon $100/t.
- **Dơ nhất:** europe-central2 (660 gCO2/kWh) vừa đắt điện vừa phát thải gấp 22× europe-north1. Chọn vùng là lever hiếm hoi giảm cả $ lẫn carbon cùng lúc.
- **Trade-off:** chỉ job train/batch/eval (không nhạy độ trễ) nên dời đi; inference realtime giữ gần user. Cần xét thêm data residency và phí egress khi chuyển dataset qua Đại Tây Dương.

## Phần mở rộng "Your Turn" (đã thực hiện cả 5)

1. **Tier policy v2** (`pricing.recommend_tier/tier_costs`): interruption rate theo GPU (H100 3%, H200 3%, B200 4%, A100 5%, MI300X 6%, A10G 8%, L4 8%), reserved tính trung thực 24h/ngày, so 1yr vs 3yr theo horizon. Savings purchasing: v1 39.1% → v2 38.2%. v2 thấp hơn vì v1 chỉ tính tiền reserved cho giờ chạy (job-infer-search L4 18h/ngày bị đánh giá thấp). Độ nhạy horizon: 6 tháng = 23.1%, 12 tháng = 29.7%, 24 tháng = 29.7%, 36 tháng = 38.2%.
2. **Right-size theo MBU** (M1): 2x A100 → 1x MI300X (cần 134.7 GB, 2.31 TB/s; rẻ nhất theo $/GPU-hr sẽ là 8x L4 $6.40/h); 2x A10G → 1x A100 (cần 40.7 GB, 0.66 TB/s; rẻ nhất theo $/GPU-hr sẽ là 3x L4 $2.40/h). Tổng $1,325/tháng. MI300X có $/GB-VRAM thấp nhất ($0.010) và 5.3 TB/s băng thông, nên một GPU gộp được nhiều replica.
3. **`cache_is_worth_it()`**: break-even 0.28 lần đọc/prefix (write 1.25×, read 0.1×). Bật cache cho assistant, rag (1.03/1.03 lần đọc/lần ghi), tắt cho eval, search (0.08/0.08). Với cache tính phí lưu trữ ($1/1M-token-giờ), break-even là 5.56 lần đọc cho model nhỏ và 0.37 cho model lớn. Bật cache cho mọi team: $9.11/ngày, còn gated: $8.87/ngày.
4. **Ngân sách reasoning**: 8.4% traffic → 15.9% chi phí và 94.0% năng lượng. Cap 10% không có tác dụng (đã dưới ngưỡng). Cap 5%: −$0.23/ngày, −7.9 kWh/ngày. Rule đề xuất: chỉ bật reasoning cho task eval nhiều bước khi classifier độ khó > ngưỡng (≈ đầu ra kỳ vọng ≥ 3,240 token), kèm `max_thinking_tokens` và hạn mức Wh/ngày cho mỗi team.
5. **Carbon-aware scheduling** (`missions/m6_carbon_scheduling.py`): xem bảng 5 vùng ở trên. Sạch nhất europe-north1, rẻ nhất/cân bằng us-east-wa.

_Figures are June-2026 as-of snapshots; re-baseline before acting._