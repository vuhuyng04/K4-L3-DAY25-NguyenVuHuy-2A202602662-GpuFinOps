# Lab 25 — GPU FinOps: Bài viết ngắn

**Học viên:** Nguyễn Vũ Huy · 2A202602662
**Kết quả kiểm tra:** `verify.py` 11/11 · `pytest` 26 passed (15 test gốc + 11 test cho extension)
**Số liệu chi tiết:** [`outputs/report.md`](outputs/report.md) · biểu đồ [`outputs/savings.png`](outputs/savings.png)

## 1. Baseline vs. Optimized

| | Baseline | Optimized | Thay đổi |
|---|---|---|---|
| Tổng chi / tháng | $27,133 | $13,705 | **−49.5%** ($13,428) |
| $/1M-token blended (toàn bộ chi ÷ 226M token/tháng) | $120.06 | $60.64 | −49.5% |
| $/1M-token riêng inference API | $6.488 | $1.178 | −81.8% |

## 2. Đòn bẩy nào đóng góp nhiều nhất

| Lever | $/tháng | % tổng tiết kiệm |
|---|---|---|
| Purchasing (spot + reserved) | $9,792 | 73% |
| Right-size (util-lie + MBU) | $1,836 | 14% |
| Inference (cascade/cache/batch) | $1,200 | 9% |
| Kill idle GPUs | $600 | 4% |

**Purchasing** lớn nhất vì nó tác động vào phần lớn nhất của hoá đơn: thuê GPU cho workload ($25.7k/tháng on-demand). Spot cho 5 job interruptible tiết kiệm $5,926, reserved 3yr cho 3 job inference 24/7 tiết kiệm $3,866. **Inference** chỉ đóng góp 9% về số tiền tuyệt đối nhưng giảm unit cost mạnh nhất: −82% $/1M-token. Trong đó **cascade** chiếm phần lớn ($6.49 → $1.52/1M) vì 79% request đủ dễ cho model nhỏ, vốn rẻ hơn 15–37.5×. Cache và batch là chiết khấu nhân trên phần còn lại.

## 3. GPU-Util Lie

`gpu-h100-4` báo **98.2% GPU-Util nhưng MFU chỉ 0.194**, trong khi các H100 cùng cụm đạt 0.40–0.43. `nvidia-smi` đo tỷ lệ *thời gian* có kernel chạy, không đo SM hay Tensor Core có thực sự làm việc. Batch nhỏ, launch overhead, dataloader chậm, chờ NCCL hay memory stall đều khiến GPU "bận" mà không sinh FLOPs. Tác động tài chính: ta trả đủ $1,800/tháng cho chiếc H100 này nhưng nhận chưa tới một nửa FLOPs, tương đương **$927/tháng trả cho FLOPs không nhận được**. Mỗi FLOP hữu ích đắt gấp ~2.1× so với GPU khoẻ. `gpu-a10g-1` (97% util, MBU 0.30) là bản memory-bound của cùng hiện tượng: lãng phí $358/tháng. Nguy hiểm nhất là dashboard GPU-Util sẽ báo "hết công suất, cần mua thêm".

## 4. Phần mở rộng đã làm (cả 5)

1. **Tier policy v2** (`finops/pricing.py`: `tier_costs`, `recommend_tier(..., prices=...)`): thêm interruption rate theo GPU, tính reserved 24h/ngày (cam kết không tạm dừng khi idle) và so 1yr vs 3yr theo planning horizon. Kèm ma trận GPU × duty cycle × interruptible. *Kết quả:* savings purchasing **39.1% → 38.2%**. Policy mới cho số thấp hơn vì v1 lạc quan sai: nó tính tiền reserved chỉ cho giờ chạy, nên job L4 18h/ngày được tính $972 thay vì $1,296. Horizon quyết định term: 36 tháng → 3yr (38.2%), 12–24 tháng → 1yr (29.7%), 6 tháng → hầu hết on-demand (23.1%). *Insight:* interruption rate chỉ đổi quyết định khi chiết khấu spot mỏng (~12%). Với chiết khấu 40% của H100, spot vẫn thắng.
2. **Right-sizing theo MBU** (M1): tính $/GB-VRAM và $/(TB/s), rồi size lại từng nhóm replica theo VRAM đỉnh và băng thông đỉnh ÷ MBU mục tiêu 0.6. *Kết quả:* 2×A100 → 1×MI300X ($1,174/tháng), 2×A10G → 1×A100 ($151), **tổng $1,325/tháng**. *Vì sao không chọn GPU rẻ nhất theo $/GPU-hr:* L4 ($0.80/h) chỉ có 0.3 TB/s. Phục vụ nhóm A100 bằng L4 cần 8 chiếc ($6.40/h, đắt gấp 3.3× MI300X). Map hạ cấp ngây thơ A10G → L4 trong M5 gốc còn *không đủ* băng thông.
3. **`cache_is_worth_it()`**: break-even = (write premium + phí lưu trữ) / (1 − read discount) = **0.28 lần đọc** (Anthropic write 1.25×). Dataset: assistant/rag ~1.03 lần đọc/lần ghi → bật; search/eval ~0.08 → tắt. Với cache tính phí lưu trữ $1/1M-token-giờ, model nhỏ cần **5.56** lần đọc còn model lớn chỉ cần **0.37**. Gated cache rẻ hơn bật cache cho mọi team ($8.87 so với $9.11/ngày). M2 gốc bỏ qua phí ghi cache nên lạc quan $0.39/ngày.
4. **Ngân sách reasoning**: reasoning là **8.4% traffic nhưng 15.9% chi phí và 94% năng lượng**: 148 Wh/truy vấn, gấp 173× truy vấn thường (80× do multiplier năng lượng, nhân với ~2.2× số token vì output dài gấp 6). Cap 10% **không có tác dụng** vì traffic đã dưới ngưỡng. Cap 5% tiết kiệm 7.9 kWh/ngày (236 kWh/tháng). Bỏ hẳn tiết kiệm 888 kWh/tháng. Đề xuất rule: chỉ bật reasoning cho task eval nhiều bước khi classifier đánh giá độ khó cao, kèm `max_thinking_tokens` và hạn mức Wh/ngày cho mỗi team.
5. **Carbon-aware scheduling** (`missions/m6_carbon_scheduling.py`): 5 job interruptible tiêu thụ 2,147 kWh/tháng. Chuyển us-east-1 → **europe-north1** giảm **751 kg CO2e (−92%)** và $64 tiền điện. **us-east-wa** rẻ nhất ($118) và cân bằng nhất, vẫn giảm 76% CO2. Nó cũng thắng khi định giá carbon $100/t. Vùng "tối ưu" phụ thuộc ưu tiên: cam kết net-zero chọn europe-north1, tối ưu tiền chọn us-east-wa. Inference realtime vẫn giữ gần user vì latency và data residency.

## 5. Ba hành động đầu tiên nếu tôi là FinOps lead

1. **Tuần 1, không hối tiếc:** bật auto-shutdown GPU idle ($600/tháng), chuyển job interruptible sang spot + checkpoint ($5,926), bật cascade + batch + cache có điều kiện cho inference ($1,200). Không cần cam kết dài hạn, rollback dễ.
2. **Đo đúng trước khi mua:** đưa MFU/MBU (DCGM) vào dashboard thay cho GPU-Util. Sửa nguyên nhân gốc của `gpu-h100-4` và right-size fleet inference theo băng thông ($1,836/tháng).
3. **Rồi mới cam kết:** chỉ mua reserved 3yr ($3,866/tháng) cho inference 24/7 *sau khi* fleet đã đúng loại GPU. Song song đó, bật chargeback (tag coverage 92% ≥ 80%) để mỗi team thấy $/1M-token của mình, và đặt ngân sách reasoning theo Wh.

_Số liệu là snapshot giá tháng 6/2026 trên dữ liệu tổng hợp seed=25. Cần re-baseline trước khi áp dụng thật._
