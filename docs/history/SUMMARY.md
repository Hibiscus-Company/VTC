# Tổng kết dự án — Novel-View Synthesis cho cuộc thi VAR 2026 (Digital Twin trạm BTS)

> Tài liệu này viết cho người **chưa từng đụng vào** dự án. Đọc xong sẽ hiểu:
> bài toán là gì, đang ở đâu, đã thử gì, cái gì thắng/thua và **tại sao**, và
> nên đi hướng nào tiếp. Cập nhật lần cuối: sau khi nộp bản g2 (điểm 74.77).

---

## 1. Bài toán (giải thích từ đầu)

**3D Gaussian Splatting (3DGS)** là kỹ thuật dựng lại một cảnh 3D từ nhiều ảnh
chụp, bằng cách biểu diễn cảnh thành hàng triệu "hạt Gaussian" 3D có màu và độ
mờ. Sau khi tối ưu, ta có thể **render (vẽ) ảnh của cảnh từ một góc nhìn mới bất
kỳ** — gọi là *Novel-View Synthesis (NVS)*.

**Cuộc thi:** VAR 2026 "Digital Twin cho trạm BTS" (cột phát sóng). Với mỗi
"scene" (một trạm BTS chụp bằng drone), ban tổ chức cho:
- Ảnh train (drone bay quanh trạm, ~100–240 ảnh)
- Kết quả COLMAP (vị trí camera + đám mây điểm thưa)
- File `test_poses.csv`: danh sách các **góc nhìn cần vẽ ra** (test poses)

**Nhiệm vụ:** vẽ ảnh RGB tại các test pose đó. Nộp file zip gồm ảnh của 13 scene.

**Chấm điểm** (so ảnh vẽ vs ảnh thật, trung bình):
```
Score = 0.4·(1 − LPIPS) + 0.3·SSIM + 0.3·clamp(PSNR / 50)
```
- **LPIPS**: độ khác biệt "cảm quan" (deep features), càng thấp càng tốt.
- **SSIM**: độ giống cấu trúc, càng cao càng tốt.
- **PSNR**: sai số pixel, càng cao càng tốt; chuẩn hóa chia cho **50** (đã xác
  nhận chính xác từ điểm chính thức).

> **Insight #1 — quan trọng nhất:** vì PSNR chia cho 50, phần PSNR chỉ chiếm
> ~20% điểm và **gần bão hòa** (thêm 1 dB chỉ +0.006). **LPIPS có trọng số 0.4
> và nhiều dư địa nhất → LPIPS là metric đáng đánh nhất.** Mọi tối ưu nên nhắm
> vào chất lượng cảm quan (nét, sạch floater), KHÔNG chạy theo PSNR thô.

---

## 2. Dữ liệu (13 scene)

| Loại | Scene | Có ảnh GT test? |
|------|-------|-----------------|
| **Public (5)** | hcm0031, hcm0034, HCM0181, HCM0193, HCM0204 | **CÓ** → chấm điểm được tại máy |
| **Private (8)** | HCM0249/0254/0276/1439, HNI0131/0265/0366/0437 | Không |

- Ảnh **1320×989** (đã downscale ~1/4 từ ảnh drone gốc). Camera **SIMPLE_RADIAL**.
- **Test pose là NỘI SUY trong đường bay drone** (đã kiểm chứng: mỗi test pose
  trùng khít một entry trong `images.bin` của COLMAP). Đây là chế độ NVS **dễ**
  (không phải ngoại suy). Đường bay: orbit + lưới trên cao quanh cột.

**EDA (khảo sát dữ liệu tự động, script `eda.py`):**

| Thuộc tính | Đo được | Ý nghĩa |
|---|---|---|
| Exposure drift (độ lệch sáng giữa ảnh) | **2–5% (thấp)** | Không cần Appearance Embedding |
| Tỉ lệ trời | **15–22%** | Vùng vô cực → dễ sinh floater → hại LPIPS |
| Scale spread (xa/gần) | **2.3–4.1×** | Có aliasing → Mip-Splatting có lý do |
| Mật độ điểm COLMAP | **0.01–0.27 (lệch 27×)** | Scene thưa cần densify mạnh hơn |
| Extent (kích thước cảnh) | 7.9–10.6 (đồng đều) | Một config scale dùng chung được |
| Distortion `k` | ~+0.008; HNI0131/0265 = **−0.115** | Đã xử lý bằng undistort |

---

## 3. Môi trường & cách chạy (cho người mới)

- **GPU:** 2× RTX 5070 Ti 16GB (Blackwell sm_120), CUDA 12.8, PyTorch 2.7.1.
- **Conda env: `fastgs2`** (KHÔNG dùng `fastgs` cũ — py3.7/cu11.6 không build
  được cho sm_120). Xem chi tiết build trong memory `fastgs-env-and-build`.
- **Dataset:** gốc ở `/mnt/d/avv/data/phase1` (mount chậm); bản làm việc nhanh
  ở `~/data/phase1` (ext4). Ảnh train đã undistort ở `<scene>/train/images_undist`.
- ⚠️ **Ổ C: kinh niên đầy 100%** → **luôn build submission/output nặng trên
  ext4 (`~/`)**, không phải C:. Đã cắn 2 lần (lỗi `OSError errno 5` khi ghi zip).

**Pipeline 1 scene (train → render → chấm):**
```bash
GRAD_ABS=0.0002 IMAGES_DIR=images_undist DISTORT=auto \
  bash run_scenes.sh <gpu_id> g2 ~/data/phase1/public_set/<scene>
```
`run_scenes.sh` tự: train 30k iter → render test poses (kèm warp distortion) →
chấm điểm nếu có GT. Có retry 3 lần, `--data_device cpu` (né VRAM).

**Đóng gói submission:**
```bash
# re-encode renders về JPG q95 (~700KB/ảnh, tên đúng CSV) rồi zip trên ext4
python make_submission.py --renders_root <dir> \
  --data_roots ~/data/phase1/public_set ~/data/phase1/private_set1 \
  --out ~/submission.zip
```

---

## 4. Kết quả hiện tại

| Bản nộp | Điểm private | PSNR | SSIM×100 | LPIPS×100 |
|---------|-------------|------|----------|-----------|
| Baseline + undistort | 74.348 | 24.850 | 83.61 | 14.11 |
| **+ g2 (hiện tại)** | **74.769** | tăng nhẹ | tăng nhẹ | **13.31** |

Điểm public g2 tại máy (trung bình **0.7485**): hcm0031 .7405 · hcm0034 .7558 ·
HCM0181 .7492 · HCM0193 .7455 · HCM0204 .7515. Bản nộp: `~/submission_round1_g2.zip`.

---

## 5. Những gì ĐÃ THỬ — thắng và thua (kèm lý do)

### ✅ Cái THẮNG
| Kỹ thuật | Hiệu quả | Vì sao ăn |
|----------|----------|-----------|
| **Undistort + warp lại** (SIMPLE_RADIAL→pinhole để train, warp render về hình méo để khớp GT) | **+0.031** | Sửa lệch hình học 5–75px ở rìa ảnh |
| **g2**: `grad_abs_thresh 0.0004→0.0002` (hạ ngưỡng densification → nhiều Gaussian hơn) | **+0.0044** trung bình, **zero regression** | Thêm capacity → fit chi tiết tốt hơn → **LPIPS giảm** |

### ❌ Cái THUA (đều đã test bằng số, đừng thử lại)
| Kỹ thuật | Kết quả | Vì sao thua |
|----------|---------|-------------|
| **Supersampling** (render 2× rồi thu nhỏ) | **−0.020** | Model đã khớp GT ở native res; thu nhỏ làm ảnh mượt hơn GT → lệch mọi metric |
| **Edge-TV loss** (DET-GS, làm mượt vùng phẳng) | **−0.001** | Over-smooth, LPIPS tăng |
| **Densify lâu hơn** (`densify_until 15k→20k`) | **−0.025**, Gaussian tụt 2.3M→1.1M | Lịch densify/prune/opacity-reset của FastGS tune chặt quanh 15k; dời ra → mất cân bằng, model co lại |
| **Opacity-sparsity loss** (entropy đẩy opacity→0/1, nhắm floater) λ=0.01 và 0.0005 | **−0.048 / −0.034**, Gaussian tụt 50%/40% | FastGS đã prune opacity<0.005 mỗi 100 iter; entropy + ngưỡng cứng → over-remove cả geometry thật. **Thừa với pruning của FastGS** (như AbsGS) |
| `mult` 0.7 vs 1.0 | ~0 (neutral) | Compact-box không hy sinh chất lượng |
| `g1` (`grad_abs 0.0001`) | Hỗn hợp: thắng scene dày, thua scene thưa | Scene thưa bão hòa ~2.6M Gaussian |

### 🐛 Bug CUDA đã sửa (cần biết nếu build lại rasterizer)
FastGS crash ngẫu nhiên "illegal memory access" trên Blackwell. **Nguyên nhân
gốc:** hàm đếm tile trong `auxiliary.h` cộng `max_tile_v - min_tile_v` có thể ÂM
→ lưu vào uint32 thành ~4 tỉ → tràn buffer. Fix: `max(0, ...)`. Cùng vài fix
phụ (`#include <cstdint>`, tràn shared-mem của fused-ssim, guard prefetch
backward). Chi tiết ở memory `fastgs-cuda-crash-fixes`.

---

## 6. KẾT LUẬN rút ra (phần quan trọng nhất)

1. **LPIPS là vua** (weight 0.4, PSNR chia 50 nên gần bão hòa). Nhắm cảm quan.
2. **Test pose nội suy dày → model đã fit GT native rất tốt.** Hệ quả lớn:
   **mọi thứ smooth / resample / regularize / đổi lịch để "cải thiện" đều kéo
   render RỜI khỏi GT → hại điểm.** 3/3 lever rẻ thua vì lý do này.
3. **Chỉ THÊM CAPACITY mới thắng** (g2 = nhiều Gaussian hơn). Nhưng ngay cả cách
   thêm capacity hiển nhiên (densify lâu hơn) cũng hỏng vì lịch FastGS cứng.
   **Nhánh BỎ-FLOATER cũng đóng**: FastGS đã prune floater rất mạnh (đa-view,
   mỗi 100 iter, min_opacity 0.005) → mọi áp lực sparsity/opacity thêm vào đều
   thừa và cắt cả geometry thật (opacity-sparsity thua ở mọi λ). Sky-mask/sky-dome
   nhiều khả năng cũng vướng lý do tương tự — cần cân nhắc kỹ trước khi làm.
4. **FastGS đã tích hợp sẵn AbsGS** (backward tích `fabs(grad)`, `grad_abs_thresh`
   chính là knob đó) → đừng ghép AbsGS/Pixel-GS, thừa.
5. **Bỏ Appearance Embedding**: drift chỉ 2–5%, và test pose không có "mã ngoại
   hình" nên lúc infer chỉ dùng được mã trung bình → vô ích, thậm chí hại.
6. **g2 (~74.77) gần TRẦN của họ FastGS trên bài này.** Không còn "cú nhảy đơn
   giản" nào. Muốn nhảy phải **đổi method**, không phải tinh chỉnh tham số.
7. **Mục tiêu 90+ không thực tế:** toán điểm cho thấy cần tái tạo gần hoàn hảo
   (LPIPS~0.04, SSIM~0.95, PSNR~35). Trần thực tế khi xếp chồng stack ≈ **78–83**.

---

## 7. Thuận lợi & Khó khăn

**Thuận lợi:**
- Test pose nội suy (chế độ dễ) → không cần model tổng quát hóa mạnh.
- Có GT trên 5 scene public → **chấm điểm & A/B tại máy** (vòng lặp nhanh).
- Drift sáng thấp, scale đồng đều → 1 config robust transfer tốt.
- Pipeline đã chạy ổn định (đã fix hết crash), 2 GPU song song.

**Khó khăn:**
- **Đã gần trần của phương pháp hiện tại** — gains rẻ đã cạn.
- Chỉ 5 scene public để validate → **dễ overfit**, param không chắc transfer
  sang private (nên chọn config robust, không tune tay 13 scene).
- Ổ C: đầy kinh niên → phải quản lý output cẩn thận (build trên ext4).
- Không có tín hiệu test-time (không thể fine-tune theo ảnh test).
- Bước nhảy tiếp đòi hỏi tích hợp method mới (~1 ngày), rủi ro cao hơn A/B.

---

## 8. Hướng đi tiếp (chưa làm — xếp theo tiềm năng)

1. **gsplat + MCMC densification** (khuyến nghị) — paradigm densification khác
   hẳn FastGS (phủ đều hơn, không kẹt lịch 15k), **kèm Mip antialiasing sẵn**.
   Đổi rasterizer (Apache-2.0). ~1 ngày. ⚠️ AA có thể làm Gaussian mảnh biến mất
   (gsplat issue #840) — phải validate trên public.
2. **Train-time Mip-Splatting** — spec sẵn ở `MIP_PLAN.md` (branch `mip-splatting`).
   Rẻ hơn gsplat chút nhưng kết quả bấp bênh (supersampling gợi ý AA có thể lệch
   GT). Cần sửa CUDA forward+backward, có FD-check gradient.
3. **Two-stage sky dome** — dựng nền trời riêng để dọn floater 15–22% vùng trời
   (mask sky loss đơn thuần KHÔNG đủ vì trời vẫn bị chấm lúc test).
4. **g1 per-scene** (micro-squeeze): dùng `grad_abs 0.0001` cho scene dày —
   +0.0015 nhưng không validate được trên private, rủi ro. Marginal.

**Đã loại (có bằng chứng):** appearance embedding, dynamic-object masking (trạm
BTS tĩnh), AbsGS/Pixel-GS (thừa), 2DGS (tối ưu geometry, thường tụt PSNR
photometric), mọi mẹo test-time-only.

---

## 9. Bản đồ file/script

| File | Công dụng |
|------|-----------|
| `run_scenes.sh` | Driver train+render+eval 1 scene (env: GRAD_ABS, IMAGES_DIR, DISTORT, MULT) |
| `render_test_poses.py` | Render CSV poses; `--distort auto` (warp méo), `--supersample`, `--png_quantize` |
| `eval_test_renders.py` | Chấm PSNR/SSIM/LPIPS/Score vs GT |
| `make_submission.py` | Validate (tên/size/số lượng) + đóng zip |
| `undistort_scene.py` | Undistort ảnh train → `images_undist` |
| `viz_pointcloud.py` | Vẽ point cloud sanity-check (6 panel), output ở `viz/` |
| `eda.py` (scratchpad) | Khảo sát exposure/sky/scale/density 13 scene |
| `PROMPT.md` | Research brief (đưa cho Claude khác research SOTA) |
| `MIP_PLAN.md` | Spec tích hợp Mip-Splatting (branch `mip-splatting`) |
| `3DGS Drone Digital Twin.docx` | Kết quả deep-research SOTA (đã phân tích ở §8) |

**Thay đổi code đáng chú ý (tất cả OFF ở default → pipeline g2 không bị ảnh
hưởng):** `train.py` thêm `edge_aware_tv_loss` (`--lambda_tv`, default 0) và
untie lịch prune theo `densify_until_iter` (inert khi =15000);
`render_test_poses.py` thêm `--supersample` (default 1).
