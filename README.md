# Đồ án Virtual Try-On với VITON-HD

## 1. Nguồn gốc đồ án

Đồ án này được phát triển dựa trên source code gốc **VITON-HD**:

- Bài báo: [VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization](https://arxiv.org/abs/2103.16874)
- GitHub gốc: [shadow2496/VITON-HD](https://github.com/shadow2496/VITON-HD)
- Tác giả gốc: Seunghwan Choi, Sunghyun Park, Minsoo Lee, Jaegul Choo (CVPR 2021)

Mục tiêu của repo này là giữ pipeline inference của VITON-HD, đồng thời bổ sung một số chức năng để hướng tới có thể chạy trên dataset của người dùng (bao gồm ảnh quần áo và ảnh người, không cần mask và pose map sẵn nhưng vẫn cần human-parse).

## 2. Các điều chỉnh so với source gốc

So với VITON-HD ban đầu, đồ án hiện tại đã bổ sung/cập nhật:

- **Custom mask cho quần áo**
  - Thêm cờ `--custom_mask` trong `test.py`.
  - Khi bật cờ này, `datasets.py` sẽ gọi `generate_cloth_mask` từ `gen_mask.py` thay vì đọc mask có sẵn trong `datasets/test/cloth-mask`.
  - Hỗ trợ cache qua `--custom_mask_cache_dir` (ví dụ: `test_mask`).

- **Custom pose từ YOLO + render pose image**
  - Fine-tune model YOLO pose với ground truth là openpose-json.
  - Thêm cờ `--custom_pose` trong `test.py`.
  - Tự động sinh:
    - JSON keypoints vào `test_pose/` (qua `gen_pose.py`)
    - Ảnh pose `_rendered.png` vào `test_pose_img/` (qua `gen_pose_img.py`)
  - Các thư mục `datasets/test/openpose-json` và `datasets/test/openpose-img` được giữ nguyên để không ảnh hưởng bộ data gốc.
 
- **Custom parse từ mô hình RES-UNet: image + pose map -> parse map**
  - Huấn luyện mô hình RES-Unet nhỏ nhằm tạo human parse chính xác nhất so với VITON-HD parse-v3.
  - Thêm cờ `--custom_parse` trong `test.py`.
  - Tự động sinh:
    - Parse mask vào `test_parse_mask/` (qua `gen_parse.py`)
    - Parse mask color vào `test_parse/`
  
- **Fix tương thích Kornia/PyTorch**
  - `test.py`: đổi Gaussian blur sang `kornia.filters.GaussianBlur2d`.
  - `networks.py`: bỏ `.cuda()` để chạy được trên môi trường CPU-only.

- **Tiện ích bổ sung**
  - `fix_pose.py`: bổ sung logic vá keypoint bị thiếu.
  - `measure.py`: chạy test các chỉ số của kết quả (so sánh với ground truth)
 
- **Link huấn luyện mô hình**
  - Nhóm huấn luyện trên Kaggle để tiết kiệm thời gian và chi phí.
  - Link (bao gồm cả RES-Unet và fine-tune YOLO): [Kaggle](https://www.kaggle.com/code/minhnewbi/vton-hd)

## 3. Cài đặt

### 3.1 Tạo môi trường

```bash
conda create -n viton_hd python=3.8 -y
conda activate viton_hd
```

### 3.2 Cài PyTorch

- Nếu dùng GPU CUDA, cài theo hướng dẫn chính thức của PyTorch phù hợp driver/CUDA trên máy.
- Nếu dùng CPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3.3 Cài thư viện cần thiết

```bash
pip install opencv-python pillow numpy kornia ultralytics rembg onnxruntime tqdm
```

## 4. Cấu trúc dữ liệu tối thiểu để test

Đặt dữ liệu theo cấu trúc VITON-HD:

- `datasets/test/image/`
- `datasets/test/image-parse/`
- `datasets/test/cloth/`
- `datasets/test/cloth-mask/` (chỉ cần nếu không dùng `--custom_mask`)
- `datasets/test/openpose-json/` và `datasets/test/openpose-img/` (chỉ cần nếu không dùng `--custom_pose`)
- `datasets/test_pairs.txt`

- Drive dẫn đến data để train / test: [Data](https://drive.google.com/file/d/1tLx8LRp-sxDp0EcYmYoV_vXdSc-jJ79w/view?usp=sharing)

Checkpoint đặt trong:

- `checkpoints/seg_final.pth`
- `checkpoints/gmm_final.pth`
- `checkpoints/alias_final.pth`
- `checkpoints/best_pose_unet.pth`

- Drive dẫn đến link checkpoint: [Data](https://drive.google.com/drive/folders/1c0uoMN44aCYdA7GXYhjkw7cqdPBuDeg1?usp=sharing)

## 5. Cách thử nghiệm

### 5.1 Chạy mặc định (giống VITON-HD gốc)

```bash
python test.py --name alias_final
```

### 5.2 Dùng mask tự tạo thay cho cloth-mask có sẵn

```bash
python test.py --name run_custom_mask --custom_mask
```

### 5.3 Dùng pose tự tạo (JSON + pose image) cho đúng danh sách test

```bash
python test.py --name run_custom_pose --custom_pose
```

Mặc định sẽ ghi:
- JSON: `test_pose/`
- Pose image: `test_pose_img/`

Có thể đổi đường dẫn:

```bash
python test.py --name run_custom_pose \
  --custom_pose \
  --custom_pose_json_dir test_pose \
  --custom_pose_img_dir test_pose_img \
  --custom_pose_model yolov8n-pose.pt \
  --custom_pose_device cpu
```

### 5.4 Kết hợp custom mask + custom pose

```bash
python test.py --name run_all_custom \
  --custom_mask --custom_mask_cache_dir test_mask \
  --custom_pose
```

Kết quả try-on được lưu ở:

- `results/<name>/`

## 6. Cách test

- Để test 1 hoặc vài cặp, hãy sửa `datasets/test_pairs.txt` thành đúng các dòng cần test.
- Khi dùng `--custom_pose`, repo đọc pose từ `test_pose` và `test_pose_img`.

## 7. Tham khảo và trích dẫn

Nếu bạn sử dụng source, vui lòng trích dẫn bài báo VITON-HD:

```bibtex
@inproceedings{choi2021viton,
  title={VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization},
  author={Choi, Seunghwan and Park, Sunghyun and Lee, Minsoo and Choo, Jaegul},
  booktitle={Proc. of the IEEE conference on computer vision and pattern recognition (CVPR)},
  year={2021}
}
```
