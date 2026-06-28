# Đồ án cuối kì môn Nhận dạng 
Hướng dẫn sử dụng 
### Tải dataset 
**Ubuntu or Colab**
```
chmod +x ./download_dataset.sh
./download_dataset.sh
```

### Tải model weights 
**Ubuntu or Colab**
```
chmod +x ./download_model.sh
./download_model.sh
```
### Cấu hình chạy eval 
chỉnh ```config/config_eval.py``` các thuộc tính sau 

- ```config.model_name``` : tên model , có thể là baseline hoặc clip hoặc dinov2

- ```config.model_path``` : đường dẫn đến file trọng số

- ```config.eval_path``` : đường dẫn đến folder chứa tập val 

- ```config.val_targets``` : danh sách dataset được dùng để eval

### Chạy Eval 
mở terminal gõ 
```
python src/eval.py
```
file eval.py sẽ load full dataset vào RAM , nếu máy ít RAM thì dùng 
```
python src/eval_optimized.py
```
### Chạy Eval trên tất cả các model 1 lần 
Tìm đến ```python src/eval_all.py```
sửa 
- ```DATASETS``` :danh sách các dataset dùng để eval
- ```MODELS_TO_EVAL``` : danh sách các model được dùng để eval 
```
python src/eval_all.py
```

---

##  Các Tập Dữ Liệu Được Sử Dụng Để Đánh Giá (Datasets Used for Evaluation)

Để kiểm thử một cách nghiêm ngặt khả năng nhận diện khuôn mặt, framework tiến hành đánh giá các mô hình trên 6 tập dữ liệu tiêu chuẩn (standard unconstrained face verification benchmarks). Toàn bộ các tập dữ liệu này đều được nạp trực tiếp từ các tệp nhị phân nén (`.bin`) để loại bỏ nút thắt cổ chai về đọc/ghi ổ cứng (Disk I/O bottlenecks) và đảm bảo tính tái lập (reproducible evaluation) của kết quả.

| Tên Tập Dữ Liệu (Dataset) | Tổng Số Lượng Ảnh (Samples) | Thử Thách / Khía Cạnh Đánh Giá Chính |
| :--- | :---: | :--- |
| **LFW** (Labeled Faces in the Wild) | 12,000 | Bộ đo kiểm tiêu chuẩn cho bài toán face verification trong môi trường tự do. |
| **CFP-FP** (Celebrities in Frontal-Profile) | 14,000 | Đánh giá sự thay đổi góc mặt cực hạn (so sánh ảnh góc thẳng - Frontal vs. ảnh góc nghiêng - Profile). |
| **CFP-FF** (Celebrities in Frontal-Frontal) | 14,000 | Bộ đo kiểm cơ sở cho việc xác thực khuôn mặt góc thẳng trong môi trường tự do. |
| **AgeDB-30** (Age Database) | 12,000 | Xác thực khuôn mặt bất biến theo thời gian với khoảng cách tuổi tác cố định là 30 tuổi. |
| **CALFW** (Cross-Age LFW) | 12,000 | Xác thực khuôn mặt bất biến theo tuổi tác dựa trên phân phối dữ liệu thế giới thực. |
| **CPLFW** (Cross-Pose LFW) | 12,000 | Xác thực khuôn mặt bất biến theo góc quay, tập trung vào các góc xoay không gian phức tạp. |

### Cơ Chế Stream Dữ Liệu Theo Batch Tối Ưu Bộ Nhớ (Memory-Optimized Batch Streaming)
Khác với các triển khai đánh giá thông thường – vốn giải nén và nạp đồng thời toàn bộ mảng ảnh thô vào RAM hệ thống cùng một lúc (gây tiêu tốn hơn **21 GB RAM**), cấu hình tối ưu của chúng tôi hoạt động như một bộ sinh luồng dữ liệu (*Stream Generator*).
* Hệ thống chỉ đọc cấu trúc byte nhị phân thô ban đầu, và chỉ tiến hành giải mã (decode) ảnh theo nhu cầu (on-demand) cho đúng số lượng ảnh thuộc Batch hiện tại.
* Ngay sau khi Batch hiện tại hoàn thành quá trình lan truyền xuôi (Forward propagation) qua mạng AI, các mảng ảnh thô sẽ lập tức bị giải phóng khỏi bộ nhớ thông qua cơ chế giải phóng bộ đệm của PyTorch (`torch.cuda.empty_cache()`), giúp giới hạn mức chiếm dụng RAM tối đa luôn dưới mức **2.5 GB**.

---

##  Cấu Hình Thiết Lập Đánh Giá (Evaluation Configurations)

Pipeline đánh giá được điều phối một cách linh hoạt nhằm hỗ trợ nhiều loại kiến trúc nền tảng (Baseline ViT, CLIP, và DINOv2) ứng với các mức độ sẵn có của dữ liệu huấn luyện khác nhau.

### 2.1 Các Siêu Tham Số & Tiền Xử Lý Ảnh (Hyperparameters & Preprocessing)
Các tham số đánh giá toàn cục được cấu hình bên trong file `config/config_eval.py`, khớp hoàn toàn với các kỳ vọng tiền xử lý của các kiến trúc Foundation Model:

```python
# Cấu hình kiến trúc cốt lõi (Core Structural Configurations)
config.image_size = 224          # Kích thước không gian đầu vào (nén về 224x224 pixels)
config.batch_size_eval = 32      # Kích thước Batch tối ưu để phù hợp với GPU 6GB VRAM
config.normalize_type = "clip"   # Chuẩn hóa giá trị Mean/Std khớp với phân phối pretrained gốc
config.interpolation_type = 3    # Sử dụng thuật toán Bicubic để resize ảnh chất lượng cao

```
## 📄 Citation & License


This project is a fork based on the research and codebase developed by the **Fraunhofer Institute for Computer Graphics Research IGD Darmstadt**.

---
### Citation

```
@article{DBLP:journals/ivc/ChettaouiDB25,
  author       = {Tahar Chettaoui and
                  Naser Damer and
                  Fadi Boutros},
  title        = {FRoundation: Are foundation models ready for face recognition?},
  journal      = {Image Vis. Comput.},
  volume       = {156},
  pages        = {105453},
  year         = {2025}
}
```
## License

This project is licensed under the terms of the **Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.  
Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt.

For more details, please see the [CC BY-NC-SA 4.0 License Official Text](https://creativecommons.org/licenses/by-nc-sa/4.0/).