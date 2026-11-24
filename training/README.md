# Hướng dẫn Huấn luyện QLoRA trên Kaggle Free (Tesla T4 16GB)

Tài liệu hướng dẫn triển khai huấn luyện mô hình **Local CAD Worker (`cad-worker-7b`)** bằng Unsloth trên Kaggle Free GPU.

---

## 1. Chuẩn bị Dữ liệu
1. Sinh file dữ liệu đầy đủ 2.000 mẫu tại máy local:
   ```powershell
   .\.venv\Scripts\python.exe training/synth_cad_generator.py --count 2000
   ```
2. Đăng nhập [Kaggle](https://www.kaggle.com), vào mục **Datasets** -> **New Dataset**.
3. Tải file `training/dataset_sft.jsonl` lên và đặt tên dataset là `cad-sft-dataset`.

---

## 2. Tạo Kaggle Notebook
1. Tạo một Notebook mới trên Kaggle:
   - **Accelerator**: Chọn **GPU T4 x 2** hoặc **GPU T4** (Free tier 30h/tuần).
   - **Persistence**: Variables and Files.
   - **Internet**: Bật **On**.
2. Thêm Dataset vừa tải lên (`cad-sft-dataset`) vào Notebook (Input data).

---

## 3. Cài đặt Unsloth & Huấn luyện
Trong ô code đầu tiên của Kaggle Notebook, chạy lệnh cài đặt:

```bash
%%bash
pip install --no-deps unsloth "xformers<0.0.28" "trl<0.9.0" peft accelerate bitsandbytes
```

Trong ô code thứ hai, dán nội dung từ file [`training/train_qlora_kaggle.py`](train_qlora_kaggle.py) (hoặc tải file lên notebook và chạy):

```bash
python train_qlora_kaggle.py
```

* Thời gian huấn luyện ước tính: ~35–45 phút cho 2 epoch trên 2.000 mẫu.
* VRAM tiêu thụ: ~5.8 GB / 16 GB (hoàn toàn vừa vặn trong VRAM của Tesla T4).

---

## 4. Xuất file GGUF & Đưa về Ollama Local
Sau khi quá trình train hoàn tất, script tự động merge trọng số LoRA và lượng tử hóa ra file `cad-worker-7b-q4_k_m.gguf` (~4.5 GB) trong thư mục output của Kaggle.

1. Tải file `cad-worker-7b-q4_k_m.gguf` về thư mục `training/` tại máy tính của bạn.
2. Mở PowerShell tại thư mục gốc dự án và chạy lệnh tạo model:
   ```powershell
   ollama create cad-worker-7b -f training/Modelfile
   ```
3. Kiểm tra mô hình mới:
   ```powershell
   ollama run cad-worker-7b "Design a slotted base plate length 120, width 80, height 15 with 4 holes."
   ```
Mô hình sẽ trực tiếp sinh mã CadQuery chuẩn xác, kín nước và khóa gốc tọa độ $Z_{\min} = 0.000$ mm.

## Bộ dữ liệu đánh giá V3

Tách bộ đánh giá theo họ hình học và nguồn mẫu trước khi sinh biến thể kích thước hay diễn đạt lại prompt. Ưu tiên nón, hộp rỗng, trụ vát đơn và gân đơn. Các ca loft rỗng, sweep 3D và lỗ hướng tâm dùng làm bộ đánh giá riêng cho quyết định mở rộng tuyến SFT.

Với mỗi ca đánh giá, lưu kích thước và thể tích kỳ vọng từ đặc tả, cùng vùng rỗng hoặc lỗ cần thông. Chạy mã sinh qua cùng bộ kiểm định production. Theo dõi Pass@1, số mẫu đạt giả, P50/P95 và số token Ollama theo từng họ. Mốc để cân nhắc đưa thêm một họ vào SFT là Pass@1 từ 90%, không có ca đạt giả trong bộ đánh giá và P95 dưới 10 giây trên máy triển khai.

Với các họ có thể dựng từ một bộ tham số ngắn, thử đích JSON trước khi huấn luyện sinh Python. Đánh giá nón bằng các biến thể đường kính/bán kính, nón nhọn và nón cụt, cùng ca đảo đáy và đỉnh. Ghi thời gian tải model và thời gian sinh riêng khi đo lượt đầu và lượt model đã tải. Chỉ so sánh model nhẹ hơn khi dùng cùng hợp đồng hình học và cùng bộ đề giữ lại.
