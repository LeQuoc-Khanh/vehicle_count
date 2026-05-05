# 🚗 Hệ Thống Đếm Xe Thông Minh (Smart Vehicle Counting System)

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-white?style=for-the-badge&logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Deep_Learning-FF1493?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

Đồ án môn học **Xử lý ảnh và Thị giác máy tính (Mã HP: 121036)**[cite: 5].  
Dự án được phát triển bởi nhóm sinh viên chuyên ngành Kỹ thuật Phần mềm, trường Đại học Giao thông Vận tải TP.HCM (UTH), bao gồm: Vũ Trí Dũng, Thân Văn Ký, Hoàng Mạnh Đức, Nguyễn Thị Hoài Linh, Nguyễn Thành Nam, và Cao Xuân Quyết.

Hệ thống này được thiết kế để giải quyết bài toán quản lý lưu lượng giao thông thực tế: tự động phát hiện, theo dõi quỹ đạo và đếm số lượng các loại phương tiện (ô tô, xe máy, xe buýt) di chuyển qua một tuyến đường.

---

## 📑 Mục lục
- [1. Đặt vấn đề](#1-đặt-vấn-đề)
- [2. Phương pháp & Pipeline Xử lý](#2-phương-pháp--pipeline-xử-lý)
- [3. Dữ liệu thực nghiệm (Dataset)](#3-dữ-liệu-thực-nghiệm-dataset)
- [4. Cấu trúc thư mục](#4-cấu-trúc-thư-mục)
- [5. Hướng dẫn Cài đặt & Chạy dự án](#5-hướng-dẫn-cài-đặt--chạy-dự-án)
- [6. Kết quả đầu ra](#6-kết-quả-đầu-ra)

---

## 1. Đặt vấn đề
Việc đo lường lưu lượng giao thông thủ công rất tốn kém và dễ sai sót. Dự án này xây dựng một hệ thống đếm xe thông minh bằng camera giám sát, giúp thu thập số liệu phương tiện một cách tự động để phục vụ cho các ứng dụng giao thông thông minh (Smart City).

---

## 2. Phương pháp & Pipeline Xử lý
Hệ thống không chỉ sử dụng AI như một "hộp đen" mà áp dụng một Pipeline tích hợp đầy đủ các kỹ thuật Computer Vision nền tảng (theo đúng chuẩn chương trình học)[cite: 2, 5]:

* **Chương 2 - Tiền xử lý ảnh:**
  * Chuẩn hóa kích thước khung hình (`cv2.resize`) để tối ưu hóa tốc độ tính toán.
  * Áp dụng bộ lọc `cv2.GaussianBlur` để giảm nhiễu (áp dụng trong khâu tiền xử lý dữ liệu)[cite: 3].

* **Chương 4 - Phân đoạn ảnh (Image Segmentation):**
  * Tạo mặt nạ nhị phân đa giác (ROI Mask) sử dụng `cv2.fillPoly` và toán tử `cv2.bitwise_and`. Kỹ thuật này giúp hệ thống chỉ tập trung vào vùng mặt đường, loại bỏ hoàn toàn nhiễu từ bối cảnh (nhà cửa, cây cối, vỉa hè)[cite: 2].

* **Chương 5 - Nhận dạng & Theo dõi (Object Detection & Tracking):**
  * Sử dụng mạng nơ-ron tích chập YOLOv8 kết hợp thuật toán ByteTrack để phát hiện xe (`car`, `motorcycle`, `bus`) và cấp phát ID độc nhất (Track ID) cho mỗi phương tiện để tránh đếm trùng[cite: 2].

* **Chương 3 - Hình học ảnh (Image Geometry & Logic đếm):**
  * Trích xuất tâm bouding box `(cx, cy)` và lưu lại quỹ đạo di chuyển[cite: 2].
  * Sử dụng toán học không gian 2D để kiểm tra giao cắt: So sánh vị trí tâm xe giữa 2 frame liên tiếp với phương trình đường thẳng vạch đếm (`LINE_Y`). Nếu quỹ đạo cắt qua vạch đúng hướng (Bottom-Up hoặc Top-Down), hệ thống sẽ tăng bộ đếm[cite: 2].

---

## 3. Dữ liệu thực nghiệm (Dataset)

* **Video Test:** Được quay/thu thập từ giao thông thực tế tại Việt Nam (`video.mp4`, `video1.mp4`, `video2.mp4`).

* **Dataset Huấn luyện:** Hệ thống tham khảo và có thể Fine-tune thêm bằng bộ dữ liệu `vietnam - vdataset linh-gia`[cite: 4]. Bộ dữ liệu gồm 565 ảnh giao thông Việt Nam, đã được gán nhãn (annotate) chuẩn format YOLOv8, lấy từ nền tảng Roboflow[cite: 4].

---

## 4. Cấu trúc thư mục

```text
VEHICLE_COUNT/
│
├── dataset/
│   └── thuthap_xulyDL/
│       ├── anh_da_xu_ly/         # Ảnh sau khi chạy Gaussian Blur giảm nhiễu
│       ├── train/                # Dữ liệu gốc ảnh giao thông
│       ├── data.yaml
│       └── README.roboflow.txt   # Nguồn dataset từ Roboflow
│
├── main.py                       # Pipeline xử lý trung tâm (Nhận diện & Đếm)
├── get_roi.py                    # Tool tương tác click chuột lấy toạ độ ROI
├── xuly_anh.ipynb                # Code Jupyter tiền xử lý, giảm nhiễu ảnh
├── yolov8*.pt                    # Trọng số mô hình YOLO (n, s, m, l, x)
├── result.csv                    # File báo cáo số liệu thống kê cuối cùng
├── video*.mp4                    # Các video giao thông thực tế đầu vào
└── output_counting.mp4           # Video kết quả đầu ra (có vẽ bounding box, vạch đếm)
```

---

## 5. Hướng dẫn Cài đặt & Chạy dự án

### 5.1. Cài đặt môi trường

Đảm bảo máy bạn đã cài đặt Python (>= 3.8). Mở Terminal và chạy lệnh sau để tải các thư viện cần thiết:

```bash
pip install ultralytics torch torchvision opencv-python numpy pandas lapx
```

(Lưu ý: Thư viện `lapx` được yêu cầu để bộ tracker ByteTrack hoạt động ổn định trên Windows).

---

### 5.2. Chạy hệ thống

**Bước 1: Lấy tọa độ vùng quan tâm (ROI)** (Tùy chọn nếu đổi video góc khác)

```bash
python get_roi.py
```

Click chuột trái theo thứ tự 4 điểm trên video để vẽ khung mặt đường.  
Nhấn ESC để thoát. Copy mảng tọa độ in ra trên Terminal dán vào biến `ROI_CORNERS` trong file `main.py`.

---

**Bước 2: Khởi động hệ thống nhận diện và đếm**

```bash
python main.py
```

Hệ thống sẽ bật cửa sổ hiển thị video thời gian thực, vẽ bouding box, vạch đường `LINE_Y`, vùng ROI màu tím nhạt và quỹ đạo xe[cite: 2].

Bảng thống kê UI góc trái sẽ đếm số lượng từng loại xe liên tục[cite: 2].

Nhấn ESC nếu muốn dừng ngang[cite: 2].

---

## 6. Kết quả đầu ra

Sau khi hoàn tất tiến trình, hệ thống sẽ tự động sinh ra:

* **Video Demo (`output_counting.mp4`)**: Bản render video đã vẽ đè đồ họa trực quan[cite: 2].

* **File báo cáo định lượng (`result.csv`)**: Thống kê số lượng theo từng lớp xe (Car, Motorcycle, Bus) và tổng cộng, đáp ứng đúng tiêu chí đánh giá bằng số liệu của đồ án[cite: 1].