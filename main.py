import cv2
import numpy as np
from ultralytics import YOLO

# 1. Khởi tạo mô hình Nhận dạng (Chương 5) 
# YOLOv8n giúp nhận diện chính xác xe máy, ô tô, xe tải mà không cần đoán qua diện tích
model = YOLO('yolov8n.pt')

# Cấu hình video thực tế [cite: 16, 28]
cap = cv2.VideoCapture('video.mp4')

# Cấu hình Vùng đếm thông minh (Buffer Zone) để tránh "miss" xe
line_y = 450      
buffer_h = 20     # Dải an toàn 20 pixel để bắt các xe di chuyển nhanh
counted_ids = set() 
vehicle_count = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

# Mapping ID từ COCO dataset của YOLO sang tên loại xe
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Chương 2: Tiền xử lý - Resize để ổn định khung hình 
    frame = cv2.resize(frame, (1024, 768))

    # Chương 5: Nhận dạng đối tượng 
    # Ta sử dụng YOLO để lấy class_id thay vì dùng diện tích
    results = model(frame, conf=0.4, iou = 0.5, verbose=False)[0]

    for box in results.boxes:
        # Lấy thông tin tọa độ và loại xe
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id in VEHICLE_CLASSES:
            label = VEHICLE_CLASSES[cls_id]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Tâm của xe

            # LOGIC VÙNG ĐẾM: Kiểm tra xe đi vào dải pixel an toàn
            if (line_y - buffer_h) < cy < (line_y + buffer_h):
                # Lưu ý: Để đếm chính xác tuyệt đối trong pipeline này, 
                # bạn nên kết hợp thêm bộ theo dõi (Tracker) ở Chương 3.
                # Ở đây ta dùng một logic đơn giản để minh họa:
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Hiển thị thông báo khi xe chạm vạch
                cv2.putText(frame, "DETECTED", (x1, y1 - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Vẽ kết quả nhận diện lên màn hình
            color = (0, 255, 0) if label == 'car' else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Vẽ vạch đếm và Vùng đệm trực quan
    cv2.line(frame, (0, line_y), (1024, line_y), (0, 255, 255), 2)
    cv2.putText(frame, "COUNTING ZONE", (10, line_y - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('He thong dem xe thong minh - YOLOv8', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()