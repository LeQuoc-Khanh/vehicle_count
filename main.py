import cv2
import numpy as np
import time
import torch
from collections import defaultdict, deque
from ultralytics import YOLO


# ============================================================
# ĐỒ ÁN: HỆ THỐNG ĐẾM XE THÔNG MINH
# ============================================================
# Hệ thống này thuộc các kiến thức đã học trong môn
# Xử lý ảnh và Thị giác máy tính:
#
# 1. Chương 1 - Tổng quan thị giác máy tính:
#    - Ứng dụng giao thông thông minh.
#    - Camera thu nhận video, máy tính xử lý và đưa ra quyết định:
#      phát hiện xe, phân loại xe và đếm số lượng xe.
#
# 2. Chương 2 - Các thuật toán xử lý ảnh:
#    - Resize frame để chuẩn hóa kích thước đầu vào.
#    - Xử lý ảnh bằng OpenCV, thao tác trên ma trận ảnh.
#    - Tạo mask để giữ lại vùng ảnh cần xử lý.
#
# 3. Chương 3 - Phát hiện đặc trưng / hình học ảnh:
#    - Sử dụng tọa độ tâm đối tượng (cx, cy).
#    - Sử dụng đường thẳng đếm xe LINE_Y.
#    - Kiểm tra chuyển động của tâm xe qua đường thẳng.
#
# 4. Chương 4 - Phân đoạn ảnh:
#    - Sử dụng ROI mask để phân đoạn vùng đường cần quan sát.
#    - Loại bỏ các vùng nền không cần thiết như nhà, cây, trời, vỉa hè.
#
# 5. Chương 5 - Nhận dạng ảnh:
#    - Sử dụng YOLOv8 để nhận dạng và phân loại đối tượng:
#      car, motorcycle, bus.
#
# Lưu ý:
# YOLO là mô hình huấn luyện sẵn, nhưng trong đồ án này YOLO không phải
# là toàn bộ giải pháp. YOLO được tích hợp vào một pipeline CV gồm:
# đọc video -> tiền xử lý -> phân đoạn ROI -> nhận dạng -> tracking
# -> kiểm tra giao cắt đường đếm -> thống kê kết quả.
# ============================================================


# =========================
# 1. CẤU HÌNH HỆ THỐNG
# =========================

VIDEO_PATH = "video1.mp4"
OUTPUT_PATH = "output_counting.mp4"

FRAME_WIDTH = 1024
FRAME_HEIGHT = 768

CONF_THRES = 0.4
IOU_THRES = 0.45

# yolov8x.pt có độ chính xác cao nhưng chạy chậm.
# Nếu máy yếu có thể đổi thành yolov8m.pt hoặc yolov8n.pt.
MODEL_PATH = "yolov8x.pt"

# Các lớp phương tiện theo bộ dữ liệu COCO:
# 2: car, 3: motorcycle, 5: bus
# Đây là phần thuộc Chương 5 - Nhận dạng và phân loại đối tượng.
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus"
}

VEHICLE_COLORS = {
    "car": (0, 255, 0),
    "motorcycle": (255, 0, 0),
    "bus": (0, 255, 255)
}

# Đường đếm xe.
# Đây là ứng dụng của kiến thức hình học ảnh:
# biểu diễn đường thẳng trong hệ tọa độ pixel.
LINE_Y = 150

# Hướng đếm:
# "up" nghĩa là xe đi từ dưới lên trên, tức cy giảm dần.
# "down" nghĩa là xe đi từ trên xuống dưới, tức cy tăng dần.
COUNT_DIRECTION = "up"

# ROI - Region of Interest.
# Đây là vùng đường cần quan sát.
# Phần này thuộc Chương 4 - Phân đoạn ảnh.
# Ta dùng đa giác để giữ lại vùng đường và loại bỏ vùng nền.
ROI_CORNERS = np.array(
    [[
        (140, 6),
        (3, 180),
        (28, 762),
        (1019, 761),
        (1017, 308),
        (823, 7)
    ]],
    dtype=np.int32
)

MAX_HISTORY = 30
MAX_MISSING_FRAMES = 60


# =========================
# 2. KHỞI TẠO THIẾT BỊ VÀ MÔ HÌNH
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang chạy trên: {device}")

if torch.cuda.is_available():
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")

# Chương 5 - Nhận dạng ảnh:
# YOLOv8 được dùng để phát hiện và phân loại xe trong từng frame.
model = YOLO(MODEL_PATH).to(device)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Không mở được video: {VIDEO_PATH}")

fps_input = cap.get(cv2.CAP_PROP_FPS)
if fps_input <= 0:
    fps_input = 30

# Ghi video kết quả để phục vụ báo cáo và demo.
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps_input,
    (FRAME_WIDTH, FRAME_HEIGHT)
)

vehicle_count = {
    "car": 0,
    "motorcycle": 0,
    "bus": 0
}

counted_ids = set()

# Lưu lịch sử vị trí tâm của từng xe.
# Phần này phục vụ tracking và xác định hướng di chuyển.
track_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
last_seen = {}

prev_time = time.time()
frame_index = 0


# =========================
# 3. CÁC HÀM PHỤ TRỢ
# =========================

def crossed_line(prev_y, curr_y, line_y, direction="up"):
    """
    Kiểm tra đối tượng có cắt qua vạch đếm hay không.

    Kiến thức liên quan:
    - Chương 3: Hình học ảnh, tọa độ pixel, đường thẳng.
    - Ta xét vị trí tâm xe ở frame trước và frame hiện tại.
    - Nếu tâm xe đi từ một phía của đường LINE_Y sang phía còn lại,
      ta xem như xe đã đi qua vạch đếm.
    """

    if direction == "up":
        return prev_y > line_y and curr_y <= line_y

    if direction == "down":
        return prev_y < line_y and curr_y >= line_y

    return False


def point_in_roi(point, roi_corners):
    """
    Kiểm tra tâm đối tượng có nằm trong vùng ROI hay không.

    Kiến thức liên quan:
    - Chương 4: Phân đoạn ảnh.
    - ROI mask giúp hệ thống chỉ xử lý vùng đường,
      giảm nhiễu từ các vùng không liên quan.
    """

    return cv2.pointPolygonTest(roi_corners, point, False) >= 0


def draw_transparent_roi(frame, roi_corners, alpha=0.12):
    """
    Vẽ vùng ROI lên frame để minh họa vùng phân đoạn.

    Kiến thức liên quan:
    - Chương 4: Phân đoạn vùng ảnh có ý nghĩa.
    """

    overlay = frame.copy()
    cv2.fillPoly(overlay, roi_corners, (255, 0, 255))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def cleanup_old_tracks(current_frame, last_seen_dict, history_dict):
    """
    Xóa lịch sử của các ID đã biến mất quá lâu.

    Ý nghĩa:
    - Tránh chương trình dùng quá nhiều bộ nhớ khi video dài.
    - Giúp hệ thống ổn định hơn khi chạy thực tế.
    """

    old_ids = [
        track_id for track_id, last_frame in last_seen_dict.items()
        if current_frame - last_frame > MAX_MISSING_FRAMES
    ]

    for track_id in old_ids:
        last_seen_dict.pop(track_id, None)
        history_dict.pop(track_id, None)


# =========================
# 4. VÒNG LẶP XỬ LÝ VIDEO
# =========================

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_index += 1

    # ------------------------------------------------------------
    # Chương 2 - Tiền xử lý ảnh:
    # Resize frame về kích thước cố định.
    # Việc chuẩn hóa kích thước giúp tốc độ xử lý và hiển thị ổn định.
    # ------------------------------------------------------------
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # ------------------------------------------------------------
    # Chương 4 - Phân đoạn ảnh bằng ROI mask:
    # Tạo ảnh mask nhị phân cùng kích thước frame.
    # Vùng trong ROI có giá trị 255, vùng ngoài ROI có giá trị 0.
    # Sau đó dùng bitwise_and để chỉ giữ vùng đường cần quan sát.
    # ------------------------------------------------------------
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, ROI_CORNERS, 255)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # ------------------------------------------------------------
    # Chương 5 - Nhận dạng ảnh:
    # Sử dụng YOLOv8 để phát hiện và phân loại phương tiện.
    #
    # Tracking:
    # model.track kết hợp phát hiện đối tượng với theo dõi ID.
    # Mỗi xe được gán một track_id để tránh đếm trùng.
    # ------------------------------------------------------------
    results = model.track(
        masked_frame,
        persist=True,
        conf=CONF_THRES,
        iou=IOU_THRES,
        classes=list(VEHICLE_CLASSES.keys()),
        verbose=False,
        tracker="bytetrack.yaml",
        half=torch.cuda.is_available()
    )[0]

    # ------------------------------------------------------------
    # Xử lý kết quả nhận dạng và tracking
    # ------------------------------------------------------------
    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        clss = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):

            if cls_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)

            # ------------------------------------------------------------
            # Chương 3 - Hình học ảnh:
            # Tính tâm bounding box.
            # Tâm xe được dùng để theo dõi quỹ đạo và kiểm tra qua vạch.
            # ------------------------------------------------------------
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Chỉ xét các xe có tâm nằm trong ROI.
            if not point_in_roi((cx, cy), ROI_CORNERS):
                continue

            label = VEHICLE_CLASSES[cls_id]
            color = VEHICLE_COLORS.get(label, (255, 255, 255))

            history = track_history[track_id]

            # ------------------------------------------------------------
            # Tracking + đếm xe:
            # Nếu xe đã xuất hiện ở frame trước, so sánh cy cũ và cy mới.
            # Khi tâm xe cắt qua LINE_Y theo đúng hướng, tăng bộ đếm.
            # ------------------------------------------------------------
            if len(history) > 0:
                prev_cx, prev_cy = history[-1]

                if crossed_line(prev_cy, cy, LINE_Y, COUNT_DIRECTION):
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        vehicle_count[label] += 1

                        cv2.circle(frame, (cx, cy), 18, (0, 0, 255), -1)
                        cv2.putText(
                            frame,
                            "+1",
                            (cx + 15, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            3
                        )

            history.append((cx, cy))
            last_seen[track_id] = frame_index

            # Vẽ bounding box kết quả nhận dạng.
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                f"{label} ID:{track_id} {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

            # Vẽ tâm đối tượng.
            cv2.circle(frame, (cx, cy), 4, color, -1)

            # Vẽ quỹ đạo di chuyển của xe.
            points = list(history)
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], color, 2)

    cleanup_old_tracks(frame_index, last_seen, track_history)

    # ------------------------------------------------------------
    # Hiển thị ROI và vạch đếm
    # ------------------------------------------------------------
    draw_transparent_roi(frame, ROI_CORNERS)
    cv2.polylines(frame, ROI_CORNERS, isClosed=True, color=(255, 0, 255), thickness=2)

    cv2.line(frame, (0, LINE_Y), (FRAME_WIDTH, LINE_Y), (0, 255, 255), 2)

    direction_text = "Huong dem: duoi len tren" if COUNT_DIRECTION == "up" else "Huong dem: tren xuong duoi"

    cv2.putText(
        frame,
        direction_text,
        (10, LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # ------------------------------------------------------------
    # Vẽ bảng thống kê
    # ------------------------------------------------------------
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 260, 150

    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "THONG KE XE",
        (panel_x + 10, panel_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    y_text = panel_y + 60
    total = 0

    for v_class, count in vehicle_count.items():
        total += count
        txt_color = VEHICLE_COLORS.get(v_class, (255, 255, 255))

        cv2.putText(
            frame,
            f"{v_class}: {count}",
            (panel_x + 10, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            txt_color,
            2
        )

        y_text += 28

    cv2.putText(
        frame,
        f"Total: {total}",
        (panel_x + 10, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ------------------------------------------------------------
    # Tính FPS để đánh giá tốc độ xử lý.
    # Đây là chỉ số định lượng quan trọng khi demo hệ thống thời gian thực.
    # ------------------------------------------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if current_time > prev_time else 0
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (FRAME_WIDTH - 160, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        3
    )

    writer.write(frame)

    cv2.imshow("He thong dem xe thong minh", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


# =========================
# 5. GIẢI PHÓNG TÀI NGUYÊN
# =========================

cap.release()
writer.release()
cv2.destroyAllWindows()

print("Hoan tat xu ly video.")
print(f"Video ket qua da luu tai: {OUTPUT_PATH}")
print("Ket qua dem xe:")

for vehicle, count in vehicle_count.items():
    print(f"{vehicle}: {count}")

print(f"Tong cong: {sum(vehicle_count.values())}")