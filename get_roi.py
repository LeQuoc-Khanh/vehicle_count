import cv2
import numpy as np

# Danh sách lưu trữ các điểm bạn click
points = []

# Hàm xử lý sự kiện click chuột
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Khi click chuột trái
        points.append((x, y))
        
        # Vẽ một chấm tròn nhỏ màu đỏ tại nơi vừa click để đánh dấu
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Lay Toa Do ROI", frame)
        print(f"Đã chọn điểm thứ {len(points)}: ({x}, {y})")

        # Đủ 4 điểm thì vẽ đường bao quanh và in kết quả ra màn hình
        if len(points) == 6:
            pts_array = np.array([points], dtype=np.int32)
            cv2.polylines(frame, pts_array, isClosed=True, color=(0, 255, 255), thickness=2)
            
            # Làm mờ phần ngoài để bạn xem trước kết quả
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, pts_array, 255)
            preview = cv2.bitwise_and(frame, frame, mask=mask)
            
            cv2.imshow("Lay Toa Do ROI", frame)
            cv2.imshow("Xem truoc Mask", preview)
            
            print("\n" + "="*50)
            print("🎉 ĐÃ LẤY ĐỦ 4 ĐIỂM. COPY ĐOẠN CODE DƯỚI ĐÂY VÀO FILE CHÍNH:")
            print(f"roi_corners = np.array([{points}], dtype=np.int32)")
            print("="*50 + "\n")

# Đọc frame đầu tiên của video
cap = cv2.VideoCapture('video1.mp4')
ret, frame = cap.read()

if not ret:
    print("Lỗi: Không thể đọc được video1.mp4. Hãy kiểm tra lại tên file.")
    exit()

# BẮT BUỘC: Phải resize kích thước giống y hệt code chính để tọa độ không bị lệch
frame = cv2.resize(frame, (1024, 768))

# Hiển thị cửa sổ và gắn hàm theo dõi chuột
cv2.imshow("Lay Toa Do ROI", frame)
cv2.setMouseCallback("Lay Toa Do ROI", get_coordinates)

print("HƯỚNG DẪN SỬ DỤNG:")
print("- Click chuột trái 4 điểm trên cửa sổ video để tạo thành khu vực mặt đường.")
print("- Lời khuyên: Click theo vòng tròn (Dưới-Trái -> Dưới-Phải -> Trên-Phải -> Trên-Trái).")
print("- Nhấn phím 'ESC' để thoát chương trình.\n")

# Chờ người dùng nhấn phím ESC để thoát
while True:
    if cv2.waitKey(1) & 0xFF == 27: # 27 là mã ASCII của phím ESC
        break

cap.release()
cv2.destroyAllWindows()