import cv2
import os

video_path = r"C:\Users\Admin\Videos\Captures\TEST.MOV"
output_folder = r"C:\Users\Admin\Videos\Captures\Frames"

# Tạo thư mục lưu ảnh nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Mở video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # Lấy fps gốc của video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Tính khoảng nhảy frame để có 2900 ảnh
desired_frames = 5900
step = max(int(total_frames / desired_frames), 1)

count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if count % step == 0:
        frame_name = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(frame_name, frame)
        saved_count += 1
    count += 1

cap.release()
print(f"Đã lưu {saved_count} ảnh vào {output_folder}")
