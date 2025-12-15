# file: show_confusion_matrix.py
from ultralytics import YOLOv10

# ĐƯỜNG DẪN ĐẾN best.pt CỦA BẠN (sửa lại cho đúng)
MODEL_PATH = r"runs/yolov10_vietnam/YOLOv10m_VietNam_Traffic_FAST/weights/best.pt"

# Load model
model = YOLOv10(MODEL_PATH)

# CHẠY LỆNH NÀY LÀ RA LUÔN CONFUSION MATRIX + TẤT CẢ BIỂU ĐỒ
model.val(
    data=r"C:/Users/Admin/Videos/Captures/dataset/data.yaml",
    imgsz=640,
    batch=16,
    device='',                    # CPU
    plots=True,                   # QUAN TRỌNG: bật vẽ biểu đồ
    save=True,                    # lưu ảnh
    name="CONFUSION_MATRIX_FINAL",   # tên thư mục kết quả
    project="evaluation_results",    # thư mục cha
    exist_ok=True,
    conf=0.001,                   # để lấy đủ điểm cho PR curve mượt
    iou=0.6
)

print("HOÀN TẤT! Mở thư mục này để xem Confusion Matrix:")
print(r"evaluation_results\CONFUSION_MATRIX_FINAL")

