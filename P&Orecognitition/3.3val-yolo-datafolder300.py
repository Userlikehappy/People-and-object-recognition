# FINAL_VIETNAM_GOD_MODE.py  ← Đặt tên này cho xứng đáng
from ultralytics import YOLOv10
import cv2
import os
from pathlib import Path

# DÙNG MODEL MẠNH NHẤT CÓ THỂ (x thay vì m) → bắt người + xe máy cực chuẩn
model = YOLOv10.from_pretrained('jameslahm/yolov10x')  # x = cực mạnh!

input_folder  = r"C:\Users\Admin\Videos\Captures\VAL\image"
output_folder = r"C:\Users\Admin\Videos\Captures\VAL\image_DETECTED_FINAL"
os.makedirs(output_folder, exist_ok=True)

# THAM SỐ "THẦN THÁNH" CHO VIỆT NAM 2025
conf_threshold = 0.01      # CỰC THẤP → không bỏ sót cái nào
iou_threshold  = 0.85      # CỰC CAO → giữ box chồng lấn
max_det        = 300       # cho cảnh 100+ xe máy cũng không sợ
imgsz          = 1280      # độ phân giải cao nhất → chi tiết kinh hoàng

# Ngưỡng lọc cuối (vẫn giữ để loại rác)
CAR_CONF    = 0.15
MOTOR_CONF  = 0.08
PERSON_CONF = 0.10         # thêm person!

def process_single_image(img_path, save_path):
    results = model.predict(
        source=img_path,
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=max_det,
        imgsz=imgsz,
        agnostic_nms=False,
        retina_masks=True,
        save=False,
        verbose=False
    )[0]

    image = cv2.imread(img_path)
    car = motor = person = 0

    for box in results.boxes:
        conf = box.conf.item()
        cls = int(box.cls.item())

        if (cls == 0 and conf >= PERSON_CONF) or \
           (cls == 2 and conf >= CAR_CONF) or \
           (cls == 3 and conf >= MOTOR_CONF):

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[cls]

            if cls == 0:   # person
                color = (0, 255, 255)   # vàng nổi bật
                person += 1
                label = f"person {conf:.2f}"
            elif cls == 2: # car
                color = (0, 120, 255)   # cam
                car += 1
                label = f"car {conf:.2f}"
            elif cls == 3: # motorcycle
                color = (255, 180, 0)   # xanh dương
                motor += 1
                label = f"motor {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)
            cv2.putText(image, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 4)

    # Tổng kết to đùng
    summary = f"Person: {person} | Car: {car} | Motor: {motor}"
    cv2.putText(image, summary, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 0), 7)

    cv2.imwrite(save_path, image)
    print(f"{Path(img_path).name:50} → P:{person:2d} | C:{car:2d} | M:{motor:3d}  BẮT HẾT!")

# ================== CHẠY TOÀN BỘ ==================
image_extensions = ('.jpg','.jpeg','.png','.bmp','.JPG','.JPEG')

print("BẮT ĐẦU CHẾ ĐỘ THẦN THÁNH – BẮT HẾT XE MÁY + NGƯỜI MẶC ÁO MƯA")
for file in sorted(os.listdir(input_folder)):
    if file.lower().endswith(image_extensions):
        src = os.path.join(input_folder, file)
        dst = os.path.join(output_folder, file)
        process_single_image(src, dst)

print("\nXONG! MỞ THƯ MỤC NÀY ĐỂ XEM KẾT QUẢ KINH HOÀNG:")
print(output_folder)