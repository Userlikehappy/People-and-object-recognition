from ultralytics import YOLOv10
import cv2
import os
from pathlib import Path

model = YOLOv10.from_pretrained('jameslahm/yolov10x')   
input_folder  = r"C:\Users\Admin\Videos\Captures\TRAIN\image"      
output_folder = r"C:\Users\Admin\Videos\Captures\TRAIN\image_VISUALIZED"  


os.makedirs(output_folder, exist_ok=True)
conf_threshold = 0.01
iou_threshold  = 0.85
max_det        = 300
imgsz          = 1280

PERSON_CONF = 0.10
CAR_CONF    = 0.15
MOTOR_CONF  = 0.08

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
    person = car = motor = 0

    for box in results.boxes:
        conf = box.conf.item()
        cls = int(box.cls.item())

        if (cls == 0 and conf >= PERSON_CONF) or \
           (cls == 2 and conf >= CAR_CONF) or \
           (cls == 3 and conf >= MOTOR_CONF):

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0:   # person
                color = (0, 255, 255)   # yellow
                label = f"person {conf:.2f}"
                person += 1
            elif cls == 2: # car
                color = (0, 120, 255)   # orange
                label = f"car {conf:.2f}"
                car += 1
            elif cls == 3: # motorcycle
                color = (255, 180, 0)   # blue
                label = f"motor {conf:.2f}"
                motor += 1
            else:
                continue

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)
            cv2.putText(image, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 4)

    summary = f"P:{person} | C:{car} | M:{motor}"
    cv2.putText(image, summary, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 7)

    cv2.imwrite(save_path, image)
    print(f"{Path(img_path).name:50} → P:{person:2d} | C:{car:2d} | M:{motor:3d}")

image_extensions = ('.jpg','.jpeg','.png','.bmp','.JPG','.JPEG','.PNG')

print("RUN")
print(f"Input  → {input_folder}")
print(f"Output → {output_folder}\n")

count = 0
for file in sorted(os.listdir(input_folder)):
    if file.lower().endswith(image_extensions):
        src = os.path.join(input_folder, file)
        dst = os.path.join(output_folder, file)
        process_single_image(src, dst)
        count += 1
        if count % 100 == 0:
            print(f"→ Đã xử lý {count}/3348 ảnh...")

print("\ndone!")
print("Open folder:")
print(output_folder)