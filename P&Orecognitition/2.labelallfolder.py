# auto_create_labels_full.py
from ultralytics import YOLOv10
import os
from tqdm import tqdm

model = YOLOv10.from_pretrained('jameslahm/yolov10m')

train_folder = r"C:\Users\Admin\Videos\Captures\TRAIN"   # 6011 ảnh
val_folder   = r"C:\Users\Admin\Videos\Captures\VAL"     # 1805 ảnh
test_folder  = r"C:\Users\Admin\Videos\Captures\TEST"    # 900 ảnh

label_root = r"C:\Users\Admin\Videos\Captures\labels"
os.makedirs(os.path.join(label_root, "train"), exist_ok=True)
os.makedirs(os.path.join(label_root, "val"),   exist_ok=True)
os.makedirs(os.path.join(label_root, "test"),  exist_ok=True)

predict_params = {
    "conf": 0.10,
    "iou": 0.4,
    "max_det": 50,
    "imgsz": 800,
    "verbose": False
}

def create_labels(img_folder, split_name, expected_count):
    files = [f for f in os.listdir(img_folder) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"\nĐang xử lý {split_name.upper()}: {len(files)} ảnh (dự kiến ~{expected_count})")
    
    for img_name in tqdm(files, desc=split_name):
        img_path = os.path.join(img_folder, img_name)
        results = model.predict(source=img_path, **predict_params)[0]
        
        h, w = results.orig_shape
        label_path = os.path.join(label_root, split_name, 
                                  os.path.splitext(img_name)[0] + ".txt")
        
        with open(label_path, 'w', encoding='utf-8') as f:
            for box in results.boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                
                # Chính xác quy tắc lọc của bạn
                if (cls == 2 and conf >= 0.25) or (cls == 3 and conf >= 0.15):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    
                    yolo_class = 0 if cls == 2 else 1   # car=0, motorcycle=1
                    f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Hoàn tất {split_name}: {len(files)} file .txt đã tạo!")

create_labels(train_folder, "train", 6011)
create_labels(val_folder,   "val",   1805)
create_labels(test_folder,  "test",   900)

print("\n" + "="*60)
print("HOÀN TẤT 100% – ĐÃ TẠO LABEL CHO 8716 ẢNH!")
print(f"→ labels/train : ~6011 files")
print(f"→ labels/val   : ~1805 files")
print(f"→ labels/test  : ~900 files")
print("="*60)