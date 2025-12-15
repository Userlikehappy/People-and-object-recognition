# train_and_get_confusion_matrix.py
from ultralytics import YOLOv10

def main():
    # 1. Load model nhỏ nhanh (n = nano, rất hợp CPU)
    model = YOLOv10.from_pretrained('jameslahm/yolov10n')   # hoặc yolov10m nếu muốn mạnh hơn

    # 2. Đường dẫn data.yaml của bạn
    data_path = r"C:/Users/Admin/Videos/Captures/dataset/data.yaml"

    # 3. CHỈ CẦN 1 DÒNG DUY NHẤT này → ra hết mọi thứ!
    results = model.train(
        data=data_path,
        epochs=100,      # để nhiều, nó tự dừng nhờ patience
        imgsz=640,
        batch=8,         # CPU i7-8700 chịu được
        device='',       # CPU
        project='runs/my_vietnam_traffic',
        name='best_model_final',
        patience=20,     # tự dừng nếu không cải thiện
        exist_ok=True
    )

    print("HOÀN TẤT! Mở thư mục này để xem confusion matrix + mọi thứ:")
    print(r"runs\my_vietnam_traffic\best_model_final")

if __name__ == '__main__':
    main()