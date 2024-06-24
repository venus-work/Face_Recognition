import cv2
import os
import cv2
import warnings
from src.anti_spoof_predict import AntiSpoofPredict
# Mở camera
cap = cv2.VideoCapture(1)

# Nhập ID từ người dùng
user_id = input("Nhập ID của người dùng: ")

# Tạo thư mục để lưu ảnh
output_dir = f"data/test_images/{user_id}"
os.makedirs(output_dir, exist_ok=True)

# Biến đếm để chỉ lưu 300 ảnh
image_count = 0

while image_count < 30:
    isSuccess, frame = cap.read()
    model_test = AntiSpoofPredict(0)
    if isSuccess:
        image_bbox = model_test.get_bbox(frame)
        if image_bbox is not None:
            x, y, w, h = (image_bbox[0]), (image_bbox[1] - 50), (image_bbox[0] + image_bbox[2]), (image_bbox[1] + image_bbox[3])

            cropped_face = frame[y:h, x:w]
            if cropped_face is not None and cropped_face.size != 0:
                cropped_face = cv2.resize(cropped_face, (160, 160))

                # Lưu ảnh khuôn mặt vào thư mục
                image_filename = os.path.join(output_dir, f"{user_id}_{image_count}.jpg")
                cv2.imwrite(image_filename, cropped_face)
                image_count += 1
        cv2.imshow('Face Detection', frame)

        # Thoát khỏi vòng lặp nếu nhấn phím Esc hoặc đủ 300 ảnh
        if cv2.waitKey(1) & 0xFF == 27 or image_count >= 300:
            break

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
