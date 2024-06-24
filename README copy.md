# Face Detection and Anti-spoofing with PyTorch, MTCNN, OpenCV

## Giới thiệu
Dự án này giới thiệu cách sử dụng PyTorch, MTCNN (Multi-task Cascaded Convolutional Networks) và OpenCV để thực hiện nhận dạng khuôn mặt và chống gian lận (Anti-spoofing). Dự án được chia thành ba bước chính: cài đặt thư viện, chụp hình và thêm dữ liệu khuôn mặt vào mô hình, sau đó nhận dạng khuôn mặt và chống gian lận.

## Bước 1: Cài đặt thư viện
Trước hết, bạn cần cài đặt các thư viện cần thiết. Chạy lệnh sau để cài đặt chúng:

```bash
pip install -r requirements.txt
# Các thư viện khác nếu cần
```

## Bước 2: Chụp hình và Thêm dữ liệu khuôn mặt vào mô hình
* Sử dụng tệp **capture.py** chụp hình khuôn mặt. Bạn cần nhập một ID từ bàn phím để gán với hình ảnh được chụp.
```bash
python capture.py
```
* Sử dụng tệp  **traindata.py**  để thêm dữ liệu khuôn mặt đã chụp vào mô hình. Điều này sẽ giúp mô hình học được cách nhận dạng khuôn mặt của các người dùng.
```bash
python traindata.py
```

## Bước 3: Nhận dạng khuôn mặt và Chống gian lận
* Sử dụng tệp recognition.py để nhận dạng khuôn mặt và thực hiện chống gian lận (Anti-spoofing).
```bash
python recognition.py
```
Cài đặt thêm
Nếu bạn cần cài đặt các thư viện bổ sung hoặc cấu hình chi tiết, hãy tham khảo tệp requirements.txt và tài liệu hướng dẫn của từng thư viện.

## Tài liệu tham khảo
* [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/master)
* PyTorch
* MTCNN
* OpenCV
## Tác giả
* Dự án này được phát triển bởi **Ngô Hoàng Phúc.**



## Liên hệ
* Nếu bạn có bất kỳ câu hỏi hoặc đề xuất, vui lòng liên hệ với chúng tôi tại *ngohoangphuc.work@gmai.com*
