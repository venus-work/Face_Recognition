import os
import cv2
import numpy as np
import argparse
import time
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from torchvision import transforms



frame_size = (640, 480)
SAMPLE_IMAGE_PATH = "./images/sample/"
DATA_PATH = './data'
power = pow(10, 6)
model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(0)
model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
frame_size = (1000, 1000)
DATA_PATH = './data'
mtcnn = MTCNN( keep_all=True, device=device)

def trans(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img)


def load_faceslist():
    if device == 'cpu':
        embeds = torch.load(DATA_PATH + '/faceslistCPU.pth')
    else:
        embeds = torch.load(DATA_PATH + '/faceslist.pth')
    names = np.load(DATA_PATH + '/usernames.npy')
    return embeds, names

embeddings, names = load_faceslist()

def inference(model, face, local_embeds, threshold=3):
    embeds = []
    embeds.append(model(trans(face).to(device).unsqueeze(0)))
    detect_embeds = torch.cat(embeds)  # [1,512]
    norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1)  # (1,n), moi cot la tong khoang cach euclide so vs embed moi
    min_dist, embed_idx = torch.min(norm_score, dim=1)
    if min_dist * power > threshold:
        return -1, -1
    else:
        return embed_idx, min_dist.double()


def extract_face(box, img, margin=20):
    face_size = 160
    img_size = frame_size
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ]
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])),
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ]
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img, (face_size, face_size), interpolation=cv2.INTER_AREA)
    face = Image.fromarray(face)
    return face

def Recognition(model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    capture = cv2.VideoCapture(1)# chỉnh thành 0 nếu không chạy được (STT camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    image_cropper = CropImage()
    while True:
        isSuccess, frame = capture.read()
        if isSuccess:
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int, box.tolist()))
                    image_bbox = bbox
                    if (image_bbox) is not None:
                        prediction = np.zeros((1, 3))
                        test_speed = 0
                        for model_name in os.listdir(model_dir):
                            h_input, w_input, model_type, scale = parse_model_name(model_name)
                            param = {
                                "org_img": frame,
                                "bbox": image_bbox,
                                "scale": scale,
                                "out_w": w_input,
                                "out_h": h_input,
                                "crop": True,
                            }
                            if scale is None:
                                param["crop"] = False
                            img = image_cropper.crop(**param)
                            start = time.time()
                            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
                            test_speed += time.time() - start
                    label = np.argmax(prediction)
                    value = prediction[0][label] / 2
                    if label == 1 and value > 0.8:
                        face2 = extract_face(bbox, frame)
                        idx, score = inference(model, face2, embeddings)
                        if idx != -1:
                            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
                            score = torch.Tensor.cpu(score[0]).detach().numpy() * power
                            frame = cv2.putText(frame, names[idx] + '_{:.2f}'.format(score), (bbox[0], bbox[1]),
                                                cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2, cv2.LINE_8)
                        else:
                            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
                            frame = cv2.putText(frame, 'Unknown', (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2,
                                                (0, 255, 0),
                                                2, cv2.LINE_8)
                    else:
                        frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
                        frame = cv2.putText(frame, 'Fake', (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0),
                                            2, cv2.LINE_8)
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    args = parser.parse_args()
    Recognition(args.model_dir, args.device_id)
