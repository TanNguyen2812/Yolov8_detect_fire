from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    model = YOLO('best.pt')
    # model.val()

    cap = cv2.VideoCapture('test_video.mp4')
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        out = model.predict(frame)
        image_res = out[0].plot(pil=True, conf=True)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        cv2.imshow('res', image_res)
