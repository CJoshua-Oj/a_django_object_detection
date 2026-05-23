import cv2
import torch
import threading

class YOLOWebcam:
    def __init__(self):
        self.model = None
        self.camera = None
        self.lock = threading.Lock()

    def load_model(self):
        if self.model is None:
            self.model = torch.hub.load(
                'ultralytics/yolov5',
                'yolov5s',
                pretrained=True
            )
            self.model.conf = 0.40
            self.model.iou = 0.45
        return self.model

    def open_camera(self):
        if self.camera is None or not self.camera.isOpened():
            # CAP_DSHOW works well on Windows/VS Code. Fallback is included.
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(0)

            if not self.camera.isOpened():
                raise RuntimeError(
                    "Webcam could not be opened. Close other camera apps and check camera permission."
                )

            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        return self.camera

    def get_frame(self):
        with self.lock:
            model = self.load_model()
            camera = self.open_camera()

            success, frame = camera.read()
            if not success:
                return None

            results = model(frame)
            detected_frame = results.render()[0]

            success, buffer = cv2.imencode('.jpg', detected_frame)
            if not success:
                return None

            return buffer.tobytes()

webcam = YOLOWebcam()

def generate_frames():
    while True:
        frame = webcam.get_frame()

        if frame is None:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )
