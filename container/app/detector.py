import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    def __init__(self, onnx_path, imgsz=640, conf=0.01, class_names=("cat",)):
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names
        self.input_name = self.session.get_inputs()[0].name

    def _letterbox(self, img: Image.Image):
        orig_w, orig_h = img.size
        scale = min(self.imgsz / orig_w, self.imgsz / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
        pad_x = (self.imgsz - new_w) / 2
        pad_y = (self.imgsz - new_h) / 2
        canvas.paste(img_resized, (int(pad_x), int(pad_y)))
        arr = np.array(canvas, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None, ...]
        return arr, scale, pad_x, pad_y

    def predict(self, image_path: str) -> list[dict]:
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        x, scale, pad_x, pad_y = self._letterbox(img)
        raw = self.session.run(None, {self.input_name: x})[0][0]  # (300, 6)

        results = []
        for x1, y1, x2, y2, score, cls in raw:
            if score < self.conf:
                continue
            x1 = float(np.clip((x1 - pad_x) / scale, 0, orig_w))
            y1 = float(np.clip((y1 - pad_y) / scale, 0, orig_h))
            x2 = float(np.clip((x2 - pad_x) / scale, 0, orig_w))
            y2 = float(np.clip((y2 - pad_y) / scale, 0, orig_h))
            results.append({
                "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                "confidence": float(score),
                "class": self.class_names[int(cls)],
            })
        return results
