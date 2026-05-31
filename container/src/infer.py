import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def letterbox(img: Image.Image, size: int = 640):
    orig_w, orig_h = img.size
    scale = min(size / orig_w, size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    canvas.paste(img_resized, (int(pad_x), int(pad_y)))
    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr, scale, pad_x, pad_y


def run(args):
    input_dir = Path(args.input)
    output_path = Path(args.output)
    threshold = args.threshold

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import subprocess
        subprocess.run(["chmod", "777", str(output_path.parent)], capture_output=True)
    except Exception:
        pass

    model_path = Path("/app/model/cat_yolo26.onnx")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    img_exts = {".jpg", ".jpeg", ".png"}
    img_paths = sorted(
        (p for p in input_dir.rglob("*") if p.suffix.lower() in img_exts),
        key=lambda p: p.name,
    )

    predictions = []
    for img_path in img_paths:
        detections = []
        try:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            x, scale, pad_x, pad_y = letterbox(img, 640)
            raw = session.run(None, {input_name: x})[0][0]  # (300, 6)
            for x1, y1, x2, y2, score, cls in raw:
                if score < threshold:
                    continue
                x1 = float(np.clip((x1 - pad_x) / scale, 0, orig_w))
                y1 = float(np.clip((y1 - pad_y) / scale, 0, orig_h))
                x2 = float(np.clip((x2 - pad_x) / scale, 0, orig_w))
                y2 = float(np.clip((y2 - pad_y) / scale, 0, orig_h))
                detections.append({
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "score": round(float(score), 4),
                    "label": "cat",
                })
        except Exception:
            pass

        predictions.append({"image": img_path.name, "detections": detections})

    result = {
        "model": "yolo26-cat-onnx",
        "threshold": threshold,
        "predictions": predictions,
    }

    output_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {len(predictions)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    run(args)
    sys.exit(0)


if __name__ == "__main__":
    main()
