import sys
import json
import csv
from pathlib import Path

STUDENT_PATH = Path("/app/STUDENT.json")
MODEL_PATH   = Path("/app/models/best.onnx")
INPUT_DIR    = Path("/data/input")
OUTPUT_CSV   = Path("/data/output/predictions.csv")
IMG_EXTS     = {".jpg", ".jpeg", ".png"}


def cmd_info():
    print(STUDENT_PATH.read_text())


def cmd_predict():
    from app.detector import CatDetector

    detector = CatDetector(MODEL_PATH, imgsz=640, conf=0.01)

    img_paths = sorted(
        p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"])

        for img_path in img_paths:
            rel = img_path.relative_to(INPUT_DIR).as_posix()
            try:
                boxes = detector.predict(str(img_path))
            except Exception:
                boxes = []

            if boxes:
                for b in boxes:
                    writer.writerow([
                        rel,
                        round(b["xmin"], 2), round(b["ymin"], 2),
                        round(b["xmax"], 2), round(b["ymax"], 2),
                        round(b["confidence"], 4),
                        b["class"],
                    ])
            else:
                writer.writerow([rel, "", "", "", "", "", ""])

    print(f"predictions.csv written — {len(img_paths)} images processed.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "info":
        cmd_info()
    elif cmd == "predict":
        cmd_predict()
    else:
        print("Usage: python /app/app/cli.py [info|predict]", file=sys.stderr)
        sys.exit(1)
