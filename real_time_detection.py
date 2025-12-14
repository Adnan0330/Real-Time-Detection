"""
Real-Time Object Detection (YOLO) demo

Usage examples:
  python real_time_detection.py --source 0
  python real_time_detection.py --source video.mp4 --weights yolov8n.pt --conf 0.3

Supports webcam (integer index) or video file path. Uses `ultralytics` YOLO interface.
"""
import argparse
import time
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError('ultralytics package not found. Install with: pip install ultralytics')


def draw_boxes(frame, results, model):
    h, w = frame.shape[:2]
    for r in results.boxes:
        # xyxy format
        xyxy = r.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        conf = float(r.conf[0].cpu().numpy()) if hasattr(r, 'conf') else 0.0
        cls = int(r.cls[0].cpu().numpy()) if hasattr(r, 'cls') else -1
        label = model.names[cls] if (cls >= 0 and cls in model.names) else str(cls)

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def run(source=0, weights='yolov8n.pt', conf=0.25, device=''):
    # Load model
    model = YOLO(weights)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open source {source}")
        return

    fps_avg = 0.0
    frame_count = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t1 = time.time()

        # Run detection (ultralytics returns a Results object list)
        results = model(frame, imgsz=640, conf=conf, device=device)

        # results may be a list with one element for the image
        res = results[0]

        if len(res.boxes) > 0:
            draw_boxes(frame, res, model)

        # FPS calculation
        t2 = time.time()
        dt = t2 - t1
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_avg = (fps_avg * (frame_count - 1) + fps) / frame_count

        cv2.putText(frame, f'FPS: {fps:.1f} (avg {fps_avg:.1f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow('Real-Time Object Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='Real-Time Object Detection with YOLO')
    parser.add_argument('--source', '-s', default='0', help='Video source: webcam index or path to video file')
    parser.add_argument('--weights', '-w', default='yolov8n.pt', help='Path to YOLO weights file')
    parser.add_argument('--conf', '-c', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', '-d', default='', help='Device for inference, e.g. cpu or 0 (GPU)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    src = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    run(source=src, weights=args.weights, conf=args.conf, device=args.device)
