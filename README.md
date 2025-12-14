# Real-Time-Detection
""Real-time object detection project using Python and OpenCV""
"""
Real-time object detection (PyTorch + OpenCV) using YOLOv5 (torch.hub).

Usage examples (PowerShell):
  python real_time_detection_pytorch.py --source 0
  python real_time_detection_pytorch.py --source video.mp4 --weights yolov5s.pt --conf 0.35 --save out.mp4
  python real_time_detection_pytorch.py --source 0 --device cpu

Arguments:
  --source    : webcam index (0,1,...) or path to video file (default 0)
  --weights   : path to weights file or model name for torch.hub (default 'yolov5s')
  --conf      : confidence threshold (default 0.25)
  --iou       : IoU threshold for NMS (default 0.45)
  --device    : device string for PyTorch, e.g. '' for auto, 'cpu', or '0' for GPU 0
  --save      : optional path to save result video
  --view-size : resize display width (maintains aspect ratio). Default 1280 (set 0 to keep original)
"""
import argparse
import time
import os
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Real-time detection (PyTorch + YOLOv5 via torch.hub)")
    p.add_argument("--source", "-s", default="0", help="Webcam index or path to video file")
    p.add_argument("--weights", "-w", default="yolov5s", help="Weights: 'yolov5s' (torch.hub) or path to .pt")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--device", "-d", default="", help="Device: '' for auto, 'cpu', or '0' for GPU 0")
    p.add_argument("--save", help="Optional: path to save output video (mp4 recommended)")
    p.add_argument("--view-size", type=int, default=1280, help="Display width (0 to keep original)")
    return p.parse_args()


def letterbox(im, new_width=640):
    # Resize with unchanged aspect ratio using padding
    h, w = im.shape[:2]
    scale = new_width / max(h, w)
    if scale == 1.0:
        return im, 1.0, 0, 0
    new_w, new_h = int(w * scale), int(h * scale)
    im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # pad to square (new_width x new_width)
    top = (new_width - new_h) // 2
    left = (new_width - new_w) // 2
    padded = np.full((new_width, new_width, 3), 114, dtype=np.uint8)
    padded[top:top + new_h, left:left + new_w] = im_resized
    return padded, scale, left, top


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return (x, y, w, h)


def draw_boxes(img, boxes, confs, classes, names, color_map=None):
    for box, conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]} {conf:.2f}"
        color = (0, 255, 0) if color_map is None else color_map[int(cls) % len(color_map)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def main():
    args = parse_args()

    # Prepare source
    src = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    # Load model from torch.hub (ultralytics/yolov5) or local .pt
    # Note: torch.hub will download repo on first run.
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # If weights is a file path, load that; otherwise pass model name to torch.hub
    if Path(args.weights).exists():
        model = torch.hub.load("ultralytics/yolov5", "custom", path=args.weights, force_reload=False)
    else:
        model = torch.hub.load("ultralytics/yolov5", args.weights, pretrained=True)

    model.to(device)
    model.conf = args.conf  # confidence
    model.iou = args.iou    # NMS IoU
    names = model.names

    # Open video capture
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {src}")
        return

    # Output writer if requested
    writer = None
    save_path = args.save
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"[INFO] Saving output to {save_path} (fps={fps}, size={width}x{height})")

    # Optional color map for classes
    color_map = [tuple(int(x) for x in np.random.randint(0, 255, 3)) for _ in range(len(names))]

    prev_time = time.time()
    frame_count = 0
    avg_fps = 0.0

    view_width = args.view_size

    print("[INFO] Starting detection. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Keep original for saving/display
        orig = frame.copy()
        h0, w0 = frame.shape[:2]

        # Inference
        # Use model directly (Ultralytics YOLOv5 model expects BGR numpy arrays)
        results = model(frame, size=640)  # size=320,640 etc.

        # results.xyxy[0] -> n x 6 tensor (x1,y1,x2,y2,conf,cls)
        det = results.xyxy[0].cpu().numpy() if hasattr(results, "xyxy") else results[0].boxes.cpu().numpy()
        if det is not None and len(det):
            boxes = det[:, :4]
            confs = det[:, 4]
            classes = det[:, 5].astype(int)
            draw_boxes(frame, boxes, confs, classes, names, color_map=color_map)

        # FPS calculation
        now = time.time()
        dt = now - prev_time
        fps = 1.0 / dt if dt > 0 else 0.0
        avg_fps = (avg_fps * (frame_count - 1) + fps) / frame_count
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f} (avg {avg_fps:.1f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Display (resize for viewing if requested)
        if view_width and view_width > 0:
            scale = view_width / frame.shape[1]
            view = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        else:
            view = frame

        cv2.imshow("Real-Time Detection", view)

        # Save original-sized frame with annotations if requested
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
