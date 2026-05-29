"""
gen_pose.py — Tạo pose JSON theo chuẩn OpenPose BODY_25 dùng YOLOv8-pose.

YOLOv8-pose trả về 17 keypoints (COCO), script này:
  1. Map COCO-17 -> BODY_25 (25 keypoints, 75 giá trị)
  2. Tính Neck   = midpoint(RShoulder, LShoulder)   [BODY_25 idx 1]
  3. Tính MidHip = midpoint(RHip, LHip)             [BODY_25 idx 8]
  4. Foot keypoints (19-24) = 0  (YOLO không detect)
  5. face / hand = zeros đúng kích thước (70 kp / 21 kp mỗi tay)

"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image
from ultralytics import YOLO


# ── Mapping COCO-17 index → BODY_25 index ───────────────────────────────────
# COCO  : 0 Nose | 1 LEye | 2 REye | 3 LEar | 4 REar
#         5 LShoulder | 6 RShoulder | 7 LElbow | 8 RElbow
#         9 LWrist | 10 RWrist | 11 LHip | 12 RHip
#         13 LKnee | 14 RKnee | 15 LAnkle | 16 RAnkle
#
# BODY_25: 0 Nose | 1 Neck* | 2 RShoulder | 3 RElbow | 4 RWrist
#          5 LShoulder | 6 LElbow | 7 LWrist | 8 MidHip*
#          9 RHip | 10 RKnee | 11 RAnkle | 12 LHip | 13 LKnee | 14 LAnkle
#          15 REye | 16 LEye | 17 REar | 18 LEar
#          19-24 foot keypoints (zeros, YOLO không detect)
#  (* = tính toán, không map trực tiếp)
# ────────────────────────────────────────────────────────────────────────────
COCO_TO_BODY25: dict[int, int] = {
    0:  0,   # Nose
    1:  16,  # LEye
    2:  15,  # REye
    3:  18,  # LEar
    4:  17,  # REar
    5:  5,   # LShoulder
    6:  2,   # RShoulder
    7:  6,   # LElbow
    8:  3,   # RElbow
    9:  7,   # LWrist
    10: 4,   # RWrist
    11: 12,  # LHip
    12: 9,   # RHip
    13: 13,  # LKnee
    14: 10,  # RKnee
    15: 14,  # LAnkle
    16: 11,  # RAnkle
}

N_BODY25 = 25
N_FACE   = 70
N_HAND   = 21


def _midpoint(a: list, b: list) -> list:
    """Trả về [x, y, conf] midpoint của 2 keypoint. Trả về zeros nếu cả 2 đều không detect."""
    if a[2] <= 0 or b[2] <= 0:
        return [0.0, 0.0, 0.0]
    return [
        round((a[0] + b[0]) / 2, 3),
        round((a[1] + b[1]) / 2, 3),
        round((a[2] + b[2]) / 2, 6),
    ]


def coco17_to_body25(xy: np.ndarray, conf: np.ndarray | None) -> list[float]:
    """
    Chuyển 17 COCO keypoints (xy shape [17,2], conf shape [17]) sang
    BODY_25 flat list 75 giá trị [x0,y0,c0, x1,y1,c1, ...].
    """
    body25 = [[0.0, 0.0, 0.0] for _ in range(N_BODY25)]

    #  Map trực tiếp COCO → BODY_25
    for coco_idx, b25_idx in COCO_TO_BODY25.items():
        if coco_idx >= len(xy):
            continue
        x = float(xy[coco_idx][0])
        y = float(xy[coco_idx][1])
        c = float(conf[coco_idx]) if conf is not None else 1.0
        # nếu trả về (0,0) với conf > 0 khi không detect thì reset
        if x == 0.0 and y == 0.0:
            c = 0.0
        body25[b25_idx] = [round(x, 3), round(y, 3), round(c, 6)]

    # Neck (idx 1) = midpoint RShoulder(2) & LShoulder(5)
    body25[1] = _midpoint(body25[2], body25[5])

    # MidHip (idx 8) = midpoint RHip(9) & LHip(12)
    body25[8] = _midpoint(body25[9], body25[12])

    # Fixing missing keypoints
    #body25 = fix_keypoints(body25)
    # print(body25)


    # Flatten
    flat: list[float] = []
    for kp in body25:
        flat.extend(kp)
    return flat


def make_openpose_person(body25_flat: list[float]) -> dict:
    """
    Trả về danh sách điểm phù hợp với cấu trúc của openpose
    """
    return {
        "person_id": [-1],
        "pose_keypoints_2d":       body25_flat,
        "face_keypoints_2d":       [0.0] * (N_FACE * 3),
        "hand_left_keypoints_2d":  [0.0] * (N_HAND * 3),
        "hand_right_keypoints_2d": [0.0] * (N_HAND * 3),
        "pose_keypoints_3d":       [],
        "face_keypoints_3d":       [],
        "hand_left_keypoints_3d":  [],
        "hand_right_keypoints_3d": [],
    }


def result_to_openpose_json(results) -> dict:
    """Chuyển YOLO results của 1 ảnh -> dict OpenPose JSON."""
    empty_body25 = [0.0] * (N_BODY25 * 3)

    if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
        return {"version": 1.3, "people": [make_openpose_person(empty_body25)]}

    r = results[0]
    kps = r.keypoints

    # Chọn người có bounding-box area lớn nhất (người chính trong ảnh)
    best_idx = 0
    if r.boxes is not None and len(r.boxes) > 1:
        areas = r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]  # w*h
        best_idx = int(areas.argmax())

    xy   = kps.xy[best_idx].cpu().numpy()       # shape (17, 2)
    conf = kps.conf[best_idx].cpu().numpy() if kps.conf is not None else None  # (17,)

    body25_flat = coco17_to_body25(xy, conf)
    return {"version": 1.3, "people": [make_openpose_person(body25_flat)]}


# Main

def gen_pose(image_dir, output_dir, model_path="yolov8n-pose.pt", device="cpu", image_names=None):
    """
    Tạo file OpenPose BODY_25 JSON cho mỗi ảnh.

    Input:
        - image_dir: thư mục chứa ảnh.
        - output_dir: thư mục đầu ra.
        - image_names: danh sách ảnh trong thư mục đầu vào cần thực thi (nếu không sẽ thực thi hết)
        - model_path: model sử dụng
        - device:thiết bị (cpu hay gpu)

    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[gen_pose] Loading model: {model_path} on {device}")
    yolo = YOLO(model_path)

    exts = {".jpg", ".jpeg", ".png"}
    if image_names is None:
        to_process = [
            n for n in sorted(os.listdir(image_dir))
            if os.path.splitext(n)[1].lower() in exts
        ]
    else:
        to_process = []
        seen = set()
        for n in image_names:
            if not n or n in seen:
                continue
            seen.add(n)
            low = n.lower()
            if os.path.splitext(low)[1] not in exts:
                continue
            if not os.path.isfile(os.path.join(image_dir, n)):
                print(f"[gen_pose] Skip (missing): {n}")
                continue
            to_process.append(n)
        to_process.sort()

    if not to_process:
        print(f"[gen_pose] Không có ảnh để xử lý trong {image_dir}")
        return

    print(f"[gen_pose] Processing {len(to_process)} images...")
    for i, image_name in enumerate(to_process, 1):
        image_path = os.path.join(image_dir, image_name)
        results = yolo(image_path, device=device, verbose=False)
        pose_json = result_to_openpose_json(results)

        json_name = os.path.splitext(image_name)[0] + "_keypoints.json"
        out_path = os.path.join(output_dir, json_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pose_json, f, separators=(",", ":"))

        if i % 50 == 0 or i == len(to_process):
            print(f"  [{i}/{len(to_process)}] {json_name}")

    print(f"[gen_pose] Done. Saved to {output_dir}/")


if __name__ == "__main__":
    gen_pose(
        image_dir="datasets/test/image",
        output_dir="test_pose",
        model_path="best_pose.pt",
        device="cuda",
    )
