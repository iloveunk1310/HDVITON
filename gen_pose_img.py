"""
render_pose.py — Vẽ skeleton OpenPose BODY_25 từ JSON ra ảnh PNG.

"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

# BODY_25 skeleton: (kp_start, kp_end, BGR_color)
LIMBS = [
    ( 0,  1, (255,   0, 255)),   # Nose      → Neck
    ( 1,  2, (255,   0, 170)),   # Neck      → RShoulder
    ( 2,  3, (255,   0,  85)),   # RShoulder → RElbow
    ( 3,  4, (255,   0,   0)),   # RElbow    → RWrist
    ( 1,  5, (170, 255,   0)),   # Neck      → LShoulder
    ( 5,  6, ( 85, 255,   0)),   # LShoulder → LElbow
    ( 6,  7, (  0, 255,   0)),   # LElbow    → LWrist
    ( 1,  8, (  0,   0, 255)),   # Neck      → MidHip
    ( 8,  9, (255, 170,   0)),   # MidHip    → RHip
    ( 9, 10, (255, 255,   0)),   # RHip      → RKnee
    (10, 11, (170, 255,   0)),   # RKnee     → RAnkle
    ( 8, 12, (  0, 255,  85)),   # MidHip    → LHip
    (12, 13, (  0, 255, 170)),   # LHip      → LKnee
    (13, 14, (  0, 255, 255)),   # LKnee     → LAnkle
    ( 0, 15, (  0,  85, 255)),   # Nose      → REye
    (15, 17, (  0,   0, 255)),   # REye      → REar
    ( 0, 16, ( 85,   0, 255)),   # Nose      → LEye
    (16, 18, (170,   0, 255)),   # LEye      → LEar
    (11, 24, (  0, 170, 255)),   # RAnkle    → RHeel
    (11, 22, (  0, 255, 170)),   # RAnkle    → RBigToe
    (22, 23, (  0, 255,  85)),   # RBigToe   → RSmallToe
    (14, 21, (255, 170,   0)),   # LAnkle    → LHeel
    (14, 19, (255, 255,   0)),   # LAnkle    → LBigToe
    (19, 20, (255,  85,   0)),   # LBigToe   → LSmallToe
]

KP_COLOR  = (0, 220, 255)   # màu chấm keypoint (cyan-yellow)
LINE_THICK = 3
DOT_RADIUS = 5


def _is_valid(x, y, c): #kiểm tra tính hợp lí của vị trí limbs và màu 
    return c > 0 and not (x == 0.0 and y == 0.0)


def render_pose_on_canvas(json_path: str, canvas_w: int = 768, canvas_h: int = 1024,
                           background_img: np.ndarray | None = None) -> np.ndarray:
    """
    Đọc file JSON và vẽ skeleton lên canvas.

    Input:
        json_path       : đường dẫn file *_keypoints.json
        canvas_w/h      : kích thước canvas nếu không có background_img
        background_img  : ảnh BGR để overlay lên (nếu None thì nền đen)

    Output:
        canvas: BGR numpy mảng
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if background_img is not None:
        canvas = background_img.copy()
    else:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for person in data.get("people", []):
        raw = person.get("pose_keypoints_2d", [])
        if not raw:
            continue
        kps = np.array(raw, dtype=np.float32).reshape(-1, 3)   # (25, 3)

        # Vẽ limbs 
        for (a, b, color) in LIMBS:
            if a >= len(kps) or b >= len(kps):
                continue
            xa, ya, ca = kps[a]
            xb, yb, cb = kps[b]
            if not _is_valid(xa, ya, ca) or not _is_valid(xb, yb, cb):
                continue
            cv2.line(canvas, (int(xa), int(ya)), (int(xb), int(yb)),
                     color, LINE_THICK, cv2.LINE_AA)

        # Vẽ dots
        for (x, y, c) in kps:
            if not _is_valid(x, y, c):
                continue
            cv2.circle(canvas, (int(x), int(y)), DOT_RADIUS, KP_COLOR, -1, cv2.LINE_AA)

    return canvas


# def process_single(json_path: str, image_path: str | None, out_path: str, canvas_w: int = 768, canvas_h: int = 1024):
#     """

#     Thực hiện cho một hình ảnh duy nhất
#     """
#     bg = None
#     if image_path and os.path.exists(image_path):
#         bg = cv2.imread(image_path)

#     canvas = render_pose_on_canvas(json_path, canvas_w, canvas_h, bg)
#     cv2.imwrite(out_path, canvas)
#     print(f"Saved: {out_path}")


def process_batch(json_dir: str, out_dir: str, image_dir: str | None,
                  canvas_w: int = 768, canvas_h: int = 1024):
    """

    Thử nghiệm ảnh trên một tập file json.

    Input:
        - json_dir: thư mục json.
        - out_dir: thư mục đích
        - image_dir: thư mục ảnh người (để chèn lên nếu cần)
        - canvas_w/h: kích thước ảnh
    
    """
    os.makedirs(out_dir, exist_ok=True)
    json_files = sorted([n for n in os.listdir(json_dir) if n.endswith(".json")])
    print(f"Rendering {len(json_files)} pose files...")

    for i, jf in enumerate(json_files, 1):
        json_path = os.path.join(json_dir, jf)

        # Tên ảnh gốc: loại bỏ _keypoints.json
        base      = jf.replace("_keypoints.json", "")
        out_path  = os.path.join(out_dir, base + "_pose.png")

        bg = None
        if image_dir:
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = os.path.join(image_dir, base + ext)
                if os.path.exists(candidate):
                    bg = cv2.imread(candidate)
                    break
        canvas = render_pose_on_canvas(json_path, canvas_w, canvas_h, bg)
        cv2.imwrite(out_path, canvas)

        if i % 100 == 0 or i == len(json_files):
            print(f"  [{i}/{len(json_files)}] {out_path}")

    print(f"Done. Saved to {out_dir}/")


def gen_openpose_rendered_for_viton(json_dir: str, image_dir: str, out_dir: str, stems: list[str],
    canvas_w: int = 768, canvas_h: int = 1024, overlay: bool = False) -> None:
    """

    Render ảnh cho quy trình tạo ảnh.

    Input:
        - json_dir: thư mục json.
        - out_dir: thư mục đích
        - image_dir: thư mục ảnh người
        - stem: danh sách tên các ảnh sẽ thực hiện (vì không phải thực hiện trên toàn bộ json_dir)
        - canvas_w/h: kích thước ảnh
    
    """
    os.makedirs(out_dir, exist_ok=True)
    stems = sorted(set(stems))
    print(f"[gen_pose_img] Rendering {len(stems)} pose images → {out_dir}/")

    for i, stem in enumerate(stems, 1):
        json_path = os.path.join(json_dir, stem + "_keypoints.json")
        if not os.path.isfile(json_path):
            print(f"[gen_pose_img] Skip (missing JSON): {json_path}")
            continue
        out_path = os.path.join(out_dir, stem + "_rendered.png")

        bg = None
        if overlay:
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = os.path.join(image_dir, stem + ext)
                if os.path.exists(candidate):
                    bg = cv2.imread(candidate)
                    break

        if bg is not None:
            canvas = render_pose_on_canvas(json_path, canvas_w, canvas_h, bg)
        else:
            canvas = render_pose_on_canvas(json_path, canvas_w, canvas_h, None)

        cv2.imwrite(out_path, canvas)
        if i % 50 == 0 or i == len(stems):
            print(f"  [{i}/{len(stems)}] {os.path.basename(out_path)}")

    print(f"[gen_pose_img] Done.")



if __name__ == "__main__":
    process_batch(json_dir="test_pose", out_dir="test_pose_img", image_dir=None, canvas_w=768, canvas_h=1024)