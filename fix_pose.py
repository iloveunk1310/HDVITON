"""
fix_keypoints.py — Vá các keypoint bị thiếu bằng ngoại suy tuyến tính.

Tích hợp vào gen_pose.py: gọi fix_missing_keypoints(body25) trước khi flatten.
"""

from __future__ import annotations

import numpy as np

# BODY_25 index
KP = {
    "Nose":0,"Neck":1,"RShoulder":2,"RElbow":3,"RWrist":4,
    "LShoulder":5,"LElbow":6,"LWrist":7,"MidHip":8,
    "RHip":9,"RKnee":10,"RAnkle":11,
    "LHip":12,"LKnee":13,"LAnkle":14,
    "REye":15,"LEye":16,"REar":17,"LEar":18,
}

# Chuỗi phụ thuộc: nếu điểm cuối bị miss, ngoại suy từ 2 điểm trước
# (parent, grandparent, missing_child)  — child = 2*parent - grandparent
EXTRAPOLATE_CHAINS = [
    # Tay phải
    (KP["RShoulder"], KP["RElbow"],   KP["RWrist"]),
    (KP["Neck"],      KP["RShoulder"],KP["RElbow"]),
    # Tay trái
    (KP["LShoulder"], KP["LElbow"],   KP["LWrist"]),
    (KP["Neck"],      KP["LShoulder"],KP["LElbow"]),
    # Chân phải
    (KP["RHip"],  KP["RKnee"],  KP["RAnkle"]),
    (KP["MidHip"],KP["RHip"],   KP["RKnee"]),
    # Chân trái
    (KP["LHip"],  KP["LKnee"],  KP["LAnkle"]),
    (KP["MidHip"],KP["LHip"],   KP["LKnee"]),
]

# Nếu cả 2 tay đều detect được -> tính tay còn lại bằng đối xứng qua Neck
SYMMETRY_PAIRS = [
    (KP["RShoulder"], KP["LShoulder"], KP["Neck"]),
    (KP["RElbow"],    KP["LElbow"],    KP["Neck"]),
    (KP["RWrist"],    KP["LWrist"],    KP["Neck"]),
    (KP["RHip"],      KP["LHip"],      KP["MidHip"]),
    (KP["RKnee"],     KP["LKnee"],     KP["MidHip"]),
    (KP["RAnkle"],    KP["LAnkle"],    KP["MidHip"]),
]

FALLBACK_CONF = 0.3   # confidence gán cho keypoint được interpolate


def _valid(kp):
    """Keypoint [x, y, conf] hợp lệ."""
    return kp[2] > 0 and not (kp[0] == 0 and kp[1] == 0)


def fix_missing_keypoints(kps: list[list[float]]) -> list[list[float]]:
    """
    Vá keypoint bị thiếu (conf=0) bằng:
      1. Ngoại suy tuyến tính dọc theo chuỗi khớp
      2. Đối xứng trái-phải qua trục giữa

    Input:
        - kps: list 25 phần tử, mỗi phần tử là [x, y, conf]

    Output:
        - kps: kps đã được sửa
    """
    kps = [list(k) for k in kps]   # copy để không mutate gốc

    # goại suy tuyến tính 
    for _ in range(2):
        for parent_idx, grand_idx, child_idx in EXTRAPOLATE_CHAINS:
            if _valid(kps[child_idx]):
                continue   # đã có, bỏ qua
            p = kps[parent_idx]
            g = kps[grand_idx]
            if not (_valid(p) and _valid(g)):
                continue   # không đủ thông tin
            # child ≈ parent + (parent - grandparent)
            kps[child_idx] = [
                round(2 * p[0] - g[0], 3),
                round(2 * p[1] - g[1], 3),
                FALLBACK_CONF,
            ]

    # Đối xứng trái-phải
    for right_idx, left_idx, axis_idx in SYMMETRY_PAIRS:
        r, l, ax = kps[right_idx], kps[left_idx], kps[axis_idx]
        if not _valid(ax):
            continue
        if _valid(r) and not _valid(l):
            # Tính left bằng cách lật right qua axis
            kps[left_idx] = [
                round(2 * ax[0] - r[0], 3),
                round(2 * ax[1] - r[1], 3),
                FALLBACK_CONF,
            ]
        elif _valid(l) and not _valid(r):
            kps[right_idx] = [
                round(2 * ax[0] - l[0], 3),
                round(2 * ax[1] - l[1], 3),
                FALLBACK_CONF,
            ]

    return kps


# Kiểm tra
if __name__ == "__main__":
    # Giả lập: RShoulder và LShoulder detect được, nhưng cả 2 Wrist đều miss
    dummy = [[0.0, 0.0, 0.0]] * 25
    dummy[KP["Nose"]]       = [349.0, 147.0, 0.92]
    dummy[KP["Neck"]]       = [400.0, 317.0, 0.77]
    dummy[KP["RShoulder"]]  = [292.0, 329.0, 0.74]
    dummy[KP["RElbow"]]     = [260.0, 480.0, 0.65]
    # RWrist = 0,0,0  (miss)
    dummy[KP["LShoulder"]]  = [513.0, 303.0, 0.68]
    dummy[KP["LElbow"]]     = [527.0, 450.0, 0.80]
    # LWrist = 0,0,0  (miss)
    dummy[KP["MidHip"]]     = [351.0, 711.0, 0.52]
    dummy[KP["RHip"]]       = [272.0, 697.0, 0.46]
    dummy[KP["LHip"]]       = [428.0, 723.0, 0.50]

    print("Trước khi fix:")
    for name, idx in KP.items():
        kp = dummy[idx]
        status = "✓" if _valid(kp) else "✗ MISS"
        print(f"  [{idx:2d}] {name:<12} {status}")

    fixed = fix_missing_keypoints(dummy)

    print("\nSau khi fix:")
    for name, idx in KP.items():
        kp = fixed[idx]
        was_valid = _valid(dummy[idx])
        status = "✓" if _valid(kp) else "✗ vẫn miss"
        tag = " ← interpolated" if _valid(kp) and not was_valid else ""
        print(f"  [{idx:2d}] {name:<12} x={kp[0]:.1f} y={kp[1]:.1f} conf={kp[2]:.2f} {status}{tag}")