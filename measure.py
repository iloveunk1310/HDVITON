"""
eval_metrics.py - Tính SSIM, LPIPS, FID cho toàn bộ tập kết quả.

Cấu trúc thư mục:
    results/
        generated/ : ảnh model sinh ra (tên khớp với ground_truth/)
        ground_truth/
"""

import os
import numpy as np
import cv2
import torch
import lpips
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score
import torchvision.transforms as T

# Setup
lpips_fn  = lpips.LPIPS(net='alex')
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

def compute_ssim_single(pred_path, gt_path):
    """

    Tính ssim () cho kết quả.

    Input:
        - pred_path: tập ảnh kết quả để test
        - gt_path: tập ảnh ground truth (ảnh gốc chính xác)

    Output:
        - điểm ssim.
    """
    p = cv2.imread(pred_path)
    g = cv2.imread(gt_path)
    g = cv2.resize(g, (p.shape[1], p.shape[0]))
    score, _ = ssim(
        cv2.cvtColor(p, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(g, cv2.COLOR_BGR2GRAY),
        full=True
    )
    return score

def compute_lpips_single(pred_path, gt_path):
    """

    Tính lpips() cho kết quả.

    Input:
        - pred_path: tập ảnh kết quả để test
        - gt_path: tập ảnh ground truth (ảnh gốc chính xác)

    Output:
        - điểm lpips.
    """
    pred = transform(Image.open(pred_path).convert('RGB')).unsqueeze(0)
    gt   = transform(Image.open(gt_path).convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        return lpips_fn(pred, gt).item()

def evaluate(gen_dir, gt_dir, device='cpu'):
    """

    Tính fid(), báo cáo cả 3 chỉ số cho kết quả.

    Input:
        - gen_dir: tập ảnh kết quả để test
        - gt_dir: tập ảnh ground truth (ảnh gốc chính xác)

    Output:
        - điểm fid, ssim và lpips.
    """
    names = sorted(os.listdir(gen_dir))
    ssim_scores, lpips_scores = [], []

    for name in names:
        pred_path = os.path.join(gen_dir, name)
        gt_path   = os.path.join(gt_dir,  name)
        if not os.path.exists(gt_path):
            continue

        ssim_scores.append(compute_ssim_single(pred_path, gt_path))
        lpips_scores.append(compute_lpips_single(pred_path, gt_path))

    fid = fid_score.calculate_fid_given_paths(
        [gt_dir, gen_dir],
        batch_size=8, device=device, dims=2048
    )

    print(f"Evaluated on {len(ssim_scores)} image pairs")
    print(f"  SSIM  ↑ : {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    print(f"  LPIPS ↓ : {np.mean(lpips_scores):.4f} ± {np.std(lpips_scores):.4f}")
    print(f"  FID   ↓ : {fid:.2f}")
    return {"ssim": np.mean(ssim_scores), "lpips": np.mean(lpips_scores), "fid": fid}

if __name__ == "__main__":
    evaluate("results/run_custom_mask", "results/ground_truth")