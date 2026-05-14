import cv2
import numpy as np
import os
from rembg import remove
from PIL import Image

def generate_cloth_mask(cloth_path):
    """

    Tạo mask (nền đen) cho trang phục.

    Input:
        - cloth_path: thư mục chứa ảnh trang phục.
    
    Ouput:
        - mask: thư mục mask đã tạo.

    """
    img = Image.open(cloth_path).convert("RGBA")
    
    # rembg xóa nền, trả về RGBA
    result = remove(img)
    # Lấy alpha channel làm mask
    alpha = np.array(result)[:, :, 3]
    # Nhị phân hóa
    _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    # Smooth nhẹ viền
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

if __name__ == "__main__":
    cloth_dir = "datasets/test/cloth"
    mask_dir = "test_mask"
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    for cloth_path in os.listdir(cloth_dir):
        mask = generate_cloth_mask(os.path.join(cloth_dir, cloth_path))
        cv2.imwrite(os.path.join(mask_dir, cloth_path), mask)