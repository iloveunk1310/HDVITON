import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

#---- Mô hình RESUNET----
class ResBlock(nn.Module):
    """
    Residual Block: conv-bn-relu-conv-bn + shortcut.
    Shortcut dùng 1×1 Conv nếu in_ch ≠ out_ch, còn lại là identity.
 
    Input  : (B, in_ch,  H, W)
    Output : (B, out_ch, H, W)
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # Shortcut: 1×1 conv nếu kênh thay đổi, identity nếu giữ nguyên
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.main(x) + self.shortcut(x))
 
 
class AttentionGate(nn.Module):
    """
    Attention Gate: dùng gating signal từ decoder để lọc skip connection
    từ encoder, tập trung vào vùng không gian liên quan.
 
    Tham khảo: Attention U-Net (Oktay et al., 2018).
 
    Input:
        g : gating signal từ decoder  (B, f_g, H, W) - độ phân giải thấp hơn
        x : skip connection từ encoder (B, f_l, H, W) - cùng độ phân giải với output
 
    Output:
        (B, f_l, H, W) - skip connection đã được nhân với attention map [0,1]
    """
    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g,  f_int, 1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l,  f_int, 1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Upsample g lên cùng kích thước với x nếu cần
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear",
                              align_corners=False)
        attn = self.psi(self.relu(self.W_g(g) + self.W_x(x)))  # (B,1,H,W)
        return x * attn

class ResUNet(nn.Module):
    """
    ResUNet với Attention Gate cho pose-guided human parsing.
 
    Kiến trúc:
        Encoder  : 4 tầng ResBlock + MaxPool, kênh tăng dần 32→64→128→256
        Bottleneck: ResBlock 256→512 + Dropout
        Decoder  : 4 tầng ConvTranspose + AttentionGate + ResBlock
 
    Input  : (B, in_ch, H, W)       — mặc định in_ch=21 (3 RGB + 18 pose)
    Output : (B, num_classes, H, W) — logits chưa qua softmax
    """
 
    def __init__(
        self,
        in_ch:       int   = 21,
        num_classes: int   = 20,
        base_ch:     int   = 32,    # số filter cơ sở (nhân đôi qua mỗi tầng)
        dropout:     float = 0.3,   # dropout tại bottleneck
    ):
        super().__init__()
        b = base_ch   # 32
 
        #  Encoder 
        self.enc1 = ResBlock(in_ch, b)          # (B, 32, H,   W)
        self.enc2 = ResBlock(b,     b * 2)      # (B, 64, H/2, W/2)
        self.enc3 = ResBlock(b * 2, b * 4)      # (B,128, H/4, W/4)
        self.enc4 = ResBlock(b * 4, b * 8)      # (B,256, H/8, W/8)
        self.pool = nn.MaxPool2d(2)
 
        #  Bottleneck 
        self.bottleneck = ResBlock(b * 8, b * 16, dropout=dropout)
        # (B, 512, H/16, W/16)
 
        #  Decoder 
        # up + attention gate + ResBlock
 
        self.up4  = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.attn4 = AttentionGate(f_g=b * 8, f_l=b * 8, f_int=b * 4)
        self.dec4  = ResBlock(b * 16, b * 8)    # concat → b*8 + b*8 = b*16 in
 
        self.up3  = nn.ConvTranspose2d(b * 8,  b * 4, 2, stride=2)
        self.attn3 = AttentionGate(f_g=b * 4, f_l=b * 4, f_int=b * 2)
        self.dec3  = ResBlock(b * 8,  b * 4)
 
        self.up2  = nn.ConvTranspose2d(b * 4,  b * 2, 2, stride=2)
        self.attn2 = AttentionGate(f_g=b * 2, f_l=b * 2, f_int=b)
        self.dec2  = ResBlock(b * 4,  b * 2)
 
        self.up1  = nn.ConvTranspose2d(b * 2,  b,     2, stride=2)
        self.attn1 = AttentionGate(f_g=b,     f_l=b,     f_int=b // 2)
        self.dec1  = ResBlock(b * 2,  b)
 
        #  Head 
        self.head = nn.Conv2d(b, num_classes, 1)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)                       # (B, 32,  H,    W)
        e2 = self.enc2(self.pool(e1))            # (B, 64,  H/2,  W/2)
        e3 = self.enc3(self.pool(e2))            # (B, 128, H/4,  W/4)
        e4 = self.enc4(self.pool(e3))            # (B, 256, H/8,  W/8)
 
        # Bottleneck
        b  = self.bottleneck(self.pool(e4))      # (B, 512, H/16, W/16)
 
        # Decoder: upsample → attention → concat → ResBlock
        d4 = self.up4(b)                         # (B, 256, H/8,  W/8)
        d4 = self.dec4(torch.cat([self.attn4(d4, e4), d4], dim=1))
 
        d3 = self.up3(d4)                        # (B, 128, H/4,  W/4)
        d3 = self.dec3(torch.cat([self.attn3(d3, e3), d3], dim=1))
 
        d2 = self.up2(d3)                        # (B, 64,  H/2,  W/2)
        d2 = self.dec2(torch.cat([self.attn2(d2, e2), d2], dim=1))
 
        d1 = self.up1(d2)                        # (B, 32,  H,    W)
        d1 = self.dec1(torch.cat([self.attn1(d1, e1), d1], dim=1))
 
        return self.head(d1)                     # (B, num_classes, H, W)


#--- Sử dụng mô hình để dự đoán
def get_pose_heatmaps(json_path, orig_w, orig_h, target_w=192, target_h=256, sigma=3.0):
    with open(json_path, 'r') as f:
        data = json.load(f)
    keypoints = data['people'][0]['pose_keypoints_2d']
    kpts = np.array(keypoints).reshape(-1, 3) 
    heatmaps = np.zeros((18, target_h, target_w), dtype=np.float32)
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    for i in range(18):
        x, y, conf = kpts[i]
        if conf > 0.1: 
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            grid_y, grid_x = np.mgrid[0:target_h, 0:target_w]
            dist_sq = (grid_x - x_scaled) ** 2 + (grid_y - y_scaled) ** 2
            heatmaps[i] = np.exp(-dist_sq / (2.0 * sigma ** 2))
    return torch.from_numpy(heatmaps)

# Định nghĩa màu sắc cho 20 class 
PARSE_COLORS = [
    [0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 85, 0], [170, 0, 51], [255, 85, 0], 
    [0, 0, 85], [0, 119, 221], [85, 85, 0], [0, 85, 85], [85, 51, 0], [52, 86, 128], 
    [0, 128, 0], [0, 0, 255], [51, 170, 221], [0, 255, 255], [85, 255, 170], 
    [170, 255, 85], [255, 255, 0], [255, 170, 0]
]

def process_directory(image_dir, json_dir, output_color_dir, output_mask_dir, model_path, file_list_path=None):
    """
    Hàm xử lý hàng loạt ảnh và pose để tạo Parse Map.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Khởi tạo thiết bị: {device}")

    # Load Model (Chỉ load 1 lần duy nhất)
    model = ResUNet(in_ch=21, num_classes=20).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[*] Đã load trọng số mô hình thành công.")

    # Tạo thư mục output nếu chưa có
    os.makedirs(output_color_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # Lấy danh sách file cần xử lý
    filenames = []
    if file_list_path and os.path.exists(file_list_path):
        # Nếu truyền vào 1 file txt chứa danh sách tên 
        with open(file_list_path, 'r') as f:
            for line in f:
                # Tách lấy tên file ảnh 
                img_name = line.strip().split()[0]
                if img_name:
                    filenames.append(img_name)
    else:
        # Lấy toàn bộ file .jpg và .png trong thư mục image_dir
        filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    print(f"[*] Tìm thấy {len(filenames)} ảnh cần xử lý. Bắt đầu chạy...")

    # Vòng lặp suy luận 
    # Dùng tqdm để hiển thị thanh tiến trình cho chuyên nghiệp
    for filename in tqdm(filenames, desc="Processing Images"):
        img_path = os.path.join(image_dir, filename)
        
        # Suy luận tên file JSON theo chuẩn VITON-HD 
        base_name = os.path.splitext(filename)[0]
        json_name = f"{base_name}_keypoints.json"
        json_path = os.path.join(json_dir, json_name)

        # Kiểm tra file JSON có tồn tại không
        if not os.path.exists(json_path):
            print(f"\n[!] Cảnh báo: Không tìm thấy file pose {json_name}. Bỏ qua ảnh này.")
            continue

        try:
            # Xử lý dữ liệu 
            img_pil = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img_pil.size
            target_w, target_h = 192, 256
            
            img_resized = img_pil.resize((target_w, target_h), Image.BILINEAR)
            img_tensor = TF.to_tensor(img_resized) 
            
            pose_tensor = get_pose_heatmaps(json_path, orig_w, orig_h, target_w, target_h)
            
            # Ghép Tensor và đẩy lên GPU/CPU
            input_tensor = torch.cat([img_tensor, pose_tensor], dim=0).unsqueeze(0).to(device)

            # Chạy Model 
            with torch.no_grad():
                output = model(input_tensor)
                parse_pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            hr_w, hr_h = 768, 1024 
            # Rescale Mask thô 
            parse_pred_hr = cv2.resize(
                parse_pred.astype(np.uint8), 
                (hr_w, hr_h), 
                interpolation=cv2.INTER_NEAREST
            )

            # Rescale Ảnh màu
            parse_color = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            for class_idx in range(20):
                parse_color[parse_pred == class_idx] = PARSE_COLORS[class_idx]
            
            parse_color_bgr = cv2.cvtColor(parse_color, cv2.COLOR_RGB2BGR)
            parse_color_bgr_hr = cv2.resize(
                parse_color_bgr, 
                (hr_w, hr_h), 
                interpolation=cv2.INTER_NEAREST
            )

            # --- Lưu Kết Quả Độ Phân Giải Cao ---
            cv2.imwrite(os.path.join(output_color_dir, f"{base_name}.png"), parse_color_bgr_hr)
            cv2.imwrite(os.path.join(output_mask_dir, f"{base_name}.png"), parse_pred_hr)

        except Exception as e:
            print(f"\n[!] Lỗi khi xử lý {filename}: {str(e)}")

    print("[*] HOÀN THÀNH TOÀN BỘ QUÁ TRÌNH DỰ ĐOÁN!")
# ==========================================
# CÁCH CHẠY THỬ
# ==========================================
if __name__ == "__main__":
    IMG_PATH = "datasets/test/image"
    JSON_PATH =  "datasets/test/openpose-json" 
    MODEL_WEIGHTS = "checkpoints/best_pose_unet.pth"
    OUTPUT_PARSE = "test_parse"
    
    process_directory(IMG_PATH, JSON_PATH, OUTPUT_PARSE, "test_parse_mask", MODEL_WEIGHTS)