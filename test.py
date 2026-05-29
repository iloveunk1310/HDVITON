from typing import Any


import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from kornia.filters import GaussianBlur2d

from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images

def get_opt():
    """
    Các tham số để tuỳ biến cho quá trình test.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)

    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])

    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('--dataset_list', type=str, default='test_pairs.txt')
    parser.add_argument('--custom_mask', action='store_true') #mask tự tạo
    parser.add_argument('--custom_mask_cache_dir', type=str, default='') #tạo cache lưu trữ mask
    parser.add_argument('--custom_pose', action='store_true') #pose json + map tự tạo
    parser.add_argument('--custom_pose_json_dir', type=str, default='test_pose') #thư mục lưu trữ pose json
    parser.add_argument('--custom_pose_img_dir', type=str, default='test_pose_img') #thư mục lưu trữ pose img
    parser.add_argument('--custom_pose_model', type=str, default='best_pose.pt') #mô hình sử dụng
    parser.add_argument('--custom_pose_device', type=str, default='') #thiết bị sử dụng (cpu mặc định)
    parser.add_argument('--custom_parse', action='store_true') #human parse tự tạo
    parser.add_argument('--custom_parse_dir', type=str, default='test_parse_mask') #thư mục lưu trữ parse
    parser.add_argument('--custom_parse_color_dir', type=str, default='test_parse') # thư mục lưu trữ parse màu sắc
    parser.add_argument('--custom_parse_model', type=str, default='checkpoints/best_pose_unet.pth') # model parse
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/') 
    parser.add_argument('--save_dir', type=str, default='./results/')

    parser.add_argument('--display_freq', type=int, default=1)

    parser.add_argument('--seg_checkpoint', type=str, default='seg_final.pth')
    parser.add_argument('--gmm_checkpoint', type=str, default='gmm_final.pth')
    parser.add_argument('--alias_checkpoint', type=str, default='alias_final.pth')

    # common
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'], default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    # for GMM
    parser.add_argument('--grid_size', type=int, default=5)

    # for ALIASGenerator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                             'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')

    opt = parser.parse_args()
    return opt


def ensure_custom_pose_assets(opt):
    """Tạo ra openpose-json + openpose-img cho ảnh người trong list."""
    if not getattr(opt, 'custom_pose', False):
        return

    list_path = os.path.join(opt.dataset_dir, opt.dataset_list)
    if not os.path.isfile(list_path):
        raise FileNotFoundError('Dataset list not found: {}'.format(list_path))

    image_names = []
    with open(list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            image_names.append(parts[0])

    if not image_names:
        raise RuntimeError('No valid rows in dataset list: {}'.format(list_path))

    img_dir = os.path.join(opt.dataset_dir, opt.dataset_mode, 'image')
    json_dir = opt.custom_pose_json_dir
    pose_img_dir = opt.custom_pose_img_dir
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(pose_img_dir, exist_ok=True)

    device = opt.custom_pose_device
    if not device:
        device = '0' if torch.cuda.is_available() else 'cpu'

    from gen_pose import gen_pose
    gen_pose(
        img_dir,
        json_dir,
        model_path=opt.custom_pose_model,
        device=device,
        image_names=image_names,
    )

    stems = [os.path.splitext(n)[0] for n in image_names]
    from gen_pose_img import gen_openpose_rendered_for_viton
    gen_openpose_rendered_for_viton(
        json_dir,
        img_dir,
        pose_img_dir,
        stems,
        canvas_w=opt.load_width,
        canvas_h=opt.load_height,
        overlay=False,
    )

def ensure_custom_parse_assets(opt):
    """Tự động sinh parse map từ ảnh người và pose map tương ứng."""
    if not getattr(opt, 'custom_parse', False):
        return

    # Xác định thư mục chứa pose json đầu vào dựa trên cấu hình custom_pose
    if getattr(opt, 'custom_pose', False):
        json_dir = opt.custom_pose_json_dir
    else:
        json_dir = os.path.join(opt.dataset_dir, opt.dataset_mode, 'openpose-json')

    image_dir = os.path.join(opt.dataset_dir, opt.dataset_mode, 'image')
    file_list_path = os.path.join(opt.dataset_dir, opt.dataset_list)

    # Import hàm xử lý từ gen_parse.py
    from gen_parse import process_directory

    print(f"[ensure_custom_parse] Bắt đầu tạo parse map sử dụng pose tại: {json_dir}")
    process_directory(
        image_dir=image_dir,
        json_dir=json_dir,
        output_color_dir=opt.custom_parse_color_dir,
        output_mask_dir=opt.custom_parse_dir,
        model_path=opt.custom_parse_model,
        file_list_path=file_list_path
    )


def test(opt, seg, gmm, alias):
    """
    Test với các cặp dữ liệu trong datasets\test_pairs.txt.

    Input:
        - opt: các tham số
        - seg: công cụ tạo segmentation cho riêng từng ảnh tương ứng với quần áo.
        - gmm: công cụ deformation cho ảnh

    """
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = GaussianBlur2d((15, 15), (3, 3))
    device = opt.device
    gauss.to(device)

    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            img_names = inputs['img_name']
            c_names = inputs['c_name']['unpaired']

            img_agnostic = inputs['img_agnostic'].to(device)
            parse_agnostic = inputs['parse_agnostic'].to(device)
            pose = inputs['pose'].to(device)
            c = inputs['cloth']['unpaired'].to(device)
            cm = inputs['cloth_mask']['unpaired'].to(device)

            # Part 1. Segmentation generation
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size()).to(device)), dim=1)

            parse_pred_down = seg(seg_input)
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]

            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float).to(device)
            parse_old.scatter_(1, parse_pred, 1.0)

            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float).to(device)
            for j in range(len(labels)):
                for label in labels[j][1]:
                    parse[:, j] += parse_old[:, label]

            # Part 2. Clothes Deformation
            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

            _, warped_grid = gmm(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

            # Part 3. Try-on synthesis
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask

            output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

            unpaired_names = []
            for img_name, c_name in zip(img_names, c_names):
                unpaired_names.append('{}_{}'.format(img_name.split('_')[0], c_name))

            save_images(output, unpaired_names, os.path.join(opt.save_dir, opt.name))

            if (i + 1) % opt.display_freq == 0:
                print("step: {}".format(i + 1))


def main():
    opt = get_opt()
    print(opt)
    device = opt.device
    if not os.path.exists(os.path.join(opt.save_dir, opt.name)):
        os.makedirs(os.path.join(opt.save_dir, opt.name))

    # Thực hiện tuần tự: Tạo pose trước (nếu có), sau đó tạo parse
    ensure_custom_pose_assets(opt)
    ensure_custom_parse_assets(opt)
    
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    load_checkpoint(seg, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
    load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))
    torch.cuda.empty_cache()
    seg.to(device).eval()
    gmm.to(device).eval()
    alias.to(device).eval()
    test(opt, seg, gmm, alias)


if __name__ == '__main__':
    main()
