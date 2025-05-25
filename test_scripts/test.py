
# test_derain.py
import argparse
import sys
import os
os.environ["HF_HOME"] = "https://hf-mirror.com"
sys.path.append(os.getcwd())

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from rectified_flow import RectifiedFlowDerain, Unet

def parse_args():
    parser = argparse.ArgumentParser("Test derain model (pixel space)")
    parser.add_argument("--ckpt_path",       type=str, required=True,
                        help="Path to the .pt checkpoint file")
    parser.add_argument("--input_dir",       type=str, required=True,
                        help="Directory with input (rainy) images")
    parser.add_argument("--output_dir",      type=str, required=True,
                        help="Directory to save derained outputs")
    parser.add_argument("--image_size",      type=int, default=128,
                        help="Size to which test images are resized / cropped")
    parser.add_argument("--batch_size",      type=int, default=1,
                        help="Batch size for inference (not used in this looped version)")
    parser.add_argument("--UnetDim",         type=int, default=16,
                        help="Base channel dimension for U-Net")
    parser.add_argument("--sampling_steps",  type=int, default=16,
                        help="Number of rectified‐flow sampling steps")
    parser.add_argument('--use_consistency', type=bool, default=True, help="whether use cfm")
    return parser.parse_args()

def load_model(ckpt_path: str, device: torch.device,
               dim: int, use_consistency: bool) -> RectifiedFlowDerain:

    base_unet = Unet(dim=dim, channels=3)
    model = RectifiedFlowDerain(base_unet, use_consistency=use_consistency)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # 载入模型
    model = load_model(
        ckpt_path=args.ckpt_path,
        device=device,
        dim=args.UnetDim,
        use_consistency=args.use_consistency
    )

    # 预处理与后处理
    tf = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])
    
    inv_tf = transforms.Compose([
        transforms.Resize(256)
    ])

    # 遍历输入目录
    for img_name in sorted(os.listdir(args.input_dir)):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            continue

        inp_path = Path(args.input_dir)  / img_name
        out_path = Path(args.output_dir) / img_name

        # 读图 & 预处理
        img = Image.open(inp_path).convert("RGB")
        x   = tf(img).unsqueeze(0).to(device)   # [1,3,H,W]

        # 正式采样
        with torch.no_grad():
            clean_pred = model.sample(
                rainy_image=x,
                data_shape=x.shape,
                steps=args.sampling_steps
            )
        # clamp 并保存
        clean_pred = clean_pred.clamp(0.0, 1.0)
        # post-process and save  
        clean_pred = inv_tf(clean_pred)
        
        save_image(clean_pred, out_path)

        print(f"Saved derained: {out_path}")

if __name__ == "__main__":
    main()
