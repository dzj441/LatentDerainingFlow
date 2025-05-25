# test_derain.py
import argparse
import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from rectified_flow import RectifiedFlowDerain, Unet
from diffusers import AutoencoderKL
def parse_args():
    parser = argparse.ArgumentParser("Test derain model")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to the .pt checkpoint file")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with input (rainy) images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save derained outputs")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Size to which test images are resized")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument('--vae_model',type=str,default="sd2-vae",help="VAE model to use, default is 'stabilityai/sdxl-vae'") # "stabilityai/sdxl-vae" if from remote
    parser.add_argument('--UnetDim', type=int, default=16, help="UnetInitDim")
    parser.add_argument('--sampling_steps', type=int, default=16, help="sampling steps")
    parser.add_argument('--use_consistency', type=bool, default=True, help="whether use cfm")
    
        
    return parser.parse_args()

def load_model(ckpt_path, device,dim,use_consistency=True):
    # 1) Recreate Unet + RectifiedFlowDerain with same hyperparams as training
    base_unet = Unet(dim=dim,channels=4)  # 如果训练时 dim=64
    model = RectifiedFlowDerain(base_unet,use_consistency=use_consistency)
    # 2) Load ckpt
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    # load VAE
    VAE = AutoencoderKL.from_pretrained(args.vae_model)
    VAE.to(device)
    
    # load model
    model = load_model(args.ckpt_path, device,dim=args.UnetDim,use_consistency=args.use_consistency)

    # prepare transform
    tf = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])
    # inv_tf = transforms.Compose([
    #     transforms.Resize(512)
    # ])

    # iterate through input images
    for img_name in os.listdir(args.input_dir):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            continue
        inp_path = os.path.join(args.input_dir, img_name)
        out_path = os.path.join(args.output_dir, img_name)

        # load and preprocess
        img = Image.open(inp_path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)  # shape [1,3,H,W]
        
        # VAE encoding
        latent_rainy = VAE.encode(x).latent_dist.sample()

        # run sampling (use rainy image as init)
        with torch.no_grad():
            clean_latent_pred = model.sample(rainy_image=latent_rainy,data_shape=latent_rainy.shape,steps=args.sampling_steps)


        # post-process and save  
        # pred = pred
        # VAE 
        clean_pred = VAE.decode(clean_latent_pred).sample.clamp(0., 1.)
        # pred = inv_tf(pred)

        save_image(clean_pred, out_path)   # 自动把[0,1]的tensor转为图

        print(f"Saved derained image to {out_path}")

if __name__ == "__main__":
    main()
