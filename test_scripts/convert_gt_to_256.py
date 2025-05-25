import argparse
import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path

from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

def parse_args():
    parser = argparse.ArgumentParser("Preprocess images and save as tensors")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with input images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save processed tensor images")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Size to which test images are resized")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing (not used in this version)")
    
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare the transformation pipeline
    tf = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])

    # Iterate through input images
    for img_name in os.listdir(args.input_dir):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            continue
        
        inp_path = os.path.join(args.input_dir, img_name)
        out_path = os.path.join(args.output_dir, img_name)

        # Load and preprocess the image
        img = Image.open(inp_path).convert("RGB")
        x = tf(img).unsqueeze(0)  # shape [1, 3, H, W] - add batch dimension

        # Save the processed tensor as an image
        save_image(x, out_path)

        print(f"Saved preprocessed image tensor to {out_path}")

if __name__ == "__main__":
    main()
