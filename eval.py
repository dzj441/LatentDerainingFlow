import os
import argparse
import logging
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def load_image(path):
    """使用 Pillow 读取并转换为 NumPy 数组（RGB 或灰度）"""
    with Image.open(path) as img:
        return np.array(img)


def calculate_psnr(img1, img2):
    return psnr(img1, img2)


def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2)


def compare_images(dir1, dir2):
    psnr_values = []
    ssim_values = []

    names1 = os.listdir(dir1)
    names2 = set(os.listdir(dir2))

    for name in names1:
        if name not in names2:
            logging.warning(f"{name} not found in {dir2}, skipping.")
            continue

        img1 = load_image(os.path.join(dir1, name))
        img2 = load_image(os.path.join(dir2, name))

        if img1.shape != img2.shape:
            logging.warning(
                f"Skipping {name}: shape mismatch {img1.shape} vs {img2.shape}."
            )
            continue

        p = calculate_psnr(img1, img2)
        s = calculate_ssim(img1, img2)

        psnr_values.append(p)
        ssim_values.append(s)

        # logging.info(f"Comparing {name} – PSNR: {p:.2f}, SSIM: {s:.4f}")

    return psnr_values, ssim_values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two image folders and log PSNR/SSIM (incremental log)."
    )
    parser.add_argument("--dir1",    type=str, required=True,
                        help="第一组图片目录（原图）")
    parser.add_argument("--dir2",    type=str, required=True,
                        help="第二组图片目录（复原图）")
    parser.add_argument("--log_file", type=str, default="comparison.log",
                        help="结果追加写入的日志文件")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode='a',   # 追加模式
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    logging.info("========== 新一轮图像比较开始 ==========")
    logging.info(f"目录1: {args.dir1}")
    logging.info(f"目录2: {args.dir2}")

    psnr_vals, ssim_vals = compare_images(args.dir1, args.dir2)

    if psnr_vals and ssim_vals:
        avg_p = np.mean(psnr_vals)
        avg_s = np.mean(ssim_vals)
        logging.info(f"平均 PSNR: {avg_p:.2f}")
        logging.info(f"平均 SSIM: {avg_s:.4f}")
    else:
        logging.warning("未比较任何有效图像对，无法计算平均值。")

    logging.info("========== 本轮比较结束 ==========\n")


if __name__ == "__main__":
    main()
