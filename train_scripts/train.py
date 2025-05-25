import argparse
import sys
import os
os.environ["HF_HOME"] = "https://hf-mirror.com"
sys.path.append(os.getcwd())

import wandb
from data import PairedImageDataset
from rectified_flow import RectifiedFlowDerain, Unet
from trainer import DerainTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train the derain model on pixel space")

    # wandb 配置
    parser.add_argument('--wandb_project',    type=str, default="my_project",    help="W&B project name")
    parser.add_argument('--wandb_entity',     type=str, default=None,            help="W&B entity (team/user)")

    # 训练超参
    parser.add_argument('--num_train_steps',  type=int,   default=100000,        help="Total training steps")
    parser.add_argument('--learning_rate',    type=float, default=1e-5,          help="Optimizer learning rate")
    parser.add_argument('--batch_size',       type=int,   default=1,             help="Batch size")
    parser.add_argument('--checkpoint_every', type=int,   default=10000,          help="Steps between saving checkpoints")
    parser.add_argument('--save_results_every', type=int, default=10000,         help="Steps between saving sample outputs")

    # 数据与模型设置
    parser.add_argument('--image_size',       type=int,   default=128,           help="Input/output image size")
    parser.add_argument('--dataset_folder',   type=str,   default='datasets/hybrid',
                                                     help="Path to paired train dataset")
    parser.add_argument('--num_samples',      type=int,   default=1,             help="How many samples to generate when saving results")

    # Unet 与损失
    parser.add_argument('--UnetDim',          type=int,   default=16,            help="Base channel dimension for U-Net")
    parser.add_argument('--loss_fn',          type=str,   default='mse_with_freq',
                                                     choices=['mse','pseudo','pseudo_w_lpips','mse_with_freq'],
                                                     help="Loss function")
    parser.add_argument('--use_consistency', type=bool, default=True, help="whether use cfm")
    # 文件输出路径
    parser.add_argument('--checkpoint_folder', type=str, default='./pixel_derain_checkpoints',
                                                      help="Directory to save checkpoints")
    parser.add_argument('--result_folder',     type=str, default='./pixel_derain_results',
                                                      help="Directory to save sample outputs")

    return parser.parse_args()

def main():
    args = parse_args()  # 解析命令行参数

    # 创建模型和 rectified flow 实例
    # 构建模型：直接在 RGB 像素空间操作
    model = Unet(dim=args.UnetDim, channels=3)
    rectified_flow = RectifiedFlowDerain(
        model=model,
        loss_fn=args.loss_fn,
        use_consistency=args.use_consistency
    )
    print("Model and rectified flow initialized.")

    # 数据集
    img_dataset = PairedImageDataset(
        folder=args.dataset_folder,
        image_size=args.image_size
    )
    print("Paired dataset loaded.")

    # 训练器
    trainer = DerainTrainer(
        rectified_flow=rectified_flow,
        dataset=img_dataset,
        num_train_steps=args.num_train_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        save_results_every=args.save_results_every,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        checkpoints_folder=args.checkpoint_folder,
        results_folder=args.result_folder,
        num_samples=args.num_samples
    )
    print(f"Trainer initialized on device: {trainer.model.device}")
    print(f"Checkpoints → {args.checkpoint_folder}, Results → {args.result_folder}")

    # 开始训练
    trainer()

if __name__ == "__main__":
    main()
