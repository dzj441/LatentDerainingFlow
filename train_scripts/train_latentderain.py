import argparse
import sys
import os
os.environ["HF_HOME"] = "https://hf-mirror.com"
sys.path.append(os.getcwd())

import wandb
from data import PairedImageDataset
from rectified_flow import RectifiedFlowDerain, Unet
from trainer import LatentDerainTrainer
from diffusers import  AutoencoderKL

def parse_args():
    parser = argparse.ArgumentParser(description="Train the derain model")
    
    # 使用命令行指定配置文件


    # 其他命令行参数
    parser.add_argument('--wandb_project', type=str, default="my_project", help="Wandb project name")
    parser.add_argument('--wandb_entity', type=str, default=None, help="Wandb entity")
    parser.add_argument('--num_train_steps', type=int, default=100000, help="Number of training steps")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--checkpoint_every', type=int, default=5000, help="Checkpoint interval")
    parser.add_argument('--save_results_every', type=int, default=5000, help="Save results interval")
    parser.add_argument('--image_size', type=int, default=256, help="Size of the images")
    parser.add_argument('--dataset_folder', type=str, default="datasets/hybrid", help="Path to the dataset folder")
    parser.add_argument('--num_samples', type=int, default=1, help="Number of samples to generate")
    parser.add_argument('--vae_model',type=str,default="sd2-vae",help="VAE model to use, default is 'stabilityai/sdxl-vae'") # "stabilityai/sdxl-vae" if from remote
    parser.add_argument('--checkpoint_folder', type=str, default='./latent_derain_checkpoints/SD2vae_checkpoints', help="Path to save checkpoints")
    parser.add_argument('--result_folder', type=str, default='./latent_derain_samples/SD2vae_samples', help="Path to save generated samples")
    parser.add_argument('--UnetDim', type=int, default=16, help="UnetInitDim")
    parser.add_argument('--loss_fn', type=str, default='mse_with_freq', help="loss type :  mse | pseudo | psrudo wiz lpips | mse_with_freq")
    parser.add_argument('--use_consistency', type=bool, default=True, help="whether use cfm")
            
    return parser.parse_args()

def main():
    args = parse_args()  # 解析命令行参数

    VAE = AutoencoderKL.from_pretrained(args.vae_model)
    
    # 创建模型和 rectified flow 实例
    model = Unet(dim=args.UnetDim,channels=4) # for vae latents
    rectified_flow = RectifiedFlowDerain(
        model = model,
        loss_fn = args.loss_fn,
        use_consistency= args.use_consistency
    )

    print("model built successfully")


    # 创建数据集实例
    img_dataset = PairedImageDataset(
        folder=args.dataset_folder,
        image_size=args.image_size
    )
    print("paired dataset built successfully")

    # 创建训练实例
    trainer = LatentDerainTrainer(
        vae = VAE,
        rectified_flow=rectified_flow,
        dataset=img_dataset,
        num_train_steps=args.num_train_steps,
        checkpoint_every=args.checkpoint_every,
        save_results_every=args.save_results_every,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        checkpoints_folder=args.checkpoint_folder,
        results_folder=args.result_folder
    )
    print(f"checkpoint_every={args.checkpoint_every}, save_results_every={args.save_results_every},")
    print("trainer built successfully")
    print(trainer.model.device)
    
    # 启动训练
    trainer()

if __name__ == "__main__":
    main()
