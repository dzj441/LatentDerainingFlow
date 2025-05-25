import wandb
from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn import Module, ModuleList
from ema_pytorch import EMA
import math
import torch
from torchvision.utils import save_image
from pathlib import Path
from functools import partial
from torch import nn
from torchdiffeq import odeint
import einx
from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange

from rectified_flow import RectifiedFlow,RectifiedFlowDerain,MSEWithFreqLoss
from torch.utils.data import Dataset 
from data import ImageDataset,PairedImageDataset
from diffusers import AutoencoderKL

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

def divisible_by(num, den):
    return (num % den) == 0


class DerainTrainer(Module):
    def __init__(
        self,
        rectified_flow: dict | RectifiedFlowDerain,
        *,
        dataset: dict | Dataset,
        num_train_steps = 150_0000,
        learning_rate = 1e-4,
        batch_size = 1,
        checkpoints_folder: str = './checkpoints',
        results_folder: str = './samples',
        save_results_every: int = 1_0000,
        checkpoint_every: int = 1_0000,
        log_every: int = 100,
        sample_temperature: float = 1.,
        num_samples: int = 1,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        use_ema = True,
        wandb_project: str = "my_project",  # Add the wandb project name
        wandb_entity: str = None           # Optional, specify your entity
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        # # 输出当前使用的设备
        # print(f"Using device: {self.device}")

        if isinstance(dataset, dict):
            dataset = ImageDataset(**dataset)

        if isinstance(rectified_flow, dict):
            rectified_flow = RectifiedFlow(**rectified_flow)

        self.model = rectified_flow

        # Initialize wandb
        self.wandb = wandb
        self.wandb.init(project=wandb_project, entity=wandb_entity)

        use_ema &= not getattr(self.model, 'use_consistency', False)

        self.use_ema = use_ema
        self.ema_model = None

        if self.is_main and use_ema:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                **ema_kwargs
            )



        # Optimizer, DataLoader setup
        self.optimizer = Adam(rectified_flow.parameters(), lr=learning_rate, **adam_kwargs)
        self.dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


        self.model, self.optimizer, self.dl,self.ema_model = self.accelerator.prepare(self.model, self.optimizer, self.dl,self.ema_model)

        self.num_train_steps = num_train_steps
        self.return_loss_breakdown = isinstance(rectified_flow, int) # WON‘T use breakdown as is not tested

        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        self.checkpoints_folder.mkdir(exist_ok=True, parents=True)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every
        self.log_every = log_every
        self.sample_temperature = sample_temperature

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples

        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        # Always save the main model and optimizer
        save_package = {
            'model':     self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer': self.accelerator.unwrap_model(self.optimizer).state_dict(),
        }
        # Only save EMA model if it exists
        if self.ema_model is not None:
            save_package['ema_model'] = self.accelerator.unwrap_model(self.ema_model).state_dict()

        torch.save(save_package, str(self.checkpoints_folder / path))

    def load(self, path):
        if not self.is_main:
            return

        load_package = torch.load(path)

        # Restore main model parameters
        self.model.load_state_dict(load_package['model'])

        # Restore EMA model parameters if both the checkpoint and trainer have one
        if 'ema_model' in load_package and self.ema_model is not None:
            self.ema_model.load_state_dict(load_package['ema_model'])

        # Restore optimizer state
        self.optimizer.load_state_dict(load_package['optimizer'])


    def log(self, *args, **kwargs):
        return self.wandb.log(*args, **kwargs)

    def log_images(self, images: torch.Tensor, name: str, step: int):
        """
        始终以 dict 形式调用 wandb.log
        """
        if not self.is_main:
            return
        self.wandb.log({name: self.wandb.Image(images)}, step=step)
        
    def sample(self,
            rainy_image: torch.Tensor,
            fname: str = None,
            steps: int = 16,
            temperature: float = 1.):
        eval_model = default(self.ema_model, self.model)
        device = next(eval_model.parameters()).device

        # 1. 移到 device 并归一化
        x0 = self.model.data_normalize_fn(rainy_image.to(device))

        # 2. 时间张量也在同一个 device、dtype
        times = torch.linspace(0., 1., steps, device=device, dtype=x0.dtype)

        # 3. 定义 ODE 函数，包含噪声网络的 temperature 处理
        def ode_fn(t, x):
            t_mapped = self.model.noise_schedule(t)
            output, flow = self.model.predict_flow(eval_model, x, times=t_mapped)
            if self.model.mean_variance_net:
                mean, var = output
                flow = torch.normal(mean, var * temperature)
            return flow

        # 4. 求解
        trajectory = odeint(ode_fn, x0, times, **self.model.odeint_kwargs)

        # 5. 得到去雨结果，反归一化并 clamp
        clean_pred = self.model.data_unnormalize_fn(trajectory[-1]).clamp(0., 1.)
        # clean_pred = (trajectory[-1]).clamp(0., 1.)

        # 6. 可选：保存 & log
        if fname:
            save_image(rainy_image, fname + "_rainy.png")
            save_image(clean_pred,  fname + "_pred.png")
        if self.is_main:
            self.wandb.log({"sampled_image": self.wandb.Image(clean_pred)})

        return clean_pred


    def forward(self):

        dl = cycle(self.dl)

        # 在进入训练循环前初始化
        running_loss = 0.0
        running_mse = 0.0
        running_freq = 0.0
        log_interval = 100

        for ind in range(self.num_train_steps):
            step = ind + 1

            self.model.train()

            # Get the data pair (rainy_image, clean_image)
            data = next(dl)
            rainy_image, clean_image = data  # Unpack the pair
            if self.return_loss_breakdown:
                loss, breakdown = self.model(
                    rainy_image,
                    clean_image,
                    return_loss_breakdown=True
                )
                if self.is_main and divisible_by(step, self.log_every):
                    self.log(breakdown._asdict(), step=step)
            else:
                loss = self.model(rainy_image, clean_image)

            running_loss += loss.item()
            lf = getattr(self.model, 'loss_fn', None)
            if isinstance(lf, MSEWithFreqLoss):
                running_mse += lf.last_mse.item()
                running_freq += lf.last_freq.item()
            
            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                msg = f'[{step}] avg loss: {avg_loss:.3f}'

                log_dict = {"train_loss": avg_loss}
                # 如果是 mse_with_freq，还要算 avg_mse 和 avg_freq
                if isinstance(lf, MSEWithFreqLoss):
                    avg_mse = running_mse / log_interval
                    avg_freq = running_freq / log_interval
                    msg += f' | avg mse: {avg_mse:.3f} | avg freq: {avg_freq:.3f}'
                    log_dict.update({
                        "train_mse": avg_mse,
                        "train_freq": avg_freq
                    })

                self.accelerator.print(msg)
                if self.is_main:
                    self.wandb.log(log_dict, step=step)

                running_loss = 0.0
                running_mse = 0.0
                running_freq = 0.0

            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Log training loss to wandb（如需改为 avg_loss，可在此修改）
            if self.is_main:
                self.wandb.log({"train_loss": loss.item()})

            if getattr(self.model, 'use_consistency', False):
                self.model.ema_model.update()

            if self.is_main and self.use_ema:
                self.ema_model.ema_model.data_shape = self.model.data_shape
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            if self.is_main:
                if divisible_by(step, self.save_results_every):
                    print(f"[{step}] sampling......")
                    data = next(dl)
                    rainy_image, clean_image = data
                    sampled = self.sample(rainy_image=rainy_image, fname=str(self.results_folder / f'{step}_'))

                if divisible_by(step, self.checkpoint_every):
                    print(f"[{step}] saving ckpt ........")
                    self.save(f'checkpoint.{step}.pt')

            self.accelerator.wait_for_everyone()

        print('training complete')
        if self.is_main:
            self.wandb.finish()
            
class LatentDerainTrainer(Module):
    def __init__(
        self,
        vae: AutoencoderKL,
        rectified_flow: dict | RectifiedFlowDerain,
        *,
        dataset: dict | Dataset,
        num_train_steps = 150_0000,
        learning_rate = 1e-4,
        batch_size = 1,
        checkpoints_folder: str = './checkpoints',
        results_folder: str = './samples',
        save_results_every: int = 1_0000,
        checkpoint_every: int = 1_0000,
        log_every: int = 100,
        sample_temperature: float = 1.,
        num_samples: int = 1,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        use_ema = True,
        wandb_project: str = "my_project",  # Add the wandb project name
        wandb_entity: str = None           # Optional, specify your entity
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        # # 输出当前使用的设备
        # print(f"Using device: {self.device}")

        if isinstance(dataset, dict):
            dataset = ImageDataset(**dataset)

        if isinstance(rectified_flow, dict):
            rectified_flow = RectifiedFlow(**rectified_flow)

        self.vae = vae
        self.model = rectified_flow

        self.vae.eval()
        
        # Initialize wandb
        self.wandb = wandb
        self.wandb.init(project=wandb_project, entity=wandb_entity)

        use_ema &= not getattr(self.model, 'use_consistency', False)

        self.use_ema = use_ema
        self.ema_model = None

        if self.is_main and use_ema:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                **ema_kwargs
            )



        # Optimizer, DataLoader setup
        self.optimizer = Adam(rectified_flow.parameters(), lr=learning_rate, **adam_kwargs)
        self.dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


        self.vae, self.model, self.optimizer, self.dl, self.ema_model = self.accelerator.prepare(
            self.vae, self.model, self.optimizer, self.dl, self.ema_model
        )

        self.num_train_steps = num_train_steps
        self.return_loss_breakdown = isinstance(rectified_flow, RectifiedFlow)

        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        self.checkpoints_folder.mkdir(exist_ok=True, parents=True)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every
        self.log_every = log_every
        self.sample_temperature = sample_temperature

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples

        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        # Always save the main model and optimizer
        save_package = {
            'model':     self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer': self.accelerator.unwrap_model(self.optimizer).state_dict(),
        }
        # Only save EMA model if it exists
        if self.ema_model is not None:
            save_package['ema_model'] = self.accelerator.unwrap_model(self.ema_model).state_dict()

        torch.save(save_package, str(self.checkpoints_folder / path))

    def load(self, path):
        if not self.is_main:
            return

        load_package = torch.load(path)

        # Restore main model parameters
        self.model.load_state_dict(load_package['model'])

        # Restore EMA model parameters if both the checkpoint and trainer have one
        if 'ema_model' in load_package and self.ema_model is not None:
            self.ema_model.load_state_dict(load_package['ema_model'])

        # Restore optimizer state
        self.optimizer.load_state_dict(load_package['optimizer'])

    def log(self, *args, **kwargs):
        return self.wandb.log(*args, **kwargs)

    def log_images(self, images: torch.Tensor, name: str, step: int):
        """
        始终以 dict 形式调用 wandb.log
        """
        if not self.is_main:
            return
        self.wandb.log({name: self.wandb.Image(images)}, step=step)
        
    def sample(self,
            rainy_image: torch.Tensor,
            fname: str = None,
            steps: int = 16,
            temperature: float = 1.):
        eval_model = default(self.ema_model, self.model)
        device = next(eval_model.parameters()).device
        
        latent_rainy = self.vae.encode(rainy_image.to(device)).latent_dist.mode()

        # 1. 移到 device 并归一化
        x0 = self.model.data_normalize_fn(latent_rainy)

        # 2. 时间张量也在同一个 device、dtype
        times = torch.linspace(0., 1., steps, device=device, dtype=x0.dtype)

        # 3. 定义 ODE 函数，包含噪声网络的 temperature 处理
        def ode_fn(t, x):
            t_mapped = self.model.noise_schedule(t)
            output, flow = self.model.predict_flow(eval_model, x, times=t_mapped)
            if self.model.mean_variance_net:
                mean, var = output
                flow = torch.normal(mean, var * temperature)
            return flow

        # 4. 求解
        trajectory = odeint(ode_fn, x0, times, **self.model.odeint_kwargs)

        # 5. 得到去雨结果，反归一化并 clamp
        clean_latent_pred = self.model.data_unnormalize_fn(trajectory[-1])
        clean_pred = self.vae.decode(clean_latent_pred).sample.clamp(0., 1.)

        # 6. 可选：保存 & log
        if fname:
            save_image(rainy_image, fname + "_rainy.png")
            save_image(clean_pred,  fname + "_pred.png")
        if self.is_main:
            self.wandb.log({"sampled_image": self.wandb.Image(clean_pred)})

        return clean_pred


    def forward(self):

        dl = cycle(self.dl)

        # 在进入训练循环前初始化
        running_loss = 0.0
        running_mse = 0.0
        running_freq = 0.0
        log_interval = 100

        for ind in range(self.num_train_steps):
            step = ind + 1

            self.model.train()

            # Get the data pair (rainy_image, clean_image)
            data = next(dl)
            rainy_image, clean_image = data  # Unpack the pair
            
            latent_rainy = self.vae.encode(rainy_image).latent_dist.mode()
            latent_clean = self.vae.encode(clean_image).latent_dist.mode()
            
            if self.return_loss_breakdown:
                loss, breakdown = self.model(
                    latent_rainy,
                    latent_clean,
                    return_loss_breakdown=True
                )
                if self.is_main and divisible_by(step, self.log_every):
                    self.log(breakdown._asdict(), step=step)
            else:
                loss = self.model(latent_rainy, latent_clean)

            running_loss += loss.item()
            lf = getattr(self.model, 'loss_fn', None)
            if isinstance(lf, MSEWithFreqLoss):
                running_mse += lf.last_mse.item()
                running_freq += lf.last_freq.item()
            
            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                msg = f'[{step}] avg loss: {avg_loss:.3f}'

                log_dict = {"train_loss": avg_loss}
                # 如果是 mse_with_freq，还要算 avg_mse 和 avg_freq
                if isinstance(lf, MSEWithFreqLoss):
                    avg_mse = running_mse / log_interval
                    avg_freq = running_freq / log_interval
                    msg += f' | avg mse: {avg_mse:.3f} | avg freq: {avg_freq:.3f}'
                    log_dict.update({
                        "train_mse": avg_mse,
                        "train_freq": avg_freq
                    })

                self.accelerator.print(msg)
                if self.is_main:
                    self.wandb.log(log_dict, step=step)

                running_loss = 0.0
                running_mse = 0.0
                running_freq = 0.0

            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Log training loss to wandb（如需改为 avg_loss，可在此修改）
            if self.is_main:
                self.wandb.log({"train_loss": loss.item()})

            if getattr(self.model, 'use_consistency', False):
                self.model.ema_model.update()

            if self.is_main and self.use_ema:
                self.ema_model.ema_model.data_shape = self.model.data_shape
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            if self.is_main:
                if divisible_by(step, self.save_results_every):
                    print(f"[{step}] sampling......")
                    data = next(dl)
                    rainy_image, clean_image = data
                    sampled = self.sample(rainy_image=rainy_image, fname=str(self.results_folder / f'{step}_'))

                if divisible_by(step, self.checkpoint_every):
                    print(f"[{step}] saving ckpt ........")
                    self.save(f'checkpoint.{step}.pt')

            self.accelerator.wait_for_everyone()

        print('training complete')
        if self.is_main:
            self.wandb.finish()

class ResidualLatentDerainTrainer(LatentDerainTrainer):

    def forward(self):
        dl = cycle(self.dl)
        running_loss = 0.0
        running_mse = 0.0
        running_freq = 0.0
        
        log_interval = self.log_every

        for ind in range(self.num_train_steps):
            step = ind + 1
            self.model.train()

            rainy_image, clean_image = next(dl)
            # 编码为潜变量
            latent_rainy = self.vae.encode(rainy_image).latent_dist.mode()
            latent_clean = self.vae.encode(clean_image).latent_dist.mode()
            # 构造残差标签
            latent_residual = latent_rainy - latent_clean

            if self.return_loss_breakdown:
                loss, breakdown = self.model(
                    latent_rainy,
                    latent_residual,
                    return_loss_breakdown=True
                )
                if self.is_main and divisible_by(step, self.log_every):
                    self.log(breakdown._asdict(), step=step)
            else:
                loss = self.model(latent_rainy, latent_residual)

            running_loss += loss.item()
            lf = getattr(self.model, 'loss_fn', None)
            if isinstance(lf, MSEWithFreqLoss):
                running_mse += lf.last_mse.item()
                running_freq += lf.last_freq.item()
            
            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                msg = f'[{step}] avg loss: {avg_loss:.3f}'

                log_dict = {"train_loss": avg_loss}
                # 如果是 mse_with_freq，还要算 avg_mse 和 avg_freq
                if isinstance(lf, MSEWithFreqLoss):
                    avg_mse = running_mse / log_interval
                    avg_freq = running_freq / log_interval
                    msg += f' | avg mse: {avg_mse:.3f} | avg freq: {avg_freq:.3f}'
                    log_dict.update({
                        "train_mse": avg_mse,
                        "train_freq": avg_freq
                    })

                self.accelerator.print(msg)
                if self.is_main:
                    self.wandb.log(log_dict, step=step)

                running_loss = 0.0
                running_mse = 0.0
                running_freq = 0.0

            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.is_main:
                self.wandb.log({"train_loss": loss.item()}, step=step)

            if self.use_ema:
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            if self.is_main and divisible_by(step, self.save_results_every):
                self.accelerator.print(f'[{step}] sampling……')
                sampled = self.sample(rainy_image=rainy_image,
                                      fname=str(self.results_folder / f'{step}_'))

            if self.is_main and divisible_by(step, self.checkpoint_every):
                self.accelerator.print(f'[{step}] saving ckpt……')
                self.save(f'checkpoint.{step}.pt')

        if self.is_main:
            self.wandb.finish()
        print('training complete')

    def sample(
        self,
        rainy_image: torch.Tensor,
        fname: str = None,
        steps: int = 16,
        temperature: float = 1.0,
    ) -> torch.Tensor:

        eval_model = getattr(self, 'ema_model', None) or self.model
        device = next(eval_model.parameters()).device


        latent_rainy = self.vae.encode(rainy_image.to(device)).latent_dist.mode()
        x0 = self.model.data_normalize_fn(latent_rainy)


        times = torch.linspace(0., 1., steps, device=device, dtype=x0.dtype)

        def ode_fn(t, x):
            t_mapped = self.model.noise_schedule(t)
            output, flow = self.model.predict_flow(eval_model, x, times=t_mapped)
            if self.model.mean_variance_net:
                mean, var = output
                flow = torch.normal(mean, var * temperature)
            return flow

        # 5) ODE 求解得到残差轨迹
        trajectory = odeint(ode_fn, x0, times, **self.model.odeint_kwargs)
        predicted_residual = self.model.data_unnormalize_fn(trajectory[-1])

        # 6) 还原 clean latent 并解码
        clean_latent_pred = latent_rainy - predicted_residual
        clean_pred = self.vae.decode(clean_latent_pred).sample.clamp(0., 1.)

        # 7) 保存到磁盘 & WandB 记录
        if fname:
            save_image(rainy_image, fname + "_rainy.png")
            save_image(clean_pred,    fname + "_pred.png")
        if self.is_main:
            self.wandb.log({"sampled_image": self.wandb.Image(clean_pred)})

        return clean_pred
