from __future__ import annotations

import math
from copy import deepcopy
from collections import namedtuple
from typing import Literal, Callable

import torch
from torch import Tensor
from torch import nn, pi, cat, stack, from_numpy
from torch.nn import Module, ModuleList
from torch.distributions import Normal
import torch.nn.functional as F

from torchdiffeq import odeint

import torchvision
from torchvision.utils import save_image
from torchvision.models import VGG16_Weights

import einx
from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange

from hyper_connections.hyper_connections_channel_first import get_init_and_expand_reduce_stream_functions, Residual

from scipy.optimize import linear_sum_assignment
from pathlib import Path

from focal_frequency_loss import FocalFrequencyLoss as FFL
# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

# normalizing helpers

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# noise schedules

def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))

# losses

class LPIPSLoss(Module):
    def __init__(
        self,
        vgg: Module | None = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
    ):
        super().__init__()

        if not exists(vgg):
            vgg = torchvision.models.vgg16(weights = vgg_weights)
            vgg.classifier = nn.Sequential(*vgg.classifier[:-2])

        self.vgg = [vgg]

    def forward(self, pred_data, data, reduction = 'mean'):
        vgg, = self.vgg
        vgg = vgg.to(data.device)

        pred_embed, embed = map(vgg, (pred_data, data))

        loss = F.mse_loss(embed, pred_embed, reduction = reduction)

        if reduction == 'none':
            loss = reduce(loss, 'b ... -> b', 'mean')

        return loss

class PseudoHuberLoss(Module):
    def __init__(self, data_dim: int = 3):
        super().__init__()
        self.data_dim = data_dim

    def forward(self, pred, target, reduction = 'mean', **kwargs):
        data_dim = default(self.data_dim, kwargs.pop('data_dim', None))

        c = .00054 * self.data_dim
        loss = (F.mse_loss(pred, target, reduction = reduction) + c * c).sqrt() - c

        if reduction == 'none':
            loss = reduce(loss, 'b ... -> b', 'mean')

        return loss

class PseudoHuberLossWithLPIPS(Module):
    def __init__(self, data_dim: int = 3, lpips_kwargs: dict = dict()):
        super().__init__()
        self.pseudo_huber = PseudoHuberLoss(data_dim)
        self.lpips = LPIPSLoss(**lpips_kwargs)

    def forward(self, pred_flow, target_flow, *, pred_data, times, data):
        huber_loss = self.pseudo_huber(pred_flow, target_flow, reduction = 'none')
        lpips_loss = self.lpips(data, pred_data, reduction = 'none')

        time_weighted_loss = huber_loss * (1 - times) + lpips_loss * (1. / times.clamp(min = 1e-1))
        return time_weighted_loss.mean()

class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)

class MeanVarianceNetLoss(Module):
    def forward(self, pred, target, **kwargs):
        dist = Normal(*pred)
        return -dist.log_prob(target).mean()

class MSEWithFreqLoss(nn.Module):
    def __init__(self, freq_weight: float = 1.0, ffl_kwargs: dict | None = None):
        super().__init__()
        # prepare FFL kwargs
        ffl_kwargs = {} if ffl_kwargs is None else dict(ffl_kwargs)
        ffl_kwargs['loss_weight'] = freq_weight
        ffl_kwargs.setdefault('alpha', 1.0)
        self.ffl = FFL(**ffl_kwargs)

        self.last_mse = None
        self.last_freq = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        mse_loss = F.mse_loss(pred, target)
        freq_loss = self.ffl(pred, target)

        self.last_mse = mse_loss.detach()
        self.last_freq = freq_loss.detach()

        return mse_loss + freq_loss
    
# loss breakdown

LossBreakdown = namedtuple('LossBreakdown', ['total', 'main', 'data_match', 'velocity_match'])

# main class

class RectifiedFlow(Module):
    def __init__(
        self,
        model: dict | Module,
        mean_variance_net: bool | None = None,
        time_cond_kwarg: str | None = 'times',
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        predict: Literal['flow', 'noise'] = 'flow',
        loss_fn: Literal[
            'mse',
            'pseudo_huber',
            'pseudo_huber_with_lpips'
        ] | Module = 'mse',
        noise_schedule: Literal[
            'cosmap'
        ] | Callable = identity,
        loss_fn_kwargs: dict = dict(),
        ema_update_after_step: int = 100,
        ema_kwargs: dict = dict(),
        data_shape: tuple[int, ...] | None = None,
        immiscible = False,
        use_consistency = False,
        consistency_decay = 0.9999,
        consistency_velocity_match_alpha = 1e-5,
        consistency_delta_time = 1e-3,
        consistency_loss_weight = 1.,
        data_normalize_fn = normalize_to_neg_one_to_one,
        data_unnormalize_fn = unnormalize_to_zero_to_one,
        clip_during_sampling = False,
        clip_values: tuple[float, float] = (-1., 1.),
        clip_flow_during_sampling = None, # this seems to help a lot when training with predict epsilon, at least for me
        clip_flow_values: tuple[float, float] = (-3., 3)
    ):
        super().__init__()

        if isinstance(model, dict):
            model = Unet(**model)

        self.model = model
        self.time_cond_kwarg = time_cond_kwarg # whether the model is to be conditioned on the times

        # allow for mean variance output prediction

        if not exists(mean_variance_net):
            mean_variance_net = default(model.mean_variance_net if isinstance(model, Unet) else mean_variance_net, False)

        self.mean_variance_net = mean_variance_net

        if mean_variance_net:
            loss_fn = MeanVarianceNetLoss()

        # objective - either flow or noise (proposed by Esser / Rombach et al in SD3)

        self.predict = predict

        # automatically default to a working setting for predict epsilon

        clip_flow_during_sampling = default(clip_flow_during_sampling, predict == 'noise')

        # loss fn

        if loss_fn == 'mse':
            loss_fn = MSELoss()

        elif loss_fn == 'pseudo_huber':
            assert predict == 'flow'

            # section 4.2 of https://arxiv.org/abs/2405.20320v1
            loss_fn = PseudoHuberLoss(**loss_fn_kwargs)

        elif loss_fn == 'pseudo_huber_with_lpips':
            assert predict == 'flow'

            loss_fn = PseudoHuberLossWithLPIPS(**loss_fn_kwargs)

        elif not isinstance(loss_fn, Module):
            raise ValueError(f'unknown loss function {loss_fn}')

        self.loss_fn = loss_fn

        # noise schedules

        if noise_schedule == 'cosmap':
            noise_schedule = cosmap

        elif not callable(noise_schedule):
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        self.noise_schedule = noise_schedule

        # sampling

        self.odeint_kwargs = odeint_kwargs
        self.data_shape = data_shape

        # clipping for epsilon prediction

        self.clip_during_sampling = clip_during_sampling
        self.clip_flow_during_sampling = clip_flow_during_sampling

        self.clip_values = clip_values
        self.clip_flow_values = clip_flow_values

        # consistency flow matching

        self.use_consistency = use_consistency
        self.consistency_decay = consistency_decay
        self.consistency_velocity_match_alpha = consistency_velocity_match_alpha
        self.consistency_delta_time = consistency_delta_time
        self.consistency_loss_weight = consistency_loss_weight

        if use_consistency:
            self.ema_model = EMA(
                model,
                beta = consistency_decay,
                update_after_step = ema_update_after_step,
                include_online_model = False,
                **ema_kwargs
            )

        # immiscible diffusion paper, will be removed if does not work

        self.immiscible = immiscible

        # normalizing fn

        self.data_normalize_fn = default(data_normalize_fn, identity)
        self.data_unnormalize_fn = default(data_unnormalize_fn, identity)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict_flow(self, model: Module, noised, *, times, eps = 1e-10):
        """
        returns the model output as well as the derived flow, depending on the `predict` objective
        """

        batch = noised.shape[0]

        # prepare maybe time conditioning for model

        model_kwargs = dict()
        time_kwarg = self.time_cond_kwarg

        if exists(time_kwarg):
            times = rearrange(times, '... -> (...)')

            if times.numel() == 1:
                times = repeat(times, '1 -> b', b = batch)

            model_kwargs.update(**{time_kwarg: times})

        output = self.model(noised, **model_kwargs)

        # depending on objective, derive flow

        if self.predict == 'flow':
            flow = output

        elif self.predict == 'noise':
            noise = output
            padded_times = append_dims(times, noised.ndim - 1)

            flow = (noised - noise) / padded_times.clamp(min = eps)

        else:
            raise ValueError(f'unknown objective {self.predict}')

        return output, flow

    @torch.no_grad()
    def sample(
        self,
        batch_size = 1,
        steps = 16,
        noise = None,
        data_shape: tuple[int, ...] | None = None,
        temperature: float = 1.,
        use_ema: bool = False,
        **model_kwargs
    ):
        use_ema = default(use_ema, self.use_consistency)
        assert not (use_ema and not self.use_consistency), 'in order to sample from an ema model, you must have `use_consistency` turned on'

        model = self.ema_model if use_ema else self.model

        was_training = self.training
        self.eval()

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'you need to either pass in a `data_shape` or have trained at least with one forward'

        # clipping still helps for predict noise objective
        # much like original ddpm paper trick

        maybe_clip = (lambda t: t.clamp_(*self.clip_values)) if self.clip_during_sampling else identity

        maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_during_sampling else identity

        # ode step function

        def ode_fn(t, x):
            x = maybe_clip(x)

            _, output = self.predict_flow(model, x, times = t, **model_kwargs)

            flow = output

            if self.mean_variance_net:
                mean, variance = output

                variance = variance * temperature

                flow = torch.normal(mean, variance)

            flow = maybe_clip_flow(flow)

            return flow

        # start with random gaussian noise - y0

        noise = default(noise, torch.randn((batch_size, *data_shape), device = self.device))

        # time steps

        times = torch.linspace(0., 1., steps, device = self.device)

        # ode

        trajectory = odeint(ode_fn, noise, times, **self.odeint_kwargs)

        sampled_data = trajectory[-1]

        self.train(was_training)

        return self.data_unnormalize_fn(sampled_data)

    def forward(
        self,
        data,
        noise: Tensor | None = None,
        return_loss_breakdown = False,
        **model_kwargs
    ):
        batch, *data_shape = data.shape

        data = self.data_normalize_fn(data)

        self.data_shape = default(self.data_shape, data_shape)

        # x0 - gaussian noise, x1 - data

        noise = default(noise, torch.randn_like(data))

        # maybe immiscible flow

        if self.immiscible:
            cost = torch.cdist(data.flatten(1), noise.flatten(1))
            _, reorder_indices = linear_sum_assignment(cost.cpu())
            noise = noise[from_numpy(reorder_indices).to(cost.device)]

        # times, and times with dimension padding on right

        times = torch.rand(batch, device = self.device)
        padded_times = append_dims(times, data.ndim - 1)

        # time needs to be from [0, 1 - delta_time] if using consistency loss

        if self.use_consistency:
            padded_times *= 1. - self.consistency_delta_time

        def get_noised_and_flows(model, t):

            # maybe noise schedule

            t = self.noise_schedule(t)

            # Algorithm 2 in paper
            # linear interpolation of noise with data using random times
            # x1 * t + x0 * (1 - t) - so from noise (time = 0) to data (time = 1.)

            noised = t * data + (1. - t) * noise

            # the model predicts the flow from the noised data

            flow = data - noise

            model_output, model_output = self.predict_flow(model, noised, times = t)

            # if mean variance network, sample from normal

            pred_flow = model_output

            if self.mean_variance_net:
                mean, variance = model_output
                pred_flow = torch.normal(mean, variance)

            # predicted data will be the noised xt + flow * (1. - t)

            pred_data = noised + pred_flow * (1. - t)

            return model_output, flow, pred_flow, pred_data

        # getting flow and pred flow for main model

        output, flow, pred_flow, pred_data = get_noised_and_flows(self.model, padded_times)

        # if using consistency loss, also need the ema model predicted flow

        if self.use_consistency:
            delta_t = self.consistency_delta_time
            ema_output, ema_flow, ema_pred_flow, ema_pred_data = get_noised_and_flows(self.ema_model, padded_times + delta_t)

        # determine target, depending on objective

        if self.predict == 'flow':
            target = flow
        elif self.predict == 'noise':
            target = noise
        else:
            raise ValueError(f'unknown objective {self.predict}')

        # losses

        main_loss = self.loss_fn(output, target, pred_data = pred_data, times = times, data = data)

        consistency_loss = data_match_loss = velocity_match_loss = 0.

        if self.use_consistency:
            # consistency losses from consistency fm paper - eq (6) in https://arxiv.org/html/2407.02398v1

            data_match_loss = F.mse_loss(pred_data, ema_pred_data)
            velocity_match_loss = F.mse_loss(pred_flow, ema_pred_flow)

            consistency_loss = data_match_loss + velocity_match_loss * self.consistency_velocity_match_alpha

        # total loss

        total_loss = main_loss + consistency_loss * self.consistency_loss_weight

        if not return_loss_breakdown:
            return total_loss

        # loss breakdown

        return total_loss, LossBreakdown(total_loss, main_loss, data_match_loss, velocity_match_loss)

# unet

from functools import partial

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def divisible_by(num, den):
    return (num % den) == 0

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * (self.gamma + 1) * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = einx.multiply('i, j -> i j', x, emb)
        emb = cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = cat((x, fouriered), dim = -1)
        return fouriered

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = tuple(rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads) for t in qkv)

        mk, mv = tuple(repeat(t, 'h c n -> b h c n', b = b) for t in self.mem_kv)
        k, v = map(partial(cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = einsum(k, v, 'b h d n, b h e n -> b h d e')

        out = einsum(context, q, 'b h d e, b h d n -> b h e n')
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1, bias = False)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(cat, dim = -2), ((mk, k), (mv, v)))

        q = q * self.scale
        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        channels = 3,
        mean_variance_net = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False,
        num_residual_streams = 2
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # hyper connections

        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)
        res_conv = partial(nn.Conv2d, kernel_size = 1, bias = False)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                Residual(branch = resnet_block(dim_in, dim_in)),
                Residual(branch = resnet_block(dim_in, dim_in)),
                Residual(branch = attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads)),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = init_hyper_conn(dim = mid_dim, branch = resnet_block(mid_dim, mid_dim))
        self.mid_attn = init_hyper_conn(dim = mid_dim, branch = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1]))
        self.mid_block2 = init_hyper_conn(dim = mid_dim, branch = resnet_block(mid_dim, mid_dim))

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                Residual(branch = resnet_block(dim_out + dim_in, dim_out), residual_transform = res_conv(dim_out + dim_in, dim_out)),
                Residual(branch = resnet_block(dim_out + dim_in, dim_out), residual_transform = res_conv(dim_out + dim_in, dim_out)),
                Residual(branch = attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads)),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.mean_variance_net = mean_variance_net

        default_out_dim = channels * (1 if not mean_variance_net else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = Residual(branch = resnet_block(init_dim * 2, init_dim), residual_transform = res_conv(init_dim * 2, init_dim))
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, times):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        x = self.init_conv(x)

        r = x.clone()

        t = self.time_mlp(times)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.expand_streams(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        x = self.reduce_streams(x)

        for block1, block2, attn, upsample in self.ups:
            x = cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = cat((x, r), dim = 1)

        x = self.final_res_block(x, t)

        out = self.final_conv(x)

        if not self.mean_variance_net:
            return out

        mean, log_var = rearrange(out, 'b (c mean_log_var) h w -> mean_log_var b c h w', mean_log_var = 2)
        variance = log_var.exp() # variance needs to be positive
        return stack((mean, variance))

# dataset classes

# trainer

from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import DataLoader
from ema_pytorch import EMA

def cycle(dl):
    while True:
        for batch in dl:
            yield batch


class RectifiedFlowDerain(Module):
    def __init__(
        self,
        model: dict | Module,
        mean_variance_net: bool | None = None,
        time_cond_kwarg: str | None = 'times',
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        predict: Literal['flow', 'noise'] = 'flow',
        loss_fn: Literal[
            'mse',
            'pseudo_huber',
            'pseudo_huber_with_lpips',
            'mse_with_freq'
        ] | Module = 'mse',
        noise_schedule: Literal[
            'cosmap'
        ] | Callable = identity,
        loss_fn_kwargs: dict = dict(),
        ema_update_after_step: int = 100,
        ema_kwargs: dict = dict(),
        data_shape: tuple[int, ...] | None = None,
        immiscible = False,
        use_consistency = False,
        consistency_decay = 0.9999,
        consistency_velocity_match_alpha = 1e-5,
        consistency_delta_time = 1e-3,
        consistency_loss_weight = 1.,
        data_normalize_fn = normalize_to_neg_one_to_one,
        data_unnormalize_fn = unnormalize_to_zero_to_one,
        clip_during_sampling = False,
        clip_values: tuple[float, float] = (-1., 1.),
        clip_flow_during_sampling = None,
        clip_flow_values: tuple[float, float] = (-3., 3)
    ):
        super().__init__()

        if isinstance(model, dict):
            model = Unet(**model)

        self.model = model
        self.time_cond_kwarg = time_cond_kwarg

        if not exists(mean_variance_net):
            mean_variance_net = default(model.mean_variance_net if isinstance(model, Unet) else mean_variance_net, False)

        self.mean_variance_net = mean_variance_net

        if mean_variance_net:
            loss_fn = MeanVarianceNetLoss()

        self.predict = predict

        clip_flow_during_sampling = default(clip_flow_during_sampling, predict == 'noise')

        # Loss functions
        if loss_fn == 'mse':
            loss_fn = MSELoss()

        elif loss_fn == 'pseudo_huber':
            assert predict == 'flow'
            loss_fn = PseudoHuberLoss(**loss_fn_kwargs)

        elif loss_fn == 'pseudo_huber_with_lpips':
            assert predict == 'flow'
            loss_fn = PseudoHuberLossWithLPIPS(**loss_fn_kwargs)
        elif loss_fn == 'mse_with_freq':
            assert predict == 'flow'
            loss_fn = MSEWithFreqLoss()
        elif not isinstance(loss_fn, Module):
            raise ValueError(f'unknown loss function {loss_fn}')

        self.loss_fn = loss_fn

        # Noise schedules
        if noise_schedule == 'cosmap':
            noise_schedule = cosmap

        elif not callable(noise_schedule):
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        self.noise_schedule = noise_schedule

        # Sampling
        self.odeint_kwargs = odeint_kwargs
        self.data_shape = data_shape

        # Clipping
        self.clip_during_sampling = clip_during_sampling
        self.clip_flow_during_sampling = clip_flow_during_sampling
        self.clip_values = clip_values
        self.clip_flow_values = clip_flow_values

        # Consistency flow matching
        self.use_consistency = use_consistency
        print(f"using consistency: {self.use_consistency}")

        self.consistency_decay = consistency_decay
        self.consistency_velocity_match_alpha = consistency_velocity_match_alpha
        self.consistency_delta_time = consistency_delta_time
        self.consistency_loss_weight = consistency_loss_weight

        if use_consistency:
            self.ema_model = EMA(
                model,
                beta = consistency_decay,
                update_after_step = ema_update_after_step,
                include_online_model = False,
                **ema_kwargs
            )

        self.immiscible = immiscible

        # Normalization functions
        self.data_normalize_fn = default(data_normalize_fn, identity)
        self.data_unnormalize_fn = default(data_unnormalize_fn, identity)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict_flow(self, model: Module, rainy_image, *, times, eps = 1e-10):
        """
        Predicts the flow from rainy image to clean image (without rain).
        """

        batch = rainy_image.shape[0]

        # Prepare time conditioning for model if applicable
        model_kwargs = dict()
        time_kwarg = self.time_cond_kwarg

        if exists(time_kwarg):
            times = rearrange(times, '... -> (...)')

            if times.numel() == 1:
                times = repeat(times, '1 -> b', b = batch)

            model_kwargs.update(**{time_kwarg: times})

        # Get model's prediction
        output = self.model(rainy_image, **model_kwargs)

        # Based on the objective (flow or noise), derive flow
        if self.predict == 'flow':
            flow = output

        elif self.predict == 'noise':
            noise = output
            padded_times = append_dims(times, rainy_image.ndim - 1)
            flow = (rainy_image - noise) / padded_times.clamp(min = eps)

        else:
            raise ValueError(f'unknown objective {self.predict}')

        return output, flow
        
    def sample(
        self,
        batch_size = 1,
        steps = 16,
        rainy_image = None,
        data_shape: tuple[int, ...] | None = None,
        temperature: float = 1.,
        use_ema: bool = False,
        **model_kwargs
    ):
        use_ema = default(use_ema, self.use_consistency)
        assert not (use_ema and not self.use_consistency), 'in order to sample from an ema model, you must have `use_consistency` turned on'

        model = self.ema_model if use_ema else self.model

        was_training = self.training
        self.eval()

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'you need to either pass in a `data_shape` or have trained at least with one forward'

        # clipping still helps for predict noise objective
        # much like original ddpm paper trick

        maybe_clip = (lambda t: t.clamp_(*self.clip_values)) if self.clip_during_sampling else identity

        maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_during_sampling else identity

        # ode step function

        def ode_fn(t, x):
            x = maybe_clip(x)
            t = self.noise_schedule(t)
            _, output = self.predict_flow(model, x, times = t, **model_kwargs)

            flow = output

            if self.mean_variance_net:
                mean, variance = output

                variance = variance * temperature

                flow = torch.normal(mean, variance)

            flow = maybe_clip_flow(flow)

            return flow

        # Start with the rainy image (no need to initialize with random noise here)
        rainy_image = default(rainy_image, torch.randn((batch_size, *data_shape), device = self.device))

        rainy_image = self.data_normalize_fn(rainy_image)
        
        # Time steps
        times = torch.linspace(0., 1., steps, device = self.device)

        # ODE integration
        trajectory = odeint(ode_fn, rainy_image, times, **self.odeint_kwargs)

        sampled_data = trajectory[-1]

        self.train(was_training)

        # return sampled_data
        return self.data_unnormalize_fn(sampled_data)
    
    
    def forward(
        self,
        rainy_image,
        clean_image: Tensor | None = None,
        return_loss_breakdown = False,
        **model_kwargs
    ):
        batch, *data_shape = rainy_image.shape

        rainy_image = self.data_normalize_fn(rainy_image)

        self.data_shape = default(self.data_shape, data_shape)

        clean_image = default(clean_image, torch.randn_like(rainy_image))
        
        clean_image = self.data_normalize_fn(clean_image)

        times = torch.rand(batch, device=self.device)
        padded_times = append_dims(times, rainy_image.ndim - 1)

        if self.use_consistency:
            padded_times *= 1. - self.consistency_delta_time

        def get_noised_and_flows(model, t):
            t = self.noise_schedule(t)

            # Linear interpolation of rainy image (time=0) to clean image (time=1)
            noised = t * clean_image + (1. - t) * rainy_image

            flow = clean_image - rainy_image

            model_output, pred_flow = self.predict_flow(model, noised, times=t)


            if self.mean_variance_net:
                mean, variance = model_output
                pred_flow = torch.normal(mean, variance)

            pred_clean_image = noised + pred_flow * (1. - t)

            return model_output, flow, pred_flow, pred_clean_image

        output, flow, pred_flow, pred_clean_image = get_noised_and_flows(self.model, padded_times)

        if self.use_consistency:
            delta_t = self.consistency_delta_time
            ema_output, ema_flow, ema_pred_flow, ema_pred_clean_image = get_noised_and_flows(self.ema_model, padded_times + delta_t)

        # Loss calculation
        target = flow

        main_loss = self.loss_fn(output, target, pred_data=pred_clean_image, times=times, data=clean_image)

        consistency_loss = data_match_loss = velocity_match_loss = 0.

        if self.use_consistency:
            data_match_loss = F.mse_loss(pred_clean_image, ema_pred_clean_image)
            velocity_match_loss = F.mse_loss(pred_flow, ema_pred_flow)
            consistency_loss = data_match_loss + velocity_match_loss * self.consistency_velocity_match_alpha

        total_loss = main_loss + consistency_loss * self.consistency_loss_weight

        if not return_loss_breakdown:
            return total_loss

        return total_loss, LossBreakdown(total_loss, main_loss, data_match_loss, velocity_match_loss)



if __name__ == "__main__":
    pass