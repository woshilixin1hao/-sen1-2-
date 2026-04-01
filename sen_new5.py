# 基于 sen_new4.py，新增 Time-aware VGG 感知损失实现
# 主要改动：在 loss 中加入 VGG 感知损失，并对其做时间步感知加权，避免在 t 很大时强制模型输出完美图像
import os
import math
import time
import argparse
from tqdm import tqdm
from pathlib import Path
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torchvision.models as models
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import threading
import glob
import matplotlib.pyplot as plt
from inspect import isfunction
from einops.layers.torch import Rearrange
from einops import rearrange, reduce
from functools import partial
import random
import torchvision.transforms.functional as TF


class PairDataset(torch.utils.data.Dataset):
	def __init__(self, root_dir, transform=None, L_transform=None):
		self.root_dir = Path(root_dir)
		self.transform = transform
		self.L_transform = L_transform

		self.image_pairs = []
		for category in self.root_dir.iterdir():
			s1_path = category / "s1"
			s2_path = category / "s2"
			if not (s1_path.exists() and s2_path.exists()):
				continue
			for s1_file in s1_path.glob("*.png"):
				s2_file = s2_path / s1_file.name.replace("_s1_", "_s2_")
				if s2_file.exists():
					self.image_pairs.append((s1_file, s2_file))

		print(f"Found {len(self.image_pairs)} pairs.")

	def __len__(self):
		return len(self.image_pairs)

	def __getitem__(self, idx):
		s1_path, s2_path = self.image_pairs[idx]
		s1 = Image.open(s1_path).convert("L")     # SAR
		s2 = Image.open(s2_path).convert("RGB")   # Optical
		# ==========================================
		# 同步数据增强：确保 SAR 和 Optical 翻转方向绝对一致
		# 1. 50% 概率水平翻转 (左右)
		if random.random() > 0.5:
			s1 = TF.hflip(s1)
			s2 = TF.hflip(s2)
            
		# 2. 50% 概率垂直翻转 (上下)
		if random.random() > 0.5:
			s1 = TF.vflip(s1)
			s2 = TF.vflip(s2)
		# ==========================================
		if self.L_transform:
			s1 = self.L_transform(s1)
		if self.transform:
			s2 = self.transform(s2)
		return s2, s1   # opt, sar


class GANDataset(torch.utils.data.Dataset):
    """
    支持:
    - paired：s1-s2 成对
    - unpaired：s1 和 s2 任意匹配
    """

    def __init__(self, root_dir, transform=None, L_transform=None, pair_suffix=('s1', 's2')):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.L_transform = L_transform
        self.pair_suffix = pair_suffix

        self.image_pairs = []      
        s1_path = self.root_dir / pair_suffix[0]
        s2_path = self.root_dir / pair_suffix[1]       
        for s1_file in s1_path.glob('*.png'):
            s2_file = s2_path / s1_file.name.replace(f'_{pair_suffix[0]}_', f'_{pair_suffix[1]}_')
            if s2_file.exists():
                self.image_pairs.append((s1_file, s2_file))

        print(f"Found {len(self.image_pairs)} pairs.")

    def __len__(self):
         return len(self.image_pairs)

    def __getitem__(self, idx): 
        s1_path, s2_path = self.image_pairs[idx]
        s1 = Image.open(s1_path).convert('L')
        s2 = Image.open(s2_path).convert('RGB')
        # ==========================================
        # 同步数据增强：确保 SAR 和 Optical 翻转方向绝对一致
        # 1. 50% 概率水平翻转 (左右)
        if random.random() > 0.5:
            s1 = TF.hflip(s1)
            s2 = TF.hflip(s2)
            
        # 2. 50% 概率垂直翻转 (上下)
        if random.random() > 0.5:
            s1 = TF.vflip(s1)
            s2 = TF.vflip(s2)
        # ==========================================
        if self.L_transform:
            s1 = self.L_transform(s1)
        if self.transform:
            s2 = self.transform(s2)
        return s2,s1
  
# ---------------------------
# Utilities / Scheduler / EMA
# ---------------------------


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
	"""线性 beta 日程。"""
	return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
	"""改进版 DDPM 的余弦日程，前期加噪更平滑。"""
	steps = timesteps + 1
	x = torch.linspace(0, timesteps, steps)
	alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
	alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
	betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
	return torch.clamp(betas, 0.0001, 0.999)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
	"""
	创建一个带有线性预热和余弦衰减的学习率调度器。
	"""
	def lr_lambda(current_step):
		# 1. 预热阶段
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		# 2. 余弦衰减阶段
		progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
		return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

	return LambdaLR(optimizer, lr_lambda)

class NoiseScheduler:
	"""预计算扩散系数，并提供相关工具接口。"""

	def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, schedule="linear", device="cpu"):
		self.T = T
		if schedule == "cosine":
			betas = cosine_beta_schedule(T)
		else:
			betas = linear_beta_schedule(T, beta_start, beta_end)
		alphas = 1.0 - betas
		alphas_cumprod = torch.cumprod(alphas, dim=0)
		self.device = device

		self.betas = betas.to(device)
		self.alphas = alphas.to(device)
		self.alphas_cumprod = alphas_cumprod.to(device)

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
		self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)

	def q_sample(self, x0, t, noise=None): #真实噪声
		"""利用闭式表达从 x_0 得到 x_t。"""
		if noise is None:
			noise = torch.randn_like(x0)
		sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
		sqrt_1_ab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
		xt = sqrt_ab * x0 + sqrt_1_ab * noise
		return xt, noise

	def get_time_coeffs(self, t):
		"""返回 p(x_{t-1}|x_t) 计算所需系数。"""
		beta_t = self.betas[t].view(-1, 1, 1, 1)
		alpha_t = self.alphas[t].view(-1, 1, 1, 1)
		alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
		return beta_t, alpha_t, alpha_bar_t


class EMA:
	"""
	带 Warmup 的指数滑动平均。
	初期 decay 较小，让 EMA 快速跟上模型；后期 decay 固定为设定值。
	"""
	def __init__(self, model, decay):
		self.decay = decay
		self.shadow = {}
		self.backup = {}
		for name, param in model.named_parameters():
			if param.requires_grad:
				self.shadow[name] = param.detach().clone()

	def update(self, model, step=None):
		# 动态计算当前步的 decay
		if step is not None:
			decay = min(self.decay, (1 + step) / (10 + step))
		else:
			decay = self.decay

		for name, param in model.named_parameters():
			if param.requires_grad:
				new_avg = (1.0 - decay) * param.detach() + decay * self.shadow[name]
				self.shadow[name] = new_avg.clone()

	def to_model(self, model):
		"""将 EMA 参数加载到模型，同时备份原参数。"""
		for name, param in model.named_parameters():
			if param.requires_grad:
				self.backup[name] = param.detach().clone()
				param.data.copy_(self.shadow[name].to(param.device))

	def restore(self, model):
		"""在使用 EMA 后恢复原参数。"""
		for name, param in model.named_parameters():
			if param.requires_grad:
				param.data.copy_(self.backup[name].to(param.device))
		self.backup = {}


# ---------------------------
# 网络助手
# ---------------------------
def exists(x):
	return x is not None

def default(val, d):
	if exists(val):
		return val
	return d() if isfunction(d) else d


def num_to_groups(num, divisor):
	groups = num // divisor
	remainder = num % divisor
	arr = [divisor] * groups
	if remainder > 0:
		arr.append(remainder)
	return arr


class Residual(nn.Module):
	def __init__(self, fn):
		super().__init__()
		self.fn = fn

	def forward(self, x, *args, **kwargs):
		return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out):
	return nn.Sequential(
		nn.Upsample(scale_factor=2, mode="nearest"),
		nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
	)


def Downsample(dim, dim_out):
	return nn.Sequential(
		Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
		nn.Conv2d(dim * 4, default(dim_out, dim), 1),
	)


# ---------------------------
# 位置嵌入
# ---------------------------
class SinusoidalPositionEmbeddings(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim

	def forward(self, time):
		device = time.device
		half_dim = self.dim // 2
		embeddings = math.log(10000) / (half_dim - 1)
		embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
		embeddings = time[:, None] * embeddings[None, :]
		embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
		return embeddings

# ---------------------------
# 其它网络模块（Block, ResnetBlock, Attention 等）
# ---------------------------
class WeightStandardizedConv2d(nn.Conv2d):
	def forward(self, x):
		eps = 1e-5 if x.dtype == torch.float32 else 1e-3

		weight = self.weight
		mean = reduce(weight, "o ... -> o 1 1 1", "mean")
		var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
		normalized_weight = (weight - mean) * (var + eps).rsqrt()

		return F.conv2d(
			x,
			normalized_weight,
			self.bias,
			self.stride,
			self.padding,
			self.dilation,
			self.groups,
		)


class Block(nn.Module):
	def __init__(self, dim, dim_out, groups=8):
		super().__init__()
		self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
		self.norm = nn.GroupNorm(groups, dim_out)
		self.act = nn.SiLU()

	def forward(self, x, scale_shift=None):
		x = self.proj(x)
		x = self.norm(x)

		if exists(scale_shift):
			scale, shift = scale_shift
			x = x * (scale + 1) + shift

		x = self.act(x)
		return x


class ResnetBlock(nn.Module):
	def __init__(self, dim, dim_out,* , time_emb_dim=None, groups=8):
		super().__init__()
		self.mlp = (
			nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
			if exists(time_emb_dim)
			else None
		)
		self.block1 = Block(dim, dim_out, groups=groups)
		self.block2 = Block(dim_out, dim_out, groups=groups)
		self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

	def forward(self, x, time_emb=None):
		scale_shift = None
		if exists(self.mlp) and exists(time_emb):
			time_emb = self.mlp(time_emb)
			time_emb = rearrange(time_emb, "b c -> b c 1 1")
			scale_shift = time_emb.chunk(2, dim=1)

		h = self.block1(x, scale_shift=scale_shift)
		h = self.block2(h)
		return h + self.res_conv(x)


class Attention(nn.Module):
	def __init__(self, dim, heads=4, dim_head=32):
		super().__init__()
		self.scale = dim_head**-0.5
		self.heads = heads
		hidden_dim = dim_head * heads
		self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
		self.to_out = nn.Conv2d(hidden_dim, dim, 1)

	def forward(self, x):
		b, c, h, w = x.shape
		qkv = self.to_qkv(x).chunk(3, dim=1)
		q, k, v = map(
			lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
		)
		q = q * self.scale

		sim = einsum("b h d i, b h d j -> b h i j", q, k)
		sim = sim - sim.amax(dim=-1, keepdim=True).detach()
		attn = sim.softmax(dim=-1)

		out = einsum("b h i j, b h d j -> b h i d", attn, v)
		out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
		return self.to_out(out)


class LinearAttention(nn.Module):
	def __init__(self, dim, heads=4, dim_head=32):
		super().__init__()
		self.scale = dim_head**-0.5
		self.heads = heads
		hidden_dim = dim_head * heads
		self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

		self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
									nn.GroupNorm(1, dim))

	def forward(self, x):
		b, c, h, w = x.shape
		qkv = self.to_qkv(x).chunk(3, dim=1)
		q, k, v = map(
			lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
		)

		q = q.softmax(dim=-2)
		k = k.softmax(dim=-1)

		q = q * self.scale
		context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

		out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
		out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
		return self.to_out(out)


class PreNorm(nn.Module):
	def __init__(self, dim, fn):
		super().__init__()
		self.fn = fn
		self.norm = nn.GroupNorm(1, dim)

	def forward(self, x):
		x = self.norm(x)
		return self.fn(x)


class Sar_conv_small(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.GroupNorm(8, out_channels),
			nn.SiLU()
		)

	def forward(self, x):
		return self.conv(x)


class Sar_conv_large(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3),
			nn.GroupNorm(8, out_channels),
			nn.SiLU()
		)

	def forward(self, x):
		return self.conv(x)


class SKFusion(nn.Module):
	def __init__(self, channels, reduction=4):
		super().__init__()
		self.global_pool = nn.AdaptiveAvgPool2d(1)
		mid_channels = max(32, channels // reduction)
		self.mlp = nn.Sequential(
			nn.Linear(channels, mid_channels, bias=False),
			nn.BatchNorm1d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Linear(mid_channels, channels * 2, bias=False)
		)

	def forward(self, cond_s, cond_l):
		batch_size, channels, _, _ = cond_s.shape
		U = cond_s + cond_l 
		S = self.global_pool(U).view(batch_size, channels)
		Z = self.mlp(S)
		Z = Z.view(batch_size, 2, channels, 1, 1)
		attention_weights = F.softmax(Z, dim=1)
		cond_fused = cond_s * attention_weights[:, 0, :, :, :] + \
					 cond_l * attention_weights[:, 1, :, :, :]
		return cond_fused


class Unet(nn.Module):
	def __init__(
		self,
		dim=128,
		init_dim=None,
		out_dim=None,
		dim_mults=(1, 1, 2, 2, 4, 4),
		channels=3,
		cond_channels=1,
		resnet_block_groups=8,
	):
		super().__init__()

		self.channels = channels
		self.cond_channels = cond_channels
		init_dim = default(init_dim, dim)

		self.opt_conv = nn.Conv2d(self.channels, init_dim, 1, padding=0)
		self.sar_conv = nn.Conv2d(self.cond_channels, init_dim, 1, padding=0)

		dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
		in_out = list(zip(dims[:-1], dims[1:]))

		block_klass = partial(ResnetBlock, groups=resnet_block_groups)
		time_dim = dim * 4

		self.time_mlp = nn.Sequential(
			SinusoidalPositionEmbeddings(dim),
			nn.Linear(dim, time_dim),
			nn.GELU(),
			nn.Linear(time_dim, time_dim),
		)

		self.downs = nn.ModuleList([])
		self.ups = nn.ModuleList([])
		self.sarl = nn.ModuleList([])
		self.sars = nn.ModuleList([])
		self.fusions = nn.ModuleList([])
		num_resolutions = len(in_out)

		for ind, (dim_in, dim_out) in enumerate(in_out):
			is_last = ind >= (num_resolutions - 1)

			self.downs.append(
				nn.ModuleList([
					block_klass(dim_in, dim_in, time_emb_dim=time_dim),
					block_klass(dim_in, dim_in, time_emb_dim=time_dim),
					Residual(PreNorm(dim_in, LinearAttention(dim_in))),
					Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
				])
			)
			self.sarl.append(
				nn.ModuleList([
					Sar_conv_large(dim_in, dim_in),
					Sar_conv_large(dim_in, dim_in),
					Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
				])
			)
			self.sars.append(
				nn.ModuleList([
					Sar_conv_small(dim_in, dim_in),
					Sar_conv_small(dim_in, dim_in),
					Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
				])
			)
			self.fusions.append(SKFusion(dim_in))

		mid_dim = dims[-1]
		self.mid_sarl = Sar_conv_large(mid_dim, mid_dim)
		self.mid_sars = Sar_conv_small(mid_dim, mid_dim)
		self.mid_fusion = SKFusion(mid_dim)
		self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
		self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
		self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

		for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
			is_last = ind == (len(in_out) - 1)
			self.ups.append(
				nn.ModuleList([
					block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
					block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
					Residual(PreNorm(dim_out, LinearAttention(dim_out))),
					Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
				])
			)

		self.out_dim = default(out_dim, channels)
		self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
		self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

	def forward(self, x, time, cond):
		x = self.opt_conv(x)
		cond = self.sar_conv(cond)
		r = x.clone()
		t = self.time_mlp(time)
		cond_l = cond.clone()
		cond_s = cond.clone()
		h = []
		for (opt_blocks, sarl_blocks, sars_blocks, fusion) in zip(self.downs, self.sarl, self.sars, self.fusions):
			block1, block2, attn, downsample = opt_blocks
			l_conv1, l_conv2, l_down = sarl_blocks
			s_conv1, s_conv2, s_down = sars_blocks

			cond_l = l_conv1(cond_l)
			cond_s = s_conv1(cond_s)
			cond_fused = fusion(cond_s, cond_l)
			x = block1(x, t)
			x = x + cond_fused
			h.append(x)

			cond_l = l_conv2(cond_l)
			cond_s = s_conv2(cond_s)
			cond_fused = fusion(cond_s, cond_l)
			x = block2(x, t)
			x = x + cond_fused
			x = attn(x)
			h.append(x)

			x = downsample(x)
			cond_l = l_down(cond_l)
			cond_s = s_down(cond_s)

		x = self.mid_block1(x, t)
		cond_l = self.mid_sarl(cond_l)
		cond_s = self.mid_sars(cond_s)
		cond_fused_mid = self.mid_fusion(cond_s, cond_l)
		x = x + cond_fused_mid

		x = self.mid_attn(x)
		x = self.mid_block2(x, t)

		for block1, block2, attn, upsample in self.ups:
			x = torch.cat((x, h.pop()), dim=1)
			x = block1(x, t)

			x = torch.cat((x, h.pop()), dim=1)
			x = block2(x, t)
			x = attn(x)

			x = upsample(x)

		x = torch.cat((x, r), dim=1)
		x = self.final_res_block(x, t)
		return self.final_conv(x)


# ---------------------------
# VGG 感知损失（Time-aware 可加权）
# ---------------------------

class VGGPerceptualLoss(nn.Module):
	"""提取 VGG 特征并计算分层感知损失。输入为 [-1,1] 的 RGB 图像。
	该模块不会更新 VGG 参数，但会对输入保留梯度，从而能够把感知损失传回生成模型。
	"""

	def __init__(self, device="cpu", layer_ids=None, weights=None):
		super().__init__()
		vgg = models.vgg16(weights='DEFAULT').features
		
		self.vgg = vgg.to(device).eval()
		for p in self.vgg.parameters():
			p.requires_grad = False

		self.layer_ids = layer_ids if layer_ids is not None else [3, 8]
		if weights is None:
			self.weights = [1.0 / len(self.layer_ids)] * len(self.layer_ids)
		else:
			self.weights = weights

		# 加上 device=device 避免跨设备报错
		self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
		self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

	def preprocess(self, x):
		x = (x + 1.0) / 2.0
		x = (x - self.mean) / self.std
		return x

	def forward(self, x, y):
		x = self.preprocess(x)
		y = self.preprocess(y)

		feats_x = []
		feats_y = []
		out_x = x
		out_y = y
		for i, layer in enumerate(self.vgg):
			out_x = layer(out_x)
			out_y = layer(out_y)
			if i in self.layer_ids:
				feats_x.append(out_x)
				feats_y.append(out_y)
				# 【优化】提取完所有需要的浅层后直接终止，不再计算后续层，节省时间
				if len(feats_x) == len(self.layer_ids):
					break 

		loss = torch.tensor(0.0, device=x.device)
		for fx, fy, w in zip(feats_x, feats_y, self.weights):
			loss = loss + w * F.mse_loss(fx, fy)

		return loss

# ---------------------------
# 频率一致性损失（Time-aware 可加权）
# ---------------------------
class FocalFrequencyLoss(nn.Module):
    """
    基于论文的焦点频率一致性损失 (Focal Frequency Consistency Loss)
    公式参考论文 Eq.(22) - Eq.(24)
    """
    def __init__(self, k_f=2.0):
        super().__init__()
        self.k_f = k_f # 缩放因子，论文中默认 k_f=2

    def forward(self, pred, target, t_norm):
        # 1. 计算 2D 傅里叶频谱 (使用正交归一化保证能量守恒)
        fft_pred = torch.fft.fft2(pred, norm='ortho')
        fft_target = torch.fft.fft2(target, norm='ortho')
        
        # 获取实部和虚部
        R_pred, I_pred = fft_pred.real, fft_pred.imag
        R_target, I_target = fft_target.real, fft_target.imag
        
        # 2. 计算频谱差异 d(u,v) = (R - R_hat)^2 + (I - I_hat)^2
        diff_R = R_target - R_pred
        diff_I = I_target - I_pred
        dist = diff_R**2 + diff_I**2
        
        # 3. 计算焦点重加权系数 w(u,v) = (|R - R_hat| + |I - I_hat|)^k_f
        w = (torch.abs(diff_R) + torch.abs(diff_I)) ** self.k_f
        
        # 为保证稳定性，按批次内每张图对权重 w 进行 [0, 1] 归一化
        w_max = w.view(w.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        w_norm = w / (w_max + 1e-8)
        
        # 4. 时间步感知权重 (1 - t/T)^2
        t_weight = (1.0 - t_norm).view(-1, 1, 1, 1) ** 2
        
        # 5. 组合最终损失
        loss_freq = t_weight * w_norm * dist
        return loss_freq.mean()
# ---------------------------
# 损失

# ---------------------------

def diffusion_loss_with_color(model, scheduler, x0, cond, device, vgg_module=None, base_lambda_vgg=0.1, freq_module=None, base_lambda_freq=2.0):
	b = x0.shape[0]
	# 随机采样时间步
	t = torch.randint(0, scheduler.T, (b,), device=device).long()
	xt, noise = scheduler.q_sample(x0, t)

	# 1. 预测噪声
	noise_pred = model(xt, t, cond)

	# 2. 基础扩散损失（MSE，负责生成结构和细节）
	loss_simple = F.mse_loss(noise_pred, noise, reduction="mean")

	# 3. 反推当前预测的清晰原图 x0_hat
	alpha_bar_t = scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
	x0_hat = (xt - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
	x0_hat = torch.clamp(x0_hat, -1.0, 1.0)

	# ==========================================
	# 4. 计算低频色彩损失 (Color Loss)
	# 使用 15x15 大核平均池化抹除高频细节，只对齐整体色彩基调
	# ==========================================
	pool = nn.AvgPool2d(kernel_size=31, stride=1, padding=15)
	x0_hat_blur = pool(x0_hat)
	x0_blur = pool(x0)
	loss_color = F.l1_loss(x0_hat_blur, x0_blur)

	# ==========================================
	# 5. 计算时间动态权重 (Time-aware Weighting)
	# 当 t 很大(高噪声)时，权重趋近 0；t 很小(接近清晰)时，权重最大
	# ==========================================
	t_norm = t.float() / scheduler.T
	t_norm_view = t_norm.view(-1, 1, 1, 1)
	decay_factor = (1.0 - t_norm_view) ** 2

	base_lambda_color = 0.2  # 色彩损失基础权重
	weight_color = (base_lambda_color * decay_factor).mean()

	# ==========================================
	# 6. VGG 感知损失（若提供 vgg_module）
	# ==========================================
	if vgg_module is not None:
		vgg_loss = vgg_module(x0_hat, x0)
        
		t_threshold = int(scheduler.T * 0.2) # 比如 T=1000 时，只有 t < 200 才算 VGG
		
		# 当 t < t_threshold 时，权重从 0 线性增加到 1；否则权重为 0
		decay_factor_vgg = torch.where(
			t < t_threshold, 
			(1.0 - t.float() / t_threshold).view(-1, 1, 1, 1), 
			torch.zeros_like(t.float()).view(-1, 1, 1, 1)
		)
		weight_vgg = (base_lambda_vgg * decay_factor_vgg).mean()
	else:
		vgg_loss = torch.tensor(0.0, device=device)
		weight_vgg = torch.tensor(0.0, device=device)

	# ==========================================
	# 7. 频率一致性损失（Focal Frequency Loss）
	# ==========================================
	if freq_module is not None:
		# 对预测的清晰图和真实图计算频率一致性损失
		freq_loss = freq_module(x0_hat, x0, t_norm)
		weight_freq = base_lambda_freq  # 频率一致性损失权重为 2
	else:
		freq_loss = torch.tensor(0.0, device=device)
		weight_freq = torch.tensor(0.0, device=device)

	# 8. 总损失
	loss_total = loss_simple + weight_color * loss_color + weight_vgg * vgg_loss + weight_freq * freq_loss

	# 返回总损失，以及拆分项(方便在 TensorBoard 中监控)
	return loss_total, loss_simple, loss_color, vgg_loss, freq_loss


# ---------------------------
# 采样器
# ---------------------------


@torch.no_grad()
def p_sample(model, scheduler, xt, t,cond):
	beta_t, alpha_t, alpha_bar_t = scheduler.get_time_coeffs(t)
	if t[0] == 0:
		alpha_bar_prev = torch.tensor(1.0, device=xt.device).view(-1, 1, 1, 1)
	else:
		alpha_bar_prev = scheduler.alphas_cumprod[t - 1].view(-1, 1, 1, 1)

	eps_pred = model(xt, t, cond)

	mean = (1.0 / torch.sqrt(alpha_t)) * (
		xt - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
	)

	if t[0] == 0:
		return mean

	beta_tilde = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
	noise = torch.randn_like(xt)
	xt_prev=mean + torch.sqrt(beta_tilde) * noise
   
	return xt_prev


@torch.no_grad()
def p_sample_loop_ddpm(model, scheduler, noise, device, cond):
	img = noise
	b = img.shape[0]
	total_steps = scheduler.T

	for i in reversed(range(total_steps)):
		t = torch.full((b,), i, device=device, dtype=torch.long)
		img = p_sample(model, scheduler, img, t, cond)

	return img


@torch.no_grad()
def ddim_step(model, scheduler, xt, t,cond, t_prev, eta=0.0):
	b = xt.shape[0]
	device = xt.device

	t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
	eps_pred = model(xt, t_tensor,cond)

	alpha_bar_t = scheduler.alphas_cumprod[t_tensor].view(-1, 1, 1, 1)

	if t_prev >= 0:
		t_prev_tensor = torch.full((b,), t_prev, device=device, dtype=torch.long)
		alpha_bar_prev = scheduler.alphas_cumprod[t_prev_tensor].view(-1, 1, 1, 1)
	else:
		alpha_bar_prev = torch.ones_like(alpha_bar_t)

	x0_pred = (
		(xt - torch.sqrt(1 - alpha_bar_t) * eps_pred)
		/ torch.sqrt(alpha_bar_t)
	)

	sigma = (
		eta
		* torch.sqrt(
			(1 - alpha_bar_prev) / (1 - alpha_bar_t)
			* (1 - alpha_bar_t / alpha_bar_prev)
		)
	)

	mean = (
		torch.sqrt(alpha_bar_prev) * x0_pred
		+ torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps_pred
	)

	noise = sigma * torch.randn_like(xt)
	xt_prev=mean + noise

	return xt_prev


@torch.no_grad()
def p_sample_loop_ddim(model, scheduler, noise, device, cond, steps=50, eta=0.0):
	img = noise
	b = img.shape[0]
	timesteps = torch.linspace(scheduler.T - 1, 0, steps, dtype=torch.long, device=device)

	for i, t in enumerate(timesteps):
		t_int = int(t.item())
		t_prev = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1

		img = ddim_step(model, scheduler, img, t_int, cond, t_prev, eta=eta)

	return img


# ---------------------------
# Checkpoint / 采样辅助
# ---------------------------


def save_checkpoint(model, optimizer, step, path, ema=None):
	state = {
		"model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
		"optimizer": optimizer.state_dict(),
		"step": step,
	}
	if ema is not None:
		state["ema"] = {k: v.cpu().clone() for k, v in ema.shadow.items()}

	def _save_task():
		try:
			torch.save(state, path)
			print(f"Async saved checkpoint: {path}")
		except Exception as e:
			print(f"Error saving checkpoint: {e}")

	thread = threading.Thread(target=_save_task)
	thread.start()


def load_checkpoint(path, model, optimizer=None, ema=None, device="cpu"):
	ckpt = torch.load(path, map_location=device)
	model.load_state_dict(ckpt["model"]) 
	if optimizer is not None and "optimizer" in ckpt:
		optimizer.load_state_dict(ckpt["optimizer"])
	if ema is not None and "ema" in ckpt:
		for k, v in ckpt["ema"].items():
			ema.shadow[k] = v.clone()
	return ckpt.get("step", 0)


# ---------------------------
# 训练循环
# ---------------------------


def train(args):
	device = torch.device(args.cuda if torch.cuda.is_available() and not args.no_cuda else "cpu")
	# Optical: [-1,1]
	opt_transform = transforms.Compose([
		transforms.Resize((args.image_size, args.image_size)),
		transforms.CenterCrop(args.image_size),
		transforms.ToTensor(), # [0,1]
		transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)), # -> [-1,1]
	])
	# SAR: [-1,1]
	L_transform = transforms.Compose([
		transforms.Resize((args.image_size, args.image_size)),
		transforms.CenterCrop(args.image_size),
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))
	])

	dataset = PairDataset(root_dir=args.data_dir, transform=opt_transform, L_transform=L_transform)
	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

	model = Unet(
		dim=args.base_ch,
		out_dim=3,
		channels=3,
		cond_channels=1,
		dim_mults=args.ch_mult,
	).to(device)

	scheduler = NoiseScheduler(
		T=args.timesteps,
		beta_start=args.beta_start,
		beta_end=args.beta_end,
		schedule=args.beta_schedule,
		device=str(device),
	)
    

	optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
	total_opt_steps = (args.epochs * len(loader)) // args.accumulation_steps
	warmup_steps = int(total_opt_steps * 0.05)
	lr_scheduler = get_cosine_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=warmup_steps,
		num_training_steps=total_opt_steps
	)
	ema = EMA(model, decay=args.ema_decay)

	# VGG 感知损失模块（若启用）
	vgg_module = VGGPerceptualLoss(device=device) if args.vgg_lambda > 0 else None
	if vgg_module is not None:
		vgg_module.eval()  # 确保 VGG 模块在评估模式，不更新统计信息

	# 频率一致性损失模块（若启用）
	freq_module = FocalFrequencyLoss(k_f=2.0) if args.freq_lambda > 0 else None

	writer = SummaryWriter(log_dir=args.log_dir)
    
	global_step = 0
	start_epoch = 0
    
	if args.resume:  
		if os.path.isfile(args.resume):
			print(f"Resuming training from checkpoint: {args.resume}")
			loaded_step = load_checkpoint(args.resume, model, optimizer, ema, device=str(device))
			global_step = loaded_step
			start_epoch = global_step // len(loader)
			print(f"Resumed at Step {global_step}, Epoch {start_epoch+1}")
		else:
			print(f"Warning: Checkpoint file {args.resume} not found! Starting from scratch.")

	model.train()
	for epoch in range(start_epoch, args.epochs):
		pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
		optimizer.zero_grad()
		for batch in pbar:
			x, y = batch
			x = x.to(device)
			y = y.to(device)

			loss_total, loss_simple, loss_color, loss_vgg, loss_freq = diffusion_loss_with_color(
				model, scheduler, x, y, device, 
				vgg_module=vgg_module, base_lambda_vgg=args.vgg_lambda,
				freq_module=freq_module, base_lambda_freq=args.freq_lambda
			)
			loss = loss_total / args.accumulation_steps
			loss.backward()

			global_step += 1

			if global_step % args.accumulation_steps == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()

				current_lr = lr_scheduler.get_last_lr()[0]
				if global_step % (args.accumulation_steps * 10) == 0:
					writer.add_scalar("train/lr", current_lr, global_step)
					pbar.set_postfix({"loss": float(loss_total.item()), "lr": f"{current_lr:.2e}"})
				# 记录各项损失
				writer.add_scalar("train/loss_total", float(loss_total.item()), global_step)
				writer.add_scalar("train/loss_simple", float(loss_simple.item()), global_step)
				writer.add_scalar("train/loss_color", float(loss_color.item()), global_step)
				if isinstance(loss_vgg, torch.Tensor):
					writer.add_scalar("train/loss_vgg", float(loss_vgg.item()), global_step)
				if isinstance(loss_freq, torch.Tensor):
					writer.add_scalar("train/loss_freq", float(loss_freq.item()), global_step)

				if global_step % args.ema_interval == 0:
					ema.update(model, step=global_step)
                    
				if global_step % args.save_interval == 0:
					with torch.no_grad():
						model.eval()
                        
						sar_cond = y[: args.sample_batch]
						opt_cond = x[: args.sample_batch]

						noise = torch.randn(
							args.sample_batch,
							3,
							args.image_size,
							args.image_size,
							device=device,
						)

						print(f"Sampling from EMA Model at step {global_step}...")
						ema.to_model(model)
						if args.sampler == "ddim":
							sample_ema = p_sample_loop_ddim(
								model, scheduler, noise.clone(), device, sar_cond, steps=args.ddim_steps, eta=args.ddim_eta
							)
						else:
							sample_ema = p_sample_loop_ddpm(model, scheduler, noise.clone(), device, sar_cond)
						ema.restore(model)
                        
						model.train()
                        
						sample_ema_vis = (sample_ema[:args.num] + 1.0) / 2.0
						x_vis = (opt_cond[:args.num] + 1.0) / 2.0
						y_vis = (sar_cond[:args.num].repeat(1, 3, 1, 1) + 1.0) / 2.0

						compare = torch.cat([y_vis, x_vis, sample_ema_vis], dim=0)
						compare = torch.clamp(compare, 0.0, 1.0) 
						sample_grid = utils.make_grid(compare, nrow=args.num, padding=2)
						sample_path = os.path.join(args.sample_dir, f"sample_{epoch}_{global_step:06d}.png")
						utils.save_image(sample_grid, sample_path)
						writer.add_image("samples", sample_grid, global_step)
						print("Saved sample", sample_path)

				if global_step % args.save_ckpt == 0:
					ckpt_path = os.path.join(args.model_dir, f"ckpt_{epoch}_{global_step:06d}.pt")
					save_checkpoint(model, optimizer, global_step, ckpt_path, ema=ema)
					checkpoints = sorted(glob.glob(os.path.join(args.model_dir, "ckpt_*.pt")))
					if len(checkpoints) > 5:
						os.remove(checkpoints[0])
						print(f"Removed old checkpoint: {checkpoints[0]}")

	print("Training finished")
	writer.close()



# ---------------------------
# Argument parsing & main
# ---------------------------


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument("--data_dir", type=str, default="/home/lx/deep_learning/dataset/archive/v_2_split/train")
	p.add_argument("--log_dir", type=str, default="./logs_new5")
	p.add_argument("--model_dir", type=str, default="./models_new5")
	p.add_argument("--sample_dir", type=str, default="./samples_new5")
	p.add_argument("--resume", type=str, default="./models_new5/ckpt_179_684000.pt", help="Path to checkpoint to resume training from")
	p.add_argument("--cuda", type=str, default="cuda:1")
	p.add_argument("--no_cuda", action="store_true")  
	p.add_argument("--image_size", type=int, default=256)
	p.add_argument("--batch_size", type=int, default=4)
	p.add_argument("--accumulation_steps", type=int, default=8,help="用来放大批次数降低模型训练震荡")
	p.add_argument("--num", type=int, default=4, help="每次记录和保存多少个样本的生成结果")
	p.add_argument("--epochs", type=int, default=200)
	p.add_argument("--lr", type=float, default=2e-4)
	p.add_argument("--grad_clip", type=float, default=1.0)
	p.add_argument("--timesteps", type=int, default=1000)
	p.add_argument("--beta_start", type=float, default=1e-4)
	p.add_argument("--beta_end", type=float, default=2e-2)
	p.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])
	p.add_argument("--base_ch", type=int, default=128)
	p.add_argument("--ch_mult", nargs="+", type=int, default=[1,1,2, 2, 4,4])
	p.add_argument("--num_res_blocks", type=int, default=2)
	p.add_argument("--attn_resolutions", nargs="+", type=int, default=[16, 8])
	p.add_argument("--dropout", type=float, default=0)
	p.add_argument("--ema_decay", type=float, default=0.9995)
	p.add_argument("--ema_interval", type=int, default=1)
	p.add_argument("--sample_batch", type=int, default=4)
	p.add_argument("--save_interval", type=int, default=3800)
	p.add_argument("--save_ckpt", type=int, default=38000)
	p.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
	p.add_argument("--ddim_steps", type=int, default=50)
	p.add_argument("--ddim_eta", type=float, default=0.0)
	p.add_argument("--vgg_lambda", type=float, default=0.2, help="基础 VGG 感知损失权重（会再乘以 time-aware 因子）")
	p.add_argument("--freq_lambda", type=float, default=2.0, help="焦点频率一致性损失权重")
    
	return p.parse_args()


if __name__ == "__main__":
	args = parse_args()
	os.makedirs(args.model_dir, exist_ok=True)
	os.makedirs(args.sample_dir, exist_ok=True)
	os.makedirs(args.log_dir, exist_ok=True)
	train(args)

