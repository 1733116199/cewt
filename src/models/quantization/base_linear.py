import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import integrate
from scipy.stats import norm
import math



class BaseQuantizer(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits
        self.n_levels = 2**bits


class NoQuantizer(BaseQuantizer):
    def __init__(self, **kwargs):
        super().__init__(16)

    def forward(self, x):
        return x


def blockwise_matmul(x: torch.Tensor, matrix: torch.Tensor, blocksize: int):
    return torch.reshape(
        x.reshape(*x.shape[:-1], x.shape[-1] // blocksize, blocksize) @ matrix,
        x.shape,
    )
    
hadamard_matrices = {}
def my_hadamard_transform(x: torch.Tensor, logblocksize: int = 7):
    assert logblocksize >= 1
    blocksize = 1 << logblocksize
    assert (x.shape[-1] % blocksize) == 0
    key = (x.dtype, x.device, logblocksize)
    if key not in hadamard_matrices:
        h = torch.tensor([[1.0]], dtype=x.dtype, device=x.device)
        for _ in range(logblocksize):
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        h = h * (2 ** (-logblocksize / 2))
        hadamard_matrices[key] = h
    return blockwise_matmul(x, hadamard_matrices[key], blocksize)

# class UniformQuantizer(BaseQuantizer):
#     def forward(self, x):
#         if not self.training:
#             return x
#         scale = torch.max(torch.abs(x), dim=-1, keepdim=True) + 1e-8
#         step = scale * 2 / (self.n_levels - 1)
#         x_clip = torch.clamp(x, -scale, scale)
#         xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
#         return x + (xq - x).detach()


OPTIMAL_GAUSSIAN_SCALES = {
    1: 0.7978845587140913,
    1.585: 1.2240089519030855,
    2: 1.4935346200015913,
    3: 2.051068354131873,
    4: 2.513930578568423,
    5: 2.9160938834961225,
    6: 3.276597282593217,
    7: 3.6010497188221655,
    8: 3.884938678807525,
}


class QuESTImproved(BaseQuantizer):
    '''
        Implementation of QuEST
        1. IMPROVED: std gradient NOT detached
        2. ORIGINAL: Trust estimator with default trust factor equivalent to clipping with extra 0.5 on each side
        3. ORIGINAL: Both weights and activations use per-channel quantization.
        4. ORIGINAL: default hadamard block size of 128x128
    '''
    def __init__(self, bits=2, log_blocksize=7, **kwargs):
        super().__init__(bits)
        self.n_levels = 1 << self.bits
        self.half_n_levels = 1 << (self.bits - 1)
        self.log_blocksize = log_blocksize

    def quantize(self, x: torch.Tensor, detach_step=False):
        if self.log_blocksize > 0:
            x = my_hadamard_transform(x, self.log_blocksize)
        std = x.square().mean(dim=-1, keepdim=True).sqrt() # native QuEST detaches std, we do not
        scale = std * OPTIMAL_GAUSSIAN_SCALES[self.bits] + 1e-8
        step = scale / (self.n_levels - 1) * 2
        if detach_step:
            step = step.detach()
        x = x / step - 1 / 2 # most good stuff between -half_n_levels and half_n_levels - 1
        x = torch.clip(x, -self.half_n_levels - 0.5, self.half_n_levels - 1 + 0.5)# clip good stuff with 0.5 on each side to match the trust estimator
        xq = torch.round(x).clip(-self.half_n_levels, self.half_n_levels - 1) # round and prevent cheating
        return xq, x, step

    def forward(self, x: torch.Tensor):
        xq, x, step = self.quantize(x) # quantize to unsigned
        x = xq.detach() + x - x.detach() # STE
        x = (x + 1 / 2) * step
        return x
    
    @torch.no_grad()
    def entropy(self, x: torch.Tensor):
        xq = self.quantize(x)[0].int()
        prob: torch.Tensor = torch.unique(xq, return_counts=True)[1]
        prob = prob / prob.sum()
        assert prob.numel() <= self.n_levels
        return torch.sum(-prob * torch.log2(prob))
    
    def extra_repr(self):
        return f"bits={self.bits}, log_blocksize={self.log_blocksize}"


class BBQ(torch.nn.Module):
    '''
        Implementation of BBQ:
        In the original BBQ paper this variant is refered to as BBQ-Vision.
        1. ORIGINAL: std gradient NOT detached
        2. MODIFIED: The scaling factor is set to zeta * rms / half_n_levels in every forward pass 
           instead of being a learned parameter. We do this to have a stateless quantizer which
           is easier to manage.
        3. MODIFIED: We use per-channel quantization for weights and activations. 
           We do this to avoid having to maintain two different quantizers for 
           weights and activations.
        4. ORIGINAL: default hadamard block size of 128x128
    '''
    
    def __init__(
        self,
        bits: int=2, 
        zero_point: float=-0.5,
        log_blocksize: int=7,
        zeta: float=1.694,
        **kwargs
    ) -> None:
        super().__init__()
        # enforce integer precision
        assert bits > 0
        assert round(bits) == bits
        bits = round(bits)
        self.precision = bits

        self.n_levels = 1 << self.precision
        self.half_n_levels = 1 << (self.precision - 1)
        self.zero_point = zero_point
        self.log_blocksize = log_blocksize
        self.zeta = zeta

    def quantize(self, x: torch.Tensor):
        if self.log_blocksize > 0:
            x = my_hadamard_transform(x, self.log_blocksize)
        
        rms = x.square().mean(dim=-1, keepdim=True).sqrt() + 1e-8
        
        x = 0.5 * (1 + torch.erf((2 ** (-0.5) / rms) * x))
        x = x * self.n_levels - 0.5
        xq = torch.round(x).clip(0, self.n_levels - 1) # within range [0, 2^p-1]
        
        return xq, x, rms

    def forward(self, x: torch.Tensor):
        xq, x, rms = self.quantize(x)
        x = xq.detach() + x - x.detach() # STE
        x = (x - (self.half_n_levels + self.zero_point)) # recenter
        x = x * (self.zeta * rms / self.half_n_levels) # rescale
        return x
    
    @torch.no_grad()
    def entropy(self, x: torch.Tensor):
        xq = self.quantize(x)[0].int()
        prob: torch.Tensor = torch.unique(xq, return_counts=True)[1]
        prob = prob / prob.sum()
        assert prob.numel() <= self.n_levels
        return torch.sum(-prob * torch.log2(prob))
    
    def extra_repr(self):
        return f"bits={self.precision}, zero_point={self.zero_point}, log_blocksize={self.log_blocksize}, zeta={self.zeta}"


class QuESTTGCS(BaseQuantizer):
    '''
        Implementation of QuEST with per-tensor std for quantization grid and per-channel std for scaling.
    '''
    def __init__(self, bits=2, log_blocksize=0, **kwargs):
        super().__init__(bits)
        self.n_levels = 1 << self.bits
        self.half_n_levels = 1 << (self.bits - 1)
        self.log_blocksize = log_blocksize

    def quantize(self, x: torch.Tensor):
        if self.log_blocksize > 0:
            x = my_hadamard_transform(x, self.log_blocksize)
        
        std = x.square().mean().sqrt() # native QuEST detaches std, we do not
        scale = std * OPTIMAL_GAUSSIAN_SCALES[self.bits] + 1e-8
        step = scale / (self.n_levels - 1) * 2
        
        cstd = x.square().mean(dim=-1, keepdim=True).sqrt()
        cscale = cstd * OPTIMAL_GAUSSIAN_SCALES[self.bits] + 1e-8
        cstep = cscale / (self.n_levels - 1) * 2
        
        x = x / step - 1 / 2 # most good stuff between -half_n_levels and half_n_levels - 1
        x = torch.clip(x, -self.half_n_levels - 0.5, self.half_n_levels - 1 + 0.5)# clip good stuff with 0.5 on each side to match the trust estimator
        xq = torch.round(x).clip(-self.half_n_levels, self.half_n_levels - 1) # round and prevent cheating
        return xq, x, cstep

    def forward(self, x: torch.Tensor):
        xq, x, cstep = self.quantize(x) # quantize to unsigned
        x = xq.detach() + x - x.detach() # STE
        x = (x + 1 / 2) * cstep
        return x
    
    @torch.no_grad()
    def entropy(self, x: torch.Tensor):
        xq = self.quantize(x)[0].int()
        prob: torch.Tensor = torch.unique(xq, return_counts=True)[1]
        prob = prob / prob.sum()
        assert prob.numel() <= self.n_levels
        return torch.sum(-prob * torch.log2(prob))
    
    def extra_repr(self):
        return f"bits={self.bits}, log_blocksize={self.log_blocksize}"


class BBQTGCS(torch.nn.Module):
    '''
        Implementation of BBQ with per-tensor std for quantization grid and per-channel std for scaling.
    '''
    
    def __init__(
        self,
        bits: int=2, 
        zero_point: float=-0.5,
        log_blocksize: int=0,
        zeta: float=1.694,
        **kwargs
    ) -> None:
        super().__init__()
        # enforce integer precision
        assert bits > 0
        assert round(bits) == bits
        bits = round(bits)
        self.precision = bits

        self.n_levels = 1 << self.precision
        self.half_n_levels = 1 << (self.precision - 1)
        self.zero_point = zero_point
        self.log_blocksize = log_blocksize
        self.zeta = zeta

    def quantize(self, x: torch.Tensor):
        if self.log_blocksize > 0:
            x = my_hadamard_transform(x, self.log_blocksize)
        
        rms = x.square().mean().sqrt()
        crms = x.square().mean(dim=-1, keepdim=True).sqrt()
        
        x = 0.5 * (1 + torch.erf((2 ** (-0.5) / rms) * x))
        x = x * self.n_levels - 0.5
        xq = torch.round(x).clip(0, self.n_levels - 1) # within range [0, 2^p-1]
        
        return xq, x, crms

    def forward(self, x: torch.Tensor):
        x_norm = torch.linalg.norm(x)
        xq, x, crms = self.quantize(x)
        x = xq.detach() + x - x.detach() # STE
        x = (x - (self.half_n_levels + self.zero_point)) # recenter
        x = x * (self.zeta * crms / self.half_n_levels) # rescale
        return x
    
    @torch.no_grad()
    def entropy(self, x: torch.Tensor):
        xq = self.quantize(x)[0].int()
        prob: torch.Tensor = torch.unique(xq, return_counts=True)[1]
        prob = prob / prob.sum()
        assert prob.numel() <= self.n_levels
        return torch.sum(-prob * torch.log2(prob))
    
    def extra_repr(self):
        return f"bits={self.precision}, zero_point={self.zero_point}, log_blocksize={self.log_blocksize}, zeta={self.zeta}"

class HalfHadamardTrustQuantizer(BaseQuantizer):
    def __init__(self, bits=4, log_blocksize=7, trust=None, **kwargs):
        super().__init__(bits)
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust
        self.log_blocksize = log_blocksize

    def forward(self, x):
        x_had = my_hadamard_transform(x, self.log_blocksize)
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class LSQ(BaseQuantizer):
    '''
        Implementation of QuEST with per-tensor std for quantization grid and per-channel std for scaling.
    '''
    def __init__(self, bits=2, symm=True, **kwargs):
        super().__init__(bits)
        self.n_levels = 1 << self.bits
        self.half_n_levels = 1 << (self.bits - 1)
        self.symm = symm
        if self.symm:
            self.qn = -self.half_n_levels
            self.qp = self.half_n_levels - 1
        else:
            self.qn = 0
            self.qp = self.n_levels - 1
        self.scale = torch.nn.Parameter(torch.ones([], dtype=torch.float32), requires_grad=True)
        self.scales_initialized = False
        
    def get_scale(self, x: torch.Tensor):
        if not self.scales_initialized:
            with torch.no_grad():
                if self.bits > 1:
                    scale = 2 * x.abs().mean() / (self.qp ** 0.5)
                else:
                    scale = 2 * x.abs().mean()
                self.scale.data.copy_(scale)
                if torch.distributed.is_initialized():
                    torch.distributed.broadcast(self.scale.data, src=0)
            self.scales_initialized = True
            print(f"initialized step size to {self.scale.abs() + 1e-8}")
        scale = self.scale
        if self.bits > 1:
            scale_backward = scale * ((x.numel() * self.qp) ** (-0.5))
        else:
            scale_backward = scale * (x.numel() ** (-0.5))
        scale = scale.detach() + scale_backward - scale_backward.detach()
        return scale.abs() + 1e-8

    def quantize(self, x: torch.Tensor):
        scale = self.get_scale(x)
        x = x / scale - 1 / 2
        x = torch.clip(x, self.qn, self.qp)
        xq = torch.round(x) # round
        return xq, x, scale

    def forward(self, x: torch.Tensor):
        xq, x, scale = self.quantize(x) # quantize to unsigned
        x = xq.detach() + x - x.detach() # STE
        x = (x + 1 / 2) * scale
        return x
    
    @torch.no_grad()
    def entropy(self, x: torch.Tensor):
        xq = self.quantize(x)[0].int()
        prob: torch.Tensor = torch.unique(xq, return_counts=True)[1]
        prob = prob / prob.sum()
        assert prob.numel() <= self.n_levels
        return torch.sum(-prob * torch.log2(prob))
    
    def extra_repr(self):
        return f"bits={self.bits}, symm={self.symm}"
    
    @torch.no_grad()
    def clip_min_scale(self, x: torch.Tensor):
        min_scale = x.square().mean().sqrt() * OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1) * 2
        self.scale.data.clamp_min_(min_scale)


class QuESTPQN(BaseQuantizer):
    '''
        Implementation of QuEST with PQN during training and STE (trust estimator) during inference
    '''
    def __init__(self, bits=2, log_blocksize=7, **kwargs):
        super().__init__(bits)
        self.n_levels = 1 << self.bits
        self.half_n_levels = 1 << (self.bits - 1)
        self.log_blocksize = log_blocksize

    def quantize(self, x: torch.Tensor):
        if self.log_blocksize > 0:
            x = my_hadamard_transform(x, self.log_blocksize)
        std = x.square().mean(dim=-1, keepdim=True).sqrt() # native QuEST detaches std, we do not
        scale = std * OPTIMAL_GAUSSIAN_SCALES[self.bits] + 1e-8
        step = scale / (self.n_levels - 1) * 2
        x = x / step - 1 / 2 # most good stuff between -half_n_levels and half_n_levels - 1
        x = torch.clip(x, -self.half_n_levels - 0.5, self.half_n_levels - 1 + 0.5)# clip good stuff with 0.5 on each side to match the trust estimator
        xq = torch.round(x).clip(-self.half_n_levels, self.half_n_levels - 1) # round and prevent cheating
        return xq, x, step
    
    
    def pseudo_quantize(self, x: torch.Tensor):
        if self.log_blocksize > 0:
            x = my_hadamard_transform(x, self.log_blocksize)
        std = x.square().mean(dim=-1, keepdim=True).sqrt() # native QuEST detaches std, we do not
        scale = std * OPTIMAL_GAUSSIAN_SCALES[self.bits] + 1e-8
        step = scale / (self.n_levels - 1) * 2
        x = x / step - 1 / 2 # most good stuff between -half_n_levels and half_n_levels - 1
        noise: torch.Tensor = torch.rand_like(x) - 0.5
        xq = (x + noise).clip(-self.half_n_levels, self.half_n_levels - 1) # round and prevent cheating
        return xq, step

    def forward(self, x: torch.Tensor):
        if self.training:
            x, step = self.pseudo_quantize(x) 
        else:
            xq, x, step = self.quantize(x) 
            x = xq.detach() + x - x.detach() # STE
        x = (x + 1 / 2) * step
        return x
    
    @torch.no_grad()
    def entropy(self, x: torch.Tensor):
        xq = self.quantize(x)[0].int()
        prob: torch.Tensor = torch.unique(xq, return_counts=True)[1]
        prob = prob / prob.sum()
        assert prob.numel() <= self.n_levels
        return torch.sum(-prob * torch.log2(prob))
    
    def extra_repr(self):
        return f"bits={self.bits}, log_blocksize={self.log_blocksize}"

class BBQPQN(torch.nn.Module):
    '''
        Implementation of BBQ with PQN during training and STE during inference.
    '''
    
    def __init__(
        self,
        bits: int=2, 
        zero_point: float=-0.5,
        log_blocksize: int=7,
        zeta: float=1.694,
        **kwargs
    ) -> None:
        super().__init__()
        # enforce integer precision
        assert bits > 0
        assert round(bits) == bits
        bits = round(bits)
        self.precision = bits

        self.n_levels = 1 << self.precision
        self.half_n_levels = 1 << (self.precision - 1)
        self.zero_point = zero_point
        self.log_blocksize = log_blocksize
        self.zeta = zeta

    def quantize(self, x: torch.Tensor):
        if self.log_blocksize > 0:
            x = my_hadamard_transform(x, self.log_blocksize)
        
        rms = x.square().mean(dim=-1, keepdim=True).sqrt() + 1e-8
        
        x = 0.5 * (1 + torch.erf((2 ** (-0.5) / rms) * x))
        x = x * self.n_levels - 0.5
        xq = torch.round(x).clip(0, self.n_levels - 1) # within range [0, 2^p-1]
        
        return xq, x, rms
    
    def pseudo_quantize(self, x: torch.Tensor):
        if self.log_blocksize > 0:
            x = my_hadamard_transform(x, self.log_blocksize)
        
        rms = x.square().mean(dim=-1, keepdim=True).sqrt() + 1e-8
        
        x = 0.5 * (1 + torch.erf((2 ** (-0.5) / rms) * x))
        x = x * self.n_levels - 0.5
        noise: torch.Tensor = torch.rand_like(x) - 0.5
        xq = (x + noise).clip(0, self.n_levels - 1) # within range [0, 2^p-1]
        
        return xq, rms

    def forward(self, x: torch.Tensor):
        if self.training:
            x, rms = self.pseudo_quantize(x)
        else:
            xq, x, rms = self.quantize(x)
            x = xq.detach() + x - x.detach() # STE
        x = (x - (self.half_n_levels + self.zero_point)) # recenter
        x = x * (self.zeta * rms / self.half_n_levels) # rescale
        return x
    
    @torch.no_grad()
    def entropy(self, x: torch.Tensor):
        xq = self.quantize(x)[0].int()
        prob: torch.Tensor = torch.unique(xq, return_counts=True)[1]
        prob = prob / prob.sum()
        assert prob.numel() <= self.n_levels
        return torch.sum(-prob * torch.log2(prob))
    
    def extra_repr(self):
        return f"bits={self.precision}, zero_point={self.zero_point}, log_blocksize={self.log_blocksize}, zeta={self.zeta}"

# class STEQuantizer(BaseQuantizer):
#     def __init__(self, bits=4, centered=True):
#         super().__init__(bits)
#         self.centered = centered

#     def forward(self, x):
#         scale = (
#             OPTIMAL_GAUSSIAN_SCALES[self.bits]
#             * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
#             + 1e-8
#         )
#         if self.centered:
#             step = 2 * scale / (self.n_levels - 1)
#             x_clip = torch.clamp(x, -scale, scale)
#             xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
#         else:
#             step = 2 * scale / self.n_levels
#             x_clip = torch.clamp(x, -scale * (self.n_levels - 2) / self.n_levels, scale)
#             xq = torch.round(x_clip / step) * step

#         return x + (xq - x).detach()


# class ClipQuantizer(STEQuantizer):
#     def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
#         super().__init__(bits, centered)
#         self.clip_scale = clip_scale

#     def forward(self, x):
#         scale = (
#             OPTIMAL_GAUSSIAN_SCALES[self.bits]
#             * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
#             + 1e-8
#         )
#         if self.centered:
#             step = 2 * scale / (self.n_levels - 1)
#             x_clip = torch.clamp(x, -scale, scale)
#             xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
#             mask = (torch.abs(x) <= scale * self.clip_scale).float()
#         else:
#             neg_scale = -scale * (self.n_levels - 2)
#             step = 2 * scale / self.n_levels
#             x_clip = torch.clamp(x, neg_scale, scale)
#             xq = torch.round(x_clip / step) * step
#             mask = (
#                 (neg_scale * self.clip_scale <= x) & (x <= scale * self.clip_scale)
#             ).float()
#         return x * mask + (xq - x * mask).detach()


# class HalfHadamardClipQuantizer(STEQuantizer):
#     aux_matrix = hadamard_transform(
#         torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
#     )

#     def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
#         super().__init__(bits, centered)
#         self.matrix = None
#         self.clip_scale = clip_scale

#     def forward(self, x):
#         if self.matrix is None:
#             self.matrix = torch.block_diag(
#                 *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
#             )

#         x_had = x @ self.matrix
#         with torch.no_grad():
#             scale = (
#                 OPTIMAL_GAUSSIAN_SCALES[self.bits]
#                 * torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
#                 + 1e-8
#             )
#             if self.centered:
#                 step = 2 * scale / (self.n_levels - 1)
#                 x_clip = torch.clamp(x_had, -scale, scale)
#                 xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
#                 mask = (torch.abs(x_had) <= scale * self.clip_scale).float()
#             else:
#                 neg_scale = -scale * (self.n_levels - 2)
#                 step = 2 * scale / self.n_levels
#                 x_clip = torch.clamp(x_had, neg_scale, scale)
#                 xq = torch.round(x_clip / step) * step
#                 mask = (
#                     (neg_scale * self.clip_scale <= x_had)
#                     & (x_had <= scale * self.clip_scale)
#                 ).float()

#         grad_flow_output = x_had * mask
#         return grad_flow_output + (xq - grad_flow_output).detach()


# class HadamardClipQuantizer(STEQuantizer):
#     aux_matrix = hadamard_transform(
#         torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
#     )

#     def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
#         super().__init__(bits, centered)
#         self.matrix = None
#         self.clip_scale = clip_scale

#     def forward(self, x):
#         if self.matrix is None:
#             self.matrix = torch.block_diag(
#                 *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
#             )

#         x_had = x @ self.matrix
#         with torch.no_grad():
#             scale = (
#                 OPTIMAL_GAUSSIAN_SCALES[self.bits]
#                 * torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
#                 + 1e-8
#             )
#             if self.centered:
#                 step = 2 * scale / (self.n_levels - 1)
#                 x_clip = torch.clamp(x_had, -scale, scale)
#                 xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
#                 mask = (torch.abs(x_had) <= scale * self.clip_scale).float()
#             else:
#                 neg_scale = -scale * (self.n_levels - 2)
#                 step = 2 * scale / self.n_levels
#                 x_clip = torch.clamp(x_had, neg_scale, scale)
#                 xq = torch.round(x_clip / step) * step
#                 mask = (
#                     (neg_scale * self.clip_scale <= x_had)
#                     & (x_had <= scale * self.clip_scale)
#                 ).float()
#             xq = xq @ self.matrix.T

#         grad_flow_output = (x_had * mask) @ self.matrix.T

#         return grad_flow_output + (xq - grad_flow_output).detach()


# class HalfHadamardTrustQuantizer(STEQuantizer):
#     aux_matrix = hadamard_transform(
#         torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
#     )

#     def __init__(self, bits=4, trust=None):
#         super().__init__(bits, True)
#         self.matrix = None
#         if trust is None:
#             trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
#         self.trust = trust

#     def forward(self, x):
#         if self.matrix is None:
#             self.matrix = torch.block_diag(
#                 *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
#             )

#         x_had = x @ self.matrix
#         with torch.no_grad():
#             std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
#             scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
#             step = 2 * scale / (self.n_levels - 1)
#             x_clip = torch.clamp(x_had, -scale, scale)
#             xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
#             mask = (torch.abs(xq - x_had) <= std * self.trust).float()

#         grad_flow_output = x_had * mask
#         return grad_flow_output + (xq - grad_flow_output).detach()


# class TrustQuantizer(STEQuantizer):
#     def __init__(self, bits=4, centered=True, trust=None):
#         super().__init__(bits, centered)

#         # in terms of std
#         if trust is None:
#             trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
#         self.trust = trust

#     def forward(self, x):
#         std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
#         scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
#         if self.centered:
#             step = 2 * scale / (self.n_levels - 1)
#             x_clip = torch.clamp(x, -scale, scale)
#             xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
#         else:
#             neg_scale = -scale * (self.n_levels - 2)
#             step = 2 * scale / self.n_levels
#             x_clip = torch.clamp(x, neg_scale, scale)
#             xq = torch.round(x_clip / step) * step

#         mask = (torch.abs(xq - x) <= std * self.trust).float()
#         return x * mask + (xq - x * mask).detach()


# class HadamardTrustQuantizer(TrustQuantizer):
#     aux_matrix = hadamard_transform(
#         torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
#     )

#     def __init__(self, bits=4, trust=None):
#         super().__init__(bits, True, trust)
#         self.matrix = None

#     def forward(self, x):
#         if self.matrix is None:
#             self.matrix = torch.block_diag(
#                 *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
#             )

#         x_had = x @ self.matrix
#         with torch.no_grad():
#             std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
#             scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
#             if self.centered:
#                 step = 2 * scale / (self.n_levels - 1)
#                 x_clip = torch.clamp(x_had, -scale, scale)
#                 xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
#             else:
#                 neg_scale = -scale * (self.n_levels - 2)
#                 step = 2 * scale / self.n_levels
#                 x_clip = torch.clamp(x_had, neg_scale, scale)
#                 xq = torch.round(x_clip / step) * step
#             mask = (torch.abs(xq - x_had) <= std * self.trust).float()
#             xq = xq @ self.matrix.T

#         grad_flow_output = (x_had * mask) @ self.matrix.T

#         return grad_flow_output + (xq - grad_flow_output).detach()


# class GaussianSTEQuantizer(BaseQuantizer):
#     def __init__(self, bits=4):
#         super().__init__(bits)
#         self.register_buffer("levels", self._compute_gaussian_levels())

#     def _compute_gaussian_levels(self):
#         levels = np.linspace(-3, 3, self.n_levels)
#         boundaries = np.zeros(self.n_levels + 1)

#         for _ in range(20):
#             boundaries[1:-1] = (levels[1:] + levels[:-1]) / 2
#             boundaries[0] = -float("inf")
#             boundaries[-1] = float("inf")

#             new_levels = []
#             for i in range(self.n_levels):
#                 b_left, b_right = boundaries[i], boundaries[i + 1]

#                 def f(x):
#                     return x * norm.pdf(x)

#                 integral_num = integrate.quad(f, b_left, b_right)[0]
#                 integral_den = integrate.quad(norm.pdf, b_left, b_right)[0]
#                 if integral_den > 1e-10:
#                     new_levels.append(integral_num / integral_den)
#                 else:
#                     new_levels.append(levels[i])
#             levels = np.array(new_levels)
#         return torch.tensor(levels, dtype=torch.float32)

#     def forward(self, x):
#         std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
#         x_norm = x / std
#         expanded_input = x_norm.unsqueeze(-1)
#         distances = torch.abs(expanded_input - self.levels)
#         indices = torch.argmin(distances, dim=-1)
#         xq_norm = self.levels[indices]
#         xq = xq_norm * std

#         return x + (xq - x).detach()


# class GaussianClipQuantizer(GaussianSTEQuantizer):
#     def forward(self, x):
#         std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
#         x_norm = x / std
#         expanded_input = x_norm.unsqueeze(-1)
#         distances = torch.abs(expanded_input - self.levels)
#         indices = torch.argmin(distances, dim=-1)
#         xq_norm = self.levels[indices]
#         xq = xq_norm * std

#         mask = (x_norm.abs() <= self.levels[-1]).float()
#         return x * mask + (xq - x * mask).detach()


# class GaussianTrustQuantizer(GaussianSTEQuantizer):
#     def __init__(self, bits=4, trust=None):
#         super().__init__(bits)
#         if trust is None:
#             trust = (self.levels[-1] - self.levels[-2]) / 2
#         self.trust = trust

#     def forward(self, x):
#         std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
#         x_norm = x / std
#         expanded_input = x_norm.unsqueeze(-1)
#         distances = torch.abs(expanded_input - self.levels)
#         indices = torch.argmin(distances, dim=-1)
#         xq_norm = self.levels[indices]
#         xq = xq_norm * std

#         mask = (torch.abs(xq - x) <= std * self.trust).float()
#         return x * mask + (xq - x * mask).detach()


# class HalfHadamardGaussianClipQuantizer(GaussianClipQuantizer):
#     aux_matrix = hadamard_transform(
#         torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
#     )

#     def __init__(self, bits=4):
#         super().__init__(bits)
#         self.matrix = None

#     def forward(self, x):
#         if self.matrix is None:
#             self.matrix = torch.block_diag(
#                 *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
#             )

#         x_had = x @ self.matrix
#         with torch.no_grad():
#             std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
#             x_norm = x_had / std
#             expanded_input = x_norm.unsqueeze(-1)
#             distances = torch.abs(expanded_input - self.levels)
#             indices = torch.argmin(distances, dim=-1)
#             xq_norm = self.levels[indices]
#             xq = xq_norm * std

#             mask = (x_norm.abs() <= self.levels[-1]).float()

#         grad_flow_output = x_had * mask

#         return grad_flow_output + (xq - grad_flow_output).detach()


# class HadamardGaussianClipQuantizer(GaussianClipQuantizer):
#     aux_matrix = hadamard_transform(
#         torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
#     )

#     def __init__(self, bits=4):
#         super().__init__(bits)
#         self.matrix = None

#     def forward(self, x):
#         if self.matrix is None:
#             self.matrix = torch.block_diag(
#                 *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
#             )

#         x_had = x @ self.matrix
#         with torch.no_grad():
#             std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
#             x_norm = x_had / std
#             expanded_input = x_norm.unsqueeze(-1)
#             distances = torch.abs(expanded_input - self.levels)
#             indices = torch.argmin(distances, dim=-1)
#             xq_norm = self.levels[indices]
#             xq = xq_norm * std

#             xq = xq @ self.matrix.T
#             mask = (x_norm.abs() <= self.levels[-1]).float()

#         grad_flow_output = (x_had * mask) @ self.matrix.T

#         return grad_flow_output + (xq - grad_flow_output).detach()


# class HalfHadamardGaussianTrustQuantizer(GaussianTrustQuantizer):
#     aux_matrix = hadamard_transform(
#         torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
#     )

#     def __init__(self, bits=4, trust=None):
#         super().__init__(bits, trust)
#         self.matrix = None

#     def forward(self, x):
#         if self.matrix is None:
#             self.matrix = torch.block_diag(
#                 *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
#             )

#         x_had = x @ self.matrix
#         with torch.no_grad():
#             std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
#             x_norm = x_had / std
#             expanded_input = x_norm.unsqueeze(-1)
#             distances = torch.abs(expanded_input - self.levels)
#             indices = torch.argmin(distances, dim=-1)
#             xq_norm = self.levels[indices]
#             xq = xq_norm * std

#             mask = (torch.abs(xq - x_had) <= std * self.trust).float()

#         grad_flow_output = x_had * mask
#         return grad_flow_output + (xq - grad_flow_output).detach()


# class HadamardGaussianTrustQuantizer(GaussianTrustQuantizer):
#     aux_matrix = hadamard_transform(
#         torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
#     )

#     def __init__(self, bits=4, trust=None):
#         super().__init__(bits, trust)
#         self.matrix = None

#     def forward(self, x):
#         if self.matrix is None:
#             self.matrix = torch.block_diag(
#                 *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
#             )

#         x_had = x @ self.matrix
#         with torch.no_grad():
#             std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
#             x_norm = x_had / std
#             expanded_input = x_norm.unsqueeze(-1)
#             distances = torch.abs(expanded_input - self.levels)
#             indices = torch.argmin(distances, dim=-1)
#             xq_norm = self.levels[indices]
#             xq = xq_norm * std

#             mask = (torch.abs(xq - x_had) <= std * self.trust).float()
#             xq = xq @ self.matrix.T

#         grad_flow_output = (x_had * mask) @ self.matrix.T

#         return grad_flow_output + (xq - grad_flow_output).detach()


# FP4_LEVELS = [
#     -2.92247856,
#     -1.94831904,
#     -1.46123928,
#     -0.97415952,
#     -0.73061964,
#     -0.48707976,
#     -0.24353988,
#     0.0,
#     0.0,
#     0.24353988,
#     0.48707976,
#     0.73061964,
#     0.97415952,
#     1.46123928,
#     1.94831904,
#     2.92247856,
# ]


# class FP4STEQuantizer(GaussianSTEQuantizer):
#     def __init__(self):
#         super().__init__(4)
#         self.register_buffer("levels", torch.tensor(FP4_LEVELS))


# class FP4ClipQuantizer(GaussianClipQuantizer):
#     def __init__(self):
#         super().__init__(4)
#         self.register_buffer(
#             "levels",
#             torch.tensor(
#                 [
#                     -2.92247856,
#                     -1.94831904,
#                     -1.46123928,
#                     -0.97415952,
#                     -0.73061964,
#                     -0.48707976,
#                     -0.24353988,
#                     0.0,
#                     0.0,
#                     0.24353988,
#                     0.48707976,
#                     0.73061964,
#                     0.97415952,
#                     1.46123928,
#                     1.94831904,
#                     2.92247856,
#                 ]
#             ),
#         )


# class FP4TrustQuantizer(GaussianTrustQuantizer):
#     def __init__(self, trust=None):
#         super().__init__(4, trust)
#         self.register_buffer(
#             "levels",
#             torch.tensor(
#                 [
#                     -2.92247856,
#                     -1.94831904,
#                     -1.46123928,
#                     -0.97415952,
#                     -0.73061964,
#                     -0.48707976,
#                     -0.24353988,
#                     0.0,
#                     0.0,
#                     0.24353988,
#                     0.48707976,
#                     0.73061964,
#                     0.97415952,
#                     1.46123928,
#                     1.94831904,
#                     2.92247856,
#                 ]
#             ),
#         )


# class HalfHadamardFP4ClipQuantizer(HalfHadamardGaussianClipQuantizer):
#     def __init__(self):
#         super().__init__(4)
#         self.register_buffer(
#             "levels",
#             torch.tensor(
#                 [
#                     -2.92247856,
#                     -1.94831904,
#                     -1.46123928,
#                     -0.97415952,
#                     -0.73061964,
#                     -0.48707976,
#                     -0.24353988,
#                     0.0,
#                     0.0,
#                     0.24353988,
#                     0.48707976,
#                     0.73061964,
#                     0.97415952,
#                     1.46123928,
#                     1.94831904,
#                     2.92247856,
#                 ]
#             ),
#         )


# class HadamardFP4ClipQuantizer(HadamardGaussianClipQuantizer):
#     def __init__(self):
#         super().__init__(4)
#         self.register_buffer(
#             "levels",
#             torch.tensor(
#                 [
#                     -2.92247856,
#                     -1.94831904,
#                     -1.46123928,
#                     -0.97415952,
#                     -0.73061964,
#                     -0.48707976,
#                     -0.24353988,
#                     0.0,
#                     0.0,
#                     0.24353988,
#                     0.48707976,
#                     0.73061964,
#                     0.97415952,
#                     1.46123928,
#                     1.94831904,
#                     2.92247856,
#                 ]
#             ),
#         )


# class HalfHadamardFP4TrustQuantizer(HalfHadamardGaussianTrustQuantizer):
#     def __init__(self, trust=None):
#         super().__init__(4, trust)
#         self.register_buffer(
#             "levels",
#             torch.tensor(
#                 [
#                     -2.92247856,
#                     -1.94831904,
#                     -1.46123928,
#                     -0.97415952,
#                     -0.73061964,
#                     -0.48707976,
#                     -0.24353988,
#                     0.0,
#                     0.0,
#                     0.24353988,
#                     0.48707976,
#                     0.73061964,
#                     0.97415952,
#                     1.46123928,
#                     1.94831904,
#                     2.92247856,
#                 ]
#             ),
#         )

#         if trust is None:
#             trust = (self.levels[-1] - self.levels[-2]) / 2


# class HadamardFP4TrustQuantizer(HadamardGaussianTrustQuantizer):
#     def __init__(self, trust=None):
#         super().__init__(4, trust)
#         self.register_buffer(
#             "levels",
#             torch.tensor(
#                 [
#                     -2.92247856,
#                     -1.94831904,
#                     -1.46123928,
#                     -0.97415952,
#                     -0.73061964,
#                     -0.48707976,
#                     -0.24353988,
#                     0.0,
#                     0.0,
#                     0.24353988,
#                     0.48707976,
#                     0.73061964,
#                     0.97415952,
#                     1.46123928,
#                     1.94831904,
#                     2.92247856,
#                 ]
#             ),
#         )

#         if trust is None:
#             trust = (self.levels[-1] - self.levels[-2]) / 2


# class FourEightMaskedQuantizer(BaseQuantizer):
#     def __init__(self, p=2.0):
#         super().__init__(16)
#         self.p = p

#     def forward(self, x):
#         x_reshaped = x.reshape(-1, 4, 2)
#         _, idx = x_reshaped.norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
#         mask = torch.ones_like(x_reshaped, dtype=torch.bool)
#         mask[torch.arange(x_reshaped.size(0)).repeat(2, 1).T, idx, :] = False
#         mask = mask.reshape(x.shape).float()

#         return x * mask


# class FourEightSTEQuantizer(BaseQuantizer):
#     def __init__(self, bits=4, p: float = 2.0):
#         super().__init__(bits)
#         self.p = p

#     def forward(self, x):
#         scale = (
#             OPTIMAL_GAUSSIAN_SCALES[self.bits]
#             * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
#             + 1e-8
#         )

#         step = 2 * scale / (self.n_levels - 1)
#         x_clip = torch.clamp(x, -scale, scale)
#         xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

#         _, idx = (
#             x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
#         )
#         xq = xq.reshape(-1, 4, 2)
#         xq[
#             torch.arange(xq.size(0)).repeat(2, 1).T,
#             idx,
#         ] = 0.0
#         xq = xq.reshape(x.shape)

#         return x + (xq - x).detach()


# class FourEightClipQuantizer(FourEightSTEQuantizer):
#     def forward(self, x):
#         scale = (
#             OPTIMAL_GAUSSIAN_SCALES[self.bits]
#             * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
#             + 1e-8
#         )

#         step = 2 * scale / (self.n_levels - 1)
#         x_clip = torch.clamp(x, -scale, scale)
#         xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

#         _, idx = (
#             x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
#         )
#         xq = xq.reshape(-1, 4, 2)
#         xq[
#             torch.arange(xq.size(0)).repeat(2, 1).T,
#             idx,
#         ] = 0.0
#         xq = xq.reshape(x.shape)

#         mask = (torch.abs(x) <= scale).float()
#         return x * mask + (xq - x * mask).detach()


# class FourEightTrustQuantizer(FourEightSTEQuantizer):
#     def __init__(self, bits=4, trust=None, p: float = 2.0):
#         super().__init__(bits, p)
#         if trust is None:
#             trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
#         self.trust = trust

#     def forward(self, x):
#         std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
#         scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8

#         step = 2 * scale / (self.n_levels - 1)
#         x_clip = torch.clamp(x, -scale, scale)
#         xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

#         _, idx = (
#             x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
#         )
#         xq = xq.reshape(-1, 4, 2)
#         xq[
#             torch.arange(xq.size(0)).repeat(2, 1).T,
#             idx,
#         ] = 0.0
#         xq = xq.reshape(x.shape)

#         mask = (torch.abs(xq - x) <= std * self.trust).float()
#         return x * mask + (xq - x * mask).detach()


# class HalfHadamardFourEightTrustQuantizer(HadamardTrustQuantizer):
#     def __init__(self, bits=4, trust=None, p: float = 2.0):
#         super().__init__(bits, trust)
#         self.p = p

#     def forward(self, x):
#         if self.matrix is None:
#             self.matrix = torch.block_diag(
#                 *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
#             )

#         x_had = x @ self.matrix
#         with torch.no_grad():
#             std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
#             scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std

#             step = 2 * scale / (self.n_levels - 1)
#             x_clip = torch.clamp(x_had, -scale, scale)
#             xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

#             _, idx = (
#                 x_had.reshape(-1, 4, 2)
#                 .norm(p=self.p, dim=-1)
#                 .topk(k=2, dim=-1, largest=False)
#             )
#             xq = xq.reshape(-1, 4, 2)
#             xq[
#                 torch.arange(xq.size(0)).repeat(2, 1).T,
#                 idx,
#             ] = 0.0
#             xq = xq.reshape(x.shape)

#             mask = (torch.abs(xq - x_had) <= std * self.trust).float()

#         grad_flow_output = x_had * mask

#         return grad_flow_output + (xq - grad_flow_output).detach()


# class HadamardFourEightTrustQuantizer(HadamardTrustQuantizer):
#     def __init__(self, bits=4, trust=None, p: float = 2.0):
#         super().__init__(bits, trust)
#         self.p = p

#     def forward(self, x):
#         if self.matrix is None:
#             self.matrix = torch.block_diag(
#                 *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
#             )

#         x_had = x @ self.matrix
#         with torch.no_grad():
#             std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
#             scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std

#             step = 2 * scale / (self.n_levels - 1)
#             x_clip = torch.clamp(x_had, -scale, scale)
#             xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

#             _, idx = (
#                 x_had.reshape(-1, 4, 2)
#                 .norm(p=self.p, dim=-1)
#                 .topk(k=2, dim=-1, largest=False)
#             )
#             xq = xq.reshape(-1, 4, 2)
#             xq[
#                 torch.arange(xq.size(0)).repeat(2, 1).T,
#                 idx,
#             ] = 0.0
#             xq = xq.reshape(x.shape)

#             mask = (torch.abs(xq - x_had) <= std * self.trust).float()
#             xq = xq @ self.matrix.T

#         grad_flow_output = (x_had * mask) @ self.matrix.T

#         return grad_flow_output + (xq - grad_flow_output).detach()


# # torch._dynamo.config.optimize_ddp=False # uncommend if actually using ErfClipQuantizer
# class ErfFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, xq, buffer, mask):
#         ctx.save_for_backward(buffer, mask)
#         return xq

#     @staticmethod
#     def backward(ctx, grad_output):
#         buffer, mask = ctx.saved_tensors
#         mask = mask.float()

#         return (
#             (grad_output + buffer) * mask,
#             None,
#             grad_output * (1 - mask) - buffer * mask,
#             None,
#         )


# class ErfClipQuantizer(ClipQuantizer):
#     def __init__(self, bits=4, acc_dtype=torch.float32):
#         super().__init__(bits, True)
#         self.acc_dtype = acc_dtype
#         self.register_parameter("acc", None)

#     def forward(self, x):
#         with torch.no_grad():
#             if self.acc is None:
#                 self.acc = nn.Parameter(
#                     torch.zeros_like(x, dtype=self.acc_dtype), requires_grad=True
#                 )
#             elif self.acc.grad is not None:
#                 self.acc.data += self.acc.grad
#                 self.acc.grad = None

#         scale = (
#             OPTIMAL_GAUSSIAN_SCALES[self.bits]
#             * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
#             + 1e-8
#         )

#         step = 2 * scale / (self.n_levels - 1)
#         x_clip = torch.clamp(x, -scale, scale)
#         xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
#         mask = (torch.abs(x) <= scale).float()

#         return ErfFn().apply(x, xq, self.acc, mask)


# class FlushAccFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, acc):
#         ctx.save_for_backward(acc)
#         return x

#     @staticmethod
#     def backward(ctx, grad_output):
#         (acc,) = ctx.saved_tensors
#         return grad_output + acc, None


# class ClipAccQuantizer(STEQuantizer):
#     def __init__(
#         self,
#         bits=4,
#         centered=True,
#         flush_every: int = 64,
#         acc_dtype=torch.float32,
#         scale: float = None,
#     ):
#         super().__init__(bits, centered)

#         if scale is None:
#             scale = 1 / flush_every

#         self.acc_dtype = acc_dtype
#         self.flush_every = flush_every
#         self.counter = 0
#         self.scale = scale
#         self.register_buffer("acc", None)

#     def forward(self, x):
#         with torch.no_grad():
#             if self.counter == 0:
#                 if self.acc is None:
#                     self.acc = torch.zeros_like(x, dtype=self.acc_dtype)
#                 else:
#                     self.acc.data = torch.zeros_like(x, dtype=self.acc_dtype)

#         scale = (
#             OPTIMAL_GAUSSIAN_SCALES[self.bits]
#             * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
#             + 1e-8
#         )
#         if self.centered:
#             step = 2 * scale / (self.n_levels - 1)
#             x_clip = torch.clamp(x, -scale, scale)
#             xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
#             mask = (torch.abs(x) <= scale).float()
#         else:
#             neg_scale = -scale * (self.n_levels - 2)
#             step = 2 * scale / self.n_levels
#             x_clip = torch.clamp(x, neg_scale, scale)
#             xq = torch.round(x_clip / step) * step
#             mask = ((neg_scale <= x) & (x <= scale)).float()

#         self.counter += 1
#         if self.counter == self.flush_every:
#             self.counter = 0
#             grad_flow_output = FlushAccFn().apply(
#                 x * mask + x * (1 - mask) * self.scale,
#                 (self.acc * self.scale).to(x.dtype),
#             )
#         else:
#             grad_flow_output = x * mask + self.acc * (1 - mask)

#         return grad_flow_output + (xq - grad_flow_output).detach()


# class LSQQuantizer(nn.Module):
#     """
#     Implementation of LSQ quantizer from https://arxiv.org/abs/1902.08153
#     LSQ uses a learnable step size for quantization. This learnable step size(alpha) is initialized using the optimal gaussian scale
#     ans must be normalized with a weight decay.
#     """

#     def __init__(self, bits=4, raise_zero=True, all_positive=False, **kwargs):
#         super().__init__()
#         # NOTE: raise_zero should never be used with FP quantization

#         self.bits = bits
#         self.n_levels = 2**bits
#         self.all_positive = all_positive
#         self.raise_zero = raise_zero

#         self.q_min, self.q_max = self.get_dtype_bounds()

#         self.is_alpha_init = False
#         self.alpha_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

#     def get_dtype_bounds(self):
#         if not self.all_positive:
#             q_min = -self.n_levels / 2
#             q_max = self.n_levels / 2 - 1
#         else:
#             q_min = 0
#             q_max = self.n_levels - 1
#         return q_min, q_max

#     def cast(self, x):
#         # This method can be inherited to use any casting, e.g. int, fp(e2m1, e1m2,...), optimal gaussian, etc.
#         # NOTE: raise_zero should never be used with FP quantization
#         return x.round()

#     def ste_cast(self, x):
#         return (self.cast(x) - x).detach() + x

#     def grad_scale(self, x, scale):
#         return (x - x * scale).detach() + x * scale

#     @torch.no_grad()
#     def get_initial_step_value(self, x):
#         return (
#             torch.mean(torch.abs(x.detach())) * 2 / (np.sqrt(self.q_max))
#         )  # LSQ initialization

#     def get_learnable_step(self, x):
#         if not self.is_alpha_init:
#             with torch.no_grad():
#                 step = self.get_initial_step_value(x)
#                 self.alpha_weight.data.multiply_(
#                     torch.tensor(
#                         step,
#                         dtype=self.alpha_weight.dtype,
#                         device=self.alpha_weight.device,
#                     )
#                 )
#             self.is_alpha_init = True
#         return self.alpha_weight

#     def forward(self, x):
#         step = self.get_learnable_step(x)
#         step = self.grad_scale(step, 1.0 / np.sqrt(x.numel() * self.q_max))
#         xs = x / step
#         if self.raise_zero:
#             xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
#             xscr = self.ste_cast(xsc) + 1 / 2
#         else:
#             xsc = torch.clamp(xs, self.q_min, self.q_max)
#             xscr = self.ste_cast(xsc)
#         xq = xscr * step

#         return xq + step * 1e-9  # extra term to ensure gradient flow


# class LSQPlusWeightQuantizer(LSQQuantizer):
#     @torch.no_grad()
#     def get_initial_step_value(self, x):
#         scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * torch.sqrt(torch.mean(x**2)) + 1e-8
#         step = 2 * scale / (self.n_levels - 1)
#         return step


# class LSQPlusActivationQuantizer(LSQPlusWeightQuantizer):
#     def __init__(self, bits=4, raise_zero=True, all_positive=False, **kwargs):
#         super().__init__(bits, raise_zero, all_positive, **kwargs)
#         self.beta_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
#         self.is_beta_init = False

#     @torch.no_grad()
#     def get_initial_bias_value(self, x):
#         return x.min() - self.alpha_weight * self.q_min

#     def get_learnable_bias(self, x):
#         if not self.is_beta_init:
#             with torch.no_grad():
#                 bias = self.get_initial_bias_value(x)
#                 self.beta_weight.data.add_(
#                     torch.tensor(
#                         bias,
#                         dtype=self.beta_weight.dtype,
#                         device=self.beta_weight.device,
#                     )
#                 )
#             self.is_beta_init = True
#         return self.beta_weight

#     def forward(self, x):
#         step = self.get_learnable_step(x)
#         step = self.grad_scale(step, 1.0 / np.sqrt(x.numel() * self.q_max))
#         bias = self.get_learnable_bias(x)
#         bias = self.grad_scale(bias, 1.0 / np.sqrt(x.numel() * self.q_max))
#         xs = (x - bias) / step
#         if self.raise_zero:
#             xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
#             xscr = self.ste_cast(xsc) + 1 / 2
#         else:
#             xsc = torch.clamp(xs, self.q_min, self.q_max)
#             xscr = self.ste_cast(xsc)
#         xq = xscr * step + bias
#         return xq + step * 1e-9  # extra term to ensure gradient flow


# class PACTQuantizer(LSQQuantizer):
#     """
#     Implementation of PACT quantizer from https://arxiv.org/abs/1805.06085
#     PACT and LSQ are quite similar and do the same thing for forward pass.
#     The difference is in the backward pass where PACT does not perform a full gradient flow.
#     """

#     def forward(self, x):
#         step = self.get_learnable_step(x)
#         xs = x / step
#         if self.raise_zero:
#             xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
#             with torch.no_grad():
#                 clamp_mask = ~torch.isclose(xsc, xs - 1 / 2)
#             xscr = self.ste_cast(xsc) + 1 / 2
#         else:
#             xsc = torch.clamp(xs, self.q_min, self.q_max)
#             with torch.no_grad():
#                 clamp_mask = ~torch.isclose(xsc, xs)
#             xscr = self.ste_cast(xsc)
#         xq = xscr * step
#         xq = xq * clamp_mask + (xq - xq * clamp_mask).detach()
#         return xq + step * 1e-9  # extra term to ensure gradient flow


QUANTIZER_CLASSES = {
    "NoQuantizer": NoQuantizer,
    "HalfHadamardTrustQuantizer": HalfHadamardTrustQuantizer,
    "QuESTImproved": QuESTImproved,
    "BBQ": BBQ,
    "QuESTTGCS": QuESTTGCS,
    "BBQTGCS": BBQTGCS,
    "LSQ": LSQ,
    "QuESTPQN": QuESTPQN,
    "BBQPQN": BBQPQN,
    # "UniformQuantizer": UniformQuantizer,
    # "STEQuantizer": STEQuantizer,
    # "ClipQuantizer": ClipQuantizer,
    # "HalfHadamardClipQuantizer": HalfHadamardClipQuantizer,
    # "HadamardClipQuantizer": HadamardClipQuantizer,
    # "TrustQuantizer": TrustQuantizer,
    # "HalfHadamardTrustQuantizer": HalfHadamardTrustQuantizer,
    # "HadamardTrustQuantizer": HadamardTrustQuantizer,
    # "GaussianSTEQuantizer": GaussianSTEQuantizer,
    # "GaussianClipQuantizer": GaussianClipQuantizer,
    # "GaussianTrustQuantizer": GaussianTrustQuantizer,
    # "HadamardGaussianClipQuantizer": HadamardGaussianClipQuantizer,
    # "HalfHadamardGaussianTrustQuantizer": HalfHadamardGaussianTrustQuantizer,
    # "HadamaardGaussianTrustQuantizer": HadamardGaussianTrustQuantizer,
    # "FP4STEQuantizer": FP4STEQuantizer,
    # "FP4ClipQuantizer": FP4ClipQuantizer,
    # "FP4TrustQuantizer": FP4TrustQuantizer,
    # "HalfHadamardFP4ClipQuantizer": HalfHadamardFP4ClipQuantizer,
    # "HadamardFP4ClipQuantizer": HadamardFP4ClipQuantizer,
    # "HalfHadamardFP4TrustQuantizer": HalfHadamardFP4TrustQuantizer,
    # "HadamardFP4TrustQuantizer": HadamardFP4TrustQuantizer,
    # "FourEightMaskedQuantizer": FourEightMaskedQuantizer,
    # "FourEightSTEQuantizer": FourEightSTEQuantizer,
    # "FourEightClipQuantizer": FourEightClipQuantizer,
    # "FourEightTrustQuantizer": FourEightTrustQuantizer,
    # "HalfHadamardFourEightTrustQuantizer": HalfHadamardFourEightTrustQuantizer,
    # "HadamardFourEightTrustQuantizer": HadamardFourEightTrustQuantizer,
    # "ErfClipQuantizer": ErfClipQuantizer,
    # "ClipAccQuantizer": ClipAccQuantizer,
    # "PACTQuantizer": PACTQuantizer,
    # "LSQQuantizer": LSQQuantizer,
    # "LSQPlusActivationQuantizer": LSQPlusActivationQuantizer,
    # "LSQPlusWeightQuantizer": LSQPlusWeightQuantizer,
}


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        weight_quantizer=None,
        activation_quantizer=None,
        **kwargs
    ):
        super().__init__(in_features, out_features, **kwargs)
        if weight_quantizer is None:
            weight_quantizer = NoQuantizer()
        if activation_quantizer is None:
            activation_quantizer = NoQuantizer()
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

    def forward(self, x):
        if isinstance(self.activation_quantizer, LSQ):
            x = self.activation_quantizer(x)
        else:
            # quantize activations and preserve token-wise L2 norms
            # prevent quantization from interfering with the guarantess of spectral norm hyperpshere optimizers
            original_x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
            x = self.activation_quantizer(x)
            x = x * (original_x_norm / (torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-8))
        if isinstance(self.weight_quantizer, LSQ):
            w = self.weight_quantizer(self.weight)
        else:
            # quantize weights and preserve frobenius norms
            # prevent quantization from interfering with the guarantess of frobenius norm hyperpshere optimizers
            original_w_norm = torch.linalg.norm(self.weight)
            w = self.weight_quantizer(self.weight)
            w = w * (original_w_norm / (torch.linalg.norm(w) + 1e-8))
        return F.linear(x, w, self.bias)
