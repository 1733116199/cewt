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
def my_hadamard_transform(x: torch.Tensor, logblocksize: int = 6):
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
    def __init__(self, bits=2, log_blocksize=6, **kwargs):
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
        log_blocksize: int=6,
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
    def __init__(self, bits=4, log_blocksize=6, trust=None, **kwargs):
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


QUANTIZER_CLASSES = {
    "NoQuantizer": NoQuantizer,
    "HalfHadamardTrustQuantizer": HalfHadamardTrustQuantizer,
    "QuESTImproved": QuESTImproved,
    "BBQ": BBQ,
    "QuESTTGCS": QuESTTGCS,
    "BBQTGCS": BBQTGCS,
    "LSQ": LSQ,
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
            original_x_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
            x = self.activation_quantizer(x)
            x = x * (original_x_norm / (torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-8))
        if isinstance(self.weight_quantizer, LSQ):
            w = self.weight_quantizer(self.weight)
        else:
            # quantize weights and preserve frobenius norms
            original_w_norm = torch.linalg.norm(self.weight)
            w = self.weight_quantizer(self.weight)
            w = w * (original_w_norm / (torch.linalg.norm(w) + 1e-8))
        return F.linear(x, w, self.bias)
