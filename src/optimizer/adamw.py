# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version
from .project import project_to_gaussian, my_hadamard_transform

class AdamW(Optimizer):

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        hyperball_optimization: bool = False,
        constraint: str = "none",
        log_blocksize: int = 0,
    ):
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        if log_blocksize < 0:
            raise ValueError(f"Invalid log_blocksize value: {log_blocksize} - should be >= 0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "hyperball_optimization": hyperball_optimization, "constraint": constraint, "log_blocksize": log_blocksize}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                original_param: torch.nn.Parameter = p
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                
                state = self.state[p]
                
                if group["hyperball_optimization"] and "original_norm" not in state:
                    state["original_norm"] = torch.linalg.norm(p.data, dtype=torch.float32)
                
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = group["lr"]  / bias_correction1
                
                bias_correction2_sqrt = bias_correction2**0.5
                
                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])

                # compute norm gradient
                norm_grad = exp_avg / denom
                if group["lr"] > 0.0:
                    norm_grad.mul_(step_size / group["lr"])
                    
                if group["log_blocksize"] > 0:
                    p = my_hadamard_transform(p, group["log_blocksize"])
                    norm_grad = my_hadamard_transform(norm_grad, group["log_blocksize"])
                
                if group["hyperball_optimization"]:
                    norm_grad_norm = torch.linalg.norm(norm_grad, dtype=torch.float32)
                    norm_grad.mul_(state["original_norm"] / (norm_grad_norm + 1e-10))
                
                state["norm_grad_rms"] = norm_grad.square().mean().sqrt()
                state["update_rms"] = (norm_grad * group["lr"]).square().mean().sqrt()
                state["param_rms"] = p.square().mean().sqrt()
                state["update_ratio"] = state["update_rms"] / (state["param_rms"] + 1e-10)
                
                if group["hyperball_optimization"]:
                    p.add_(norm_grad, alpha=-group["lr"])
                    if group["constraint"] == "gaussian":
                        p.data.copy_(project_to_gaussian(p.data, state["original_norm"]))
                    elif group["constraint"] == "none":
                        param_norm = torch.linalg.norm(p.data, dtype=torch.float32)
                        p.mul_(state["original_norm"] / (param_norm + 1e-10))
                    elif group["constraint"] == "debug_no_retraction":
                        pass
                    elif group["constraint"] == "debug_gaussian":
                        state["pre_projection"] = p.data.clone()
                        p.data.copy_(project_to_gaussian(p.data, state["original_norm"]))
                    else:
                        assert False, f"Invalid constraint type: {group['constraint']}"
                else:
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(norm_grad, alpha=-group["lr"])
                    
                if group["log_blocksize"] > 0:
                    original_param.data.copy_(my_hadamard_transform(p, group["log_blocksize"]))

        return loss