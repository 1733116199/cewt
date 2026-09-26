# Copyright 2023 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PyTorch implementation of the Lion optimizer."""
import torch
from torch.optim.optimizer import Optimizer
from .project import project_to_gaussian, my_hadamard_transform

# https://github.com/google/automl/blob/6a54c8741e7c3265d4547c4f35f47a0391122dc5/lion/lion_pytorch.py#L1-L86

class Lion(Optimizer):
  r"""Implements Lion algorithm."""

  def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, linear_lr_scale=0.2, hyperball_optimization=False, constraint: str = "none", log_blocksize: int = 0):
    """Initialize the hyperparameters.

    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
      linear_lr_scale (float, optional): linear learning rate scale (default: 0.2)
      hyperball_optimization (bool, optional): whether to use hyperball optimization (default: False)
      constraint (str, optional): the type of constraint to apply (default: "none")
    """

    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    if log_blocksize < 0:
        raise ValueError(f"Invalid log_blocksize value: {log_blocksize} - should be >= 0")
    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, linear_lr_scale=linear_lr_scale, hyperball_optimization=hyperball_optimization, constraint=constraint, log_blocksize=log_blocksize)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.

    Returns:
      the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        original_param: torch.nn.Parameter = p
        if p.grad is None:
          continue

        grad = p.grad
        state = self.state[p]
        # State initialization
        if len(state) == 0:
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p)
        
        if group['hyperball_optimization'] and 'original_norm' not in state:
          state['original_norm'] = torch.linalg.norm(p.data, dtype=torch.float32)

        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']
        linear_lr_scale = group['linear_lr_scale']

        # Weight update
        update = exp_avg * beta1 + grad * (1 - beta1)
        update = update.sign_() * linear_lr_scale
        
        if group["log_blocksize"] > 0:
          p = my_hadamard_transform(p, group["log_blocksize"])
          update = my_hadamard_transform(update, group["log_blocksize"])
            
        if group["hyperball_optimization"]:
          update_norm = torch.linalg.norm(update, dtype=torch.float32)
          update.mul_(state['original_norm'] / (update_norm + 1e-10))

        state["norm_grad_rms"] = update.square().mean().sqrt()
        state["update_rms"] = (update * group['lr']).square().mean().sqrt()
        state["param_rms"] = p.square().mean().sqrt()
        state["update_ratio"] = state["update_rms"] / (state["param_rms"] + 1e-10)
        
        if group["hyperball_optimization"]:
          p.add_(update, alpha=-group['lr'])
          if group["constraint"] == "gaussian":
            p.data.copy_(project_to_gaussian(p.data, state["original_norm"]))
          elif group["constraint"] == "none":
            param_norm = torch.linalg.norm(p.data, dtype=torch.float32)
            p.mul_(state['original_norm'] / (param_norm + 1e-10))
          else:
            assert False, f"Invalid constraint type: {group['constraint']}"
        else:
          if group["weight_decay"] > 0.0:
            p.data.mul_(1 - group['lr'] * group['weight_decay'])
          p.add_(update, alpha=-group['lr'])
          
        if group["log_blocksize"] > 0:
          original_param.data.copy_(my_hadamard_transform(p, group["log_blocksize"]))

        # Decay the momentum running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    return loss