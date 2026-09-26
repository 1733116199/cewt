import torch
from base_linear import my_hadamard_transform

@torch.no_grad()
@torch.jit.script
def project_to_gaussian(x: torch.Tensor, expected_norm: torch.Tensor) -> torch.Tensor:
    indices = torch.argsort(x.flatten(), descending=False)
    order = torch.zeros_like(indices)
    order[indices] = torch.arange(indices.numel(), device=x.data.device, dtype=order.dtype)
    order = ((order.to(torch.float64) + 0.5) * (2 / order.numel()) - 1)
    approx = (torch.erfinv(order) * (2 ** 0.5)).to(torch.float32)
    approx = approx.view_as(x) * (expected_norm / torch.linalg.norm(approx))
    return approx
