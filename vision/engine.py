# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from base_linear import QuantizedLinear
import wandb

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, not_compiled_model:torch.nn.Module=None, lr_scheduler=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    global_iter = epoch * len(data_loader)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        global_iter += 1
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        
        loss.backward()
        
        # log data
        if utils.get_rank() == 0 and global_iter % 50 == 0:
            param_norms = {}
            grad_norms = {}
            entropies = {}
            wscales = {}
            ascales = {}
            with torch.no_grad():
                for name, parameter in not_compiled_model.named_parameters():
                    if parameter.requires_grad:
                        assert parameter.grad is not None
                        param_norms[name] = torch.linalg.norm(parameter.data, dtype=torch.float32)
                        grad_norms[name] = torch.linalg.norm(parameter.grad, dtype=torch.float32)
                for name, module in not_compiled_model.named_modules():
                    if isinstance(module, QuantizedLinear):
                        if module.weight_quantizer is not None and hasattr(module.weight_quantizer, "entropy"):
                            entropies[name] = module.weight_quantizer.entropy(module.weight.data)
                for name, parameter in not_compiled_model.named_parameters():
                    if name.endswith("weight_quantizer.scale"):
                        wscales[name] = parameter.mean().item()
                    elif name.endswith("activation_quantizer.scale"):
                        ascales[name] = parameter.mean().item()
                    
        # clip grad and step
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        lr_scheduler.step()
        
        for name, module in not_compiled_model.named_modules():
            if isinstance(module, QuantizedLinear):
                if hasattr(module.weight_quantizer, "clip_min_scale"):
                    module.weight_quantizer.clip_min_scale(module.weight)

        # log optimizer data
        if utils.get_rank() == 0 and global_iter % 50 == 0:
            with torch.no_grad():
                norm_grad_rms = {}
                update_rms = {}
                param_rms = {}
                update_ratio = {}
                for name, parameter in not_compiled_model.named_parameters():
                    if parameter.requires_grad:
                        state = optimizer.state[parameter]
                        norm_grad_rms[name] = state.get("norm_grad_rms", 0.0)
                        update_rms[name] = state.get("update_rms", 0.0)
                        param_rms[name] = state.get("param_rms", 0.0)
                        update_ratio[name] = state.get("update_ratio", 0.0)
                        
                rbm = {}
                total_oscillating_weights = 0
                num_weights = 0
                for name, module in not_compiled_model.named_modules():
                    if isinstance(module, QuantizedLinear):
                        if hasattr(module.weight_quantizer, "quantize"):
                            pre_clip_round: torch.Tensor = module.weight_quantizer.quantize(module.weight)[1]
                            pre_clip_round = pre_clip_round.flatten()
                            closest_boundary = torch.floor(pre_clip_round) + 0.5
                            is_oscillating = torch.abs(pre_clip_round - closest_boundary) < 0.005
                            rbm[name] = is_oscillating.sum().item() / pre_clip_round.numel()
                            total_oscillating_weights += is_oscillating.sum().item()
                            num_weights += pre_clip_round.numel()
                if num_weights > 0:
                    rbm["total"] = total_oscillating_weights / num_weights
            
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if utils.get_rank() == 0 and global_iter % 50 == 0:
            wandb.log(
                {
                    "iter": global_iter, 
                    "lr": optimizer.param_groups[0]["lr"],
                    "matrix_lr": optimizer.param_groups[1]["lr"],
                 }
                | {f"norm_grad_rms/{name}": norm for name, norm in norm_grad_rms.items()}  # log RMS of normalized gradients
                | {f"update_rms/{name}": rms for name, rms in update_rms.items()}  # log RMS of updates
                | {f"param_rms/{name}": rms for name, rms in param_rms.items()}  # log RMS of parameters
                | {f"update_ratio/{name}": ratio for name, ratio in update_ratio.items()}  # log update-to-parameter ratio
                | {f"param_norm/{name}": norm for name, norm in param_norms.items()}  # log parameter norms
                | {f"grad_norm/{name}": norm for name, norm in grad_norms.items()}  # log gradient norms
                | {f"entropy/{name}": entropy for name, entropy in entropies.items()}  # log quantization entropies
                | {f"wscale/{name}": scale for name, scale in wscales.items()}  # log weight quantizer scales
                | {f"ascale/{name}": scale for name, scale in ascales.items()}  # log activation quantizer scales
                | {f"rbm/{name}": ratio for name, ratio in rbm.items()}  # log ratio of oscillating weights
            )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
