from contextlib import nullcontext
import copy
from pathlib import Path
import time
import yaml

import torch
import wandb
from tqdm import tqdm
import math

from logger.logger import DynamicsLogger
from models.quantization.base_linear import QuantizedLinear
from models.quantization.base_linear import my_hadamard_transform, QuESTImproved
from optim.weight_averaging import (
    WeightAverager,
    eval_ema,
    eval_wa,
    ExponentialWeightAverager,
)
from .utils import (
    eval,
    get_batch,
    load_checkpoint,
    load_worker_state,
    save_checkpoint,
    save_worker_state,
)


def train(
    model,
    opt,
    matrix_opt,
    datareaders,
    scheduler,
    matrix_scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
    with torch.no_grad():
        print(f"Send dummy input into model ...")
        model.eval()
        x, y = datareaders["train"].sample_batch()
        outputs = model(x, targets=y)
        print(f"Dummy input sent to model.")
    model.train()
    not_compiled_model = model
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    ema_oscillation_frequencies = None
    last_flip_directions = None
    if cfg.resume_from:
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not indended for fine tuning or
        # other scenarios where some of these should change.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter, ema_oscillation_frequencies, last_flip_directions = load_checkpoint(
            model,
            opt,
            matrix_opt,
            scheduler,
            matrix_scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
        )
        load_worker_state(ckpt_dir)
    else:
        curr_iter = 0
        if cfg.log_ema_oscillation_frequency:
            with torch.no_grad():
                ema_oscillation_frequencies = {}
                last_flip_directions = {}
                
                for name, module in not_compiled_model.named_modules():
                    if isinstance(module, QuantizedLinear):
                        ema_oscillation_frequencies[name] = torch.zeros_like(module.weight.data)
                        last_flip_directions[name] = torch.zeros_like(module.weight.data, dtype=torch.int32)
                        
    if cfg.log_ema_oscillation_frequency:
        if ema_oscillation_frequencies is None or last_flip_directions is None:
            raise ValueError("Checkpoint is missing oscillation-frequency state")

    if cfg.weight_average:
        # This does generally not support resuming training, but will work if
        # cfg.wa_interval perfectly divides the iteration number of the chkpt.
        # Otherwise, the first avg will not be correctly computed, with a bias
        # towards the first sample and missing values for earlier iterations.
        weight_averager = WeightAverager(
            not_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            save_dir=None if cfg.wa_use_temp_dir else exp_dir / "avgs",
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            count=curr_iter,
        )

    if cfg.exponential_moving_average:
        ema = ExponentialWeightAverager(
            not_compiled_model,
            interval=cfg.ema_interval,
            decay=cfg.ema_decay,
            warmup=cfg.warmup_steps if cfg.ema_after_warmup else 0,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
        )

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        assert False, "Dynamics logging is currently disabled."
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter

    substep = curr_iter * cfg.acc_steps
    train_reader, val_reader = datareaders["train"], datareaders["val"]
    train_reader.set_step(substep)
    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    model.train()

    # Initialize the progress bar
    if distributed_backend.is_master_process():
        pbar = tqdm(total=cfg.iterations, desc="Training Progress", initial=curr_iter)
    else:
        pbar = None
    
    while curr_iter <= cfg.iterations:
        # Save permanent checkpoint
        if cfg.permanent_ckpt_interval > 0:
            if curr_iter % cfg.permanent_ckpt_interval == 0:
                ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, matrix_opt, scheduler, matrix_scheduler, curr_iter, ckpt_dir, ema_oscillation_frequencies, last_flip_directions)
                save_worker_state(ckpt_dir)

        # Save temporary checkpoint for resuming training
        if cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = exp_dir / "ckpts" / "latest"
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, matrix_opt, scheduler, matrix_scheduler, curr_iter, ckpt_dir, ema_oscillation_frequencies, last_flip_directions)
                save_worker_state(ckpt_dir)

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        if (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        ):
            eval_and_log(
                curr_iter,
                epoch,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                full_eval=(curr_iter in cfg.full_eval_at),
            )

            if curr_iter > cfg.wa_interval and cfg.weight_average:
                eval_wa(
                    curr_iter,
                    not_compiled_model,
                    weight_averager,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )
            if cfg.exponential_moving_average:
                eval_ema(
                    curr_iter,
                    not_compiled_model,
                    ema,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # Train model
        t_start = time.perf_counter_ns()
        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = get_batch(train_reader, device=cfg.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    outputs = model(x, targets=y)

            loss = outputs["loss"] / cfg.acc_steps
            reg = 0
            if cfg.dl_alpha > 0.0:
                iter_ratio = curr_iter / cfg.iterations
                cos_ticks = -(math.cos(iter_ratio * torch.pi) - 1) / 2
                curr_dl_alpha = cfg.dl_alpha * cos_ticks
                for name, module in not_compiled_model.named_modules():
                    if isinstance(module, QuantizedLinear):
                        assert isinstance(module.weight_quantizer, QuESTImproved)
                        postround, preround, step = module.weight_quantizer.quantize(module.weight, detach_step=True)
                        reg = reg + torch.square(postround.detach() * step - preround * step).sum()
                reg = reg * curr_dl_alpha / cfg.acc_steps
            total_loss = loss + reg
            total_loss.backward()
            substep += 1
            
        collect_stats = (
            cfg.log_interval
            and (curr_iter + 1) % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
            and cfg.wandb
        )
        
        if collect_stats:
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
        
        if cfg.log_ema_oscillation_frequency:
            with torch.no_grad():
                my_integer_before = {}
                for name, module in not_compiled_model.named_modules():
                    if isinstance(module, QuantizedLinear):
                        if hasattr(module.weight_quantizer, "quantize"):
                            my_integer_before[name] = module.weight_quantizer.quantize(module.weight)[0].int()
            
        pi_norm_ratios = {}
        if cfg.pi_alpha > 0.0:
            with torch.no_grad():
                for name, module in not_compiled_model.named_modules():
                    if isinstance(module, QuantizedLinear):
                        assert isinstance(module.weight_quantizer, QuESTImproved)
                        if collect_stats:
                            before_norm = torch.linalg.norm(module.weight.data)
                        weight_quantizer = module.weight_quantizer
                        weight = module.weight
                        pi_alpha = cfg.pi_alpha
                        hw = my_hadamard_transform(weight.data, weight_quantizer.log_blocksize)
                        weight.data.copy_(
                            my_hadamard_transform(
                                (1 - pi_alpha) * hw + pi_alpha * weight_quantizer(weight.data),
                                weight_quantizer.log_blocksize
                            )
                        )
                        if collect_stats:
                            pi_norm_ratios[name] = torch.linalg.norm(module.weight.data) / before_norm

        if cfg.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        matrix_opt.step()
        scheduler.step()
        matrix_scheduler.step()
        opt.zero_grad(set_to_none=True)
        matrix_opt.zero_grad(set_to_none=True)
        
        for name, module in not_compiled_model.named_modules():
            if isinstance(module, QuantizedLinear):
                if hasattr(module.weight_quantizer, "clip_min_scale"):
                    # when using Muon/SSO, some of the learned scales of LSQ can decrease significantly, 
                    # likely due to numerical issues of bf16 training
                    # this adds a lower bound to the learned scales of LSQ using QuEST's scaling factor.
                    module.weight_quantizer.clip_min_scale(module.weight)
        
        if cfg.log_ema_oscillation_frequency:
            with torch.no_grad():
                total_ema_oscillation_frequency = 0.0
                num_ema_oscillation_frequencies = 0
                for name, module in not_compiled_model.named_modules():
                    if isinstance(module, QuantizedLinear):
                        if hasattr(module.weight_quantizer, "quantize"):
                            my_integer_after = module.weight_quantizer.quantize(module.weight)[0].int()
                            integer_update_direction = my_integer_after - my_integer_before[name]
                            has_flipped = torch.logical_and(
                                torch.sign(integer_update_direction).int() != torch.sign(last_flip_directions[name]).int(),
                                integer_update_direction != 0
                            )
                            ema_oscillation_frequencies[name] = cfg.ema_oscillation_frequency_momentum * has_flipped.float() + (1 - cfg.ema_oscillation_frequency_momentum) * ema_oscillation_frequencies[name]
                            last_flip_directions[name][has_flipped] = integer_update_direction[has_flipped]
                            total_ema_oscillation_frequency += ema_oscillation_frequencies[name].sum().to(torch.float64)
                            num_ema_oscillation_frequencies += ema_oscillation_frequencies[name].numel()
                if num_ema_oscillation_frequencies > 0:
                    ema_oscillation_frequencies["total"] = total_ema_oscillation_frequency / num_ema_oscillation_frequencies
                
        if collect_stats:
            with torch.no_grad():
                update_ratios = {}
                norm_grad_rmss = {}
                update_rmss = {}
                param_rmss = {}
                spectral_norms = {}
                grad_scaling_Rs = {}
                total_grad_scaling_R = 0.0
                num_grad_scaling_R = 0
                for name, parameter in not_compiled_model.named_parameters():
                    if parameter.requires_grad:
                        if parameter in matrix_opt.state:
                            state = matrix_opt.state[parameter]
                        else:
                            state = opt.state[parameter]
                        param_rmss[name] = state.get("param_rms", 0.0)
                        update_ratios[name] = state.get("update_ratio", 0.0)
                        norm_grad_rmss[name] = state.get("norm_grad_rms", 0.0)
                        update_rmss[name] = state.get("update_rms", 0.0)
                        if "spectral_norm" in state:
                            spectral_norms[name] = state.get("spectral_norm", 0.0)
                        if "grad_scaling_R" in state:
                            grad_scaling_Rs[name] = state.get("grad_scaling_R", 0.0)
                            total_grad_scaling_R += state.get("grad_scaling_R", 0.0) * parameter.numel()
                            num_grad_scaling_R += parameter.numel()
                if num_grad_scaling_R > 0:
                    grad_scaling_Rs["total"] = total_grad_scaling_R / num_grad_scaling_R

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

        if cfg.weight_average:
            weight_averager.step(
                not_compiled_model, distributed_backend.is_master_process()
            )
        if cfg.exponential_moving_average:
            ema.step(not_compiled_model, distributed_backend.is_master_process())
        dt = (time.perf_counter_ns() - t_start) / 1e9

        curr_iter += 1
        if distributed_backend.is_master_process():
            pbar.update(1)

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        ):
            train_loss = loss.detach().cpu().item() * cfg.acc_steps

            current_lrs = [param_group["lr"] for param_group in opt.param_groups]
            current_matrix_lrs = [param_group["lr"] for param_group in matrix_opt.param_groups]

            print(
                f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"lr={current_lrs[0]:.2e} "
                f"matrix_lr={current_matrix_lrs[0]:.2e}"
            )

            if cfg.wandb:
                wandb.log(
                    {
                        "iter": curr_iter,
                        "train/loss": train_loss,
                        "train/reg": reg / curr_dl_alpha * cfg.acc_steps if cfg.dl_alpha > 0.0 else 0.0,
                        "train/total_loss": total_loss * cfg.acc_steps,
                        "train/curr_dl_alpha": curr_dl_alpha if cfg.dl_alpha > 0.0 else 0.0,
                        "train/perplexity": 2.71828**train_loss,
                        "lr": current_lrs[0],
                        "matrix_lr": current_matrix_lrs[0],
                        "iter_dt": dt,
                    } | {f"param_rms/{name}": rms for name, rms in param_rmss.items()}
                      | {f"norm_grad_rms/{name}": rms for name, rms in norm_grad_rmss.items()}
                      | {f"update_rms/{name}": rms for name, rms in update_rmss.items()}
                      | {f"update_ratio/{name}": ratio for name, ratio in update_ratios.items()}
                      | {f"param_norm/{name}": norm for name, norm in param_norms.items()}  # log parameter norms
                      | {f"grad_norm/{name}": norm for name, norm in grad_norms.items()}  # log gradient norms
                      | {f"entropy/{name}": entropy for name, entropy in entropies.items()}  # log quantization entropies
                      | {f"wscale/{name}": scale for name, scale in wscales.items()}  # log weight quantizer scales
                      | {f"ascale/{name}": scale for name, scale in ascales.items()}  # log activation quantizer scales
                      | {f"rbm/{name}": rbm_value for name, rbm_value in rbm.items()}  # log RBM values
                      | {f"spectral_norm/{name}": sn for name, sn in spectral_norms.items()}  # log spectral norms
                      | {f"grad_scaling_R/{name}": R for name, R in grad_scaling_Rs.items()}  # log grad scaling R values
                      | {f"pi_norm_ratio/{name}": ratio for name, ratio in pi_norm_ratios.items()}  # log pi norm ratios
                      | ({} if not cfg.log_ema_oscillation_frequency else {f"ema_osc_freq/{name}": freq.mean() for name, freq in ema_oscillation_frequencies.items()})  # log EMA oscillation frequencies
                )
    return stats


def eval_and_log(
    curr_iter,
    epoch,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    model.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        cfg=cfg,
    )

    print(
        f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "iter": curr_iter,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
            }
        else:
            logs = {
                "iter": curr_iter,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
            }

        wandb.log(logs)
        if cfg.eval_seq_prefix != "none" and (
            curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
        ):
            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

            out_str = distributed_backend.get_raw_model(model).generate_from_string(
                cfg.eval_seq_prefix,
                max_new_tokens=40,
                temperature=0.9,
                top_k=None,
            )
            text_table.add_data(curr_iter, val_perplexity, out_str)
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})
    model.train()
