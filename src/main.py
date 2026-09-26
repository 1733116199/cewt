import argparse
import json
from pathlib import Path
import random
import os

import numpy as np
import torch
import wandb

import config
from data.utils import DataReader, get_dataset
import distributed
from models.utils import get_model
from optim.base import train
import optimizer.muon as muon
import optimizer.adamw as adamw
import optimizer.lion as lion
import optimizer.sso as sso

def main(args):
    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()

    if args.full_eval_at is None:
        args.full_eval_at = []

    # NOTE args.seed is offset per worker in get_adjusted_args_for_process
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))
    # torch.use_deterministic_algorithms(True)  # CUBLAS_WORKSPACE_CONFIG=:4096:8

    exp_name = get_exp_name(args, distributed_backend)
    exp_dir = Path(args.results_base_folder) / exp_name
    wandb_id = None
    if (exp_dir / "ckpts" / "latest" / "wandb_id.pt").exists():
        wandb_id = str(exp_dir / "ckpts" / "latest" / "wandb_id.pt")
        wandb_id = torch.load(wandb_id, map_location="cpu")
        print(f"Found existing wandb id: {wandb_id}")
    if distributed_backend.is_master_process() and args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=exp_name if wandb_id is None else None,
            config=vars(args),
            id=None if wandb_id is None else wandb_id,
            resume=None if wandb_id is None else "must",
        )
        wandb.define_metric("iter")
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("val/*", step_metric="iter")
        wandb.define_metric("lr", step_metric="iter")

    print(f"Starting Experiment: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Config:\n{vars(args)}\n")

    print(f"Loading dataset: '{args.dataset}'")
    datareaders = get_data_readers(args)

    model = get_model(args).to(args.device)
    # TODO: take care of initializing the model if args.use_pretrained != 'none'
    print(f"\nModel:\n{model}")

    model = distributed_backend.transform_model(model)
    # assert args.model != "base", "I am not sure how weight tying in GPT works with named_parameters, disable for now."
    matrix_weights = []
    other_weights_nodecay = []
    head_weights =[]
    wscale_weights = []
    ascale_weights = []
    for name, parameter in distributed_backend.get_raw_model(model).named_parameters():
        if name.endswith("c_attn.weight") or \
            name.endswith("c_proj.weight") or \
            name.endswith("w1.weight") or \
            name.endswith("w2.weight") or \
            name.endswith("c_fc.weight"):
                matrix_weights.append(parameter)
        elif name.endswith("ln_1.weight") or name.endswith("ln_2.weight") or name.endswith("ln_f.weight"):
            other_weights_nodecay.append(parameter)
        elif name.endswith("bias"):
            other_weights_nodecay.append(parameter)
        elif name.endswith("wte.weight"):
            other_weights_nodecay.append(parameter)
        elif name.endswith("wpe.weight"):
            other_weights_nodecay.append(parameter)
        elif name.endswith("lm_head.weight"):
            print(f"Found head weight parameter: {name} adding to head weights")
            head_weights.append(parameter)
        elif name.endswith("weight_quantizer.scale"):
            print(f"Found quantizer scale parameter: {name}, adding to wscale_weights with lr_scale {args.lsqw_lr_scale}")
            wscale_weights.append(parameter)
        elif name.endswith("activation_quantizer.scale"):
            print(f"Found activation quantizer scale parameter: {name}, adding to ascale_weights with lr_scale {args.lsqa_lr_scale}")
            ascale_weights.append(parameter)
        else:
            assert False, f"Unrecognized parameter name: {name}"
            

    params_cnt = distributed_backend.get_raw_model(model).get_num_params()
    nonemb_param_cnt = (
        params_cnt
        - distributed_backend.get_raw_model(model).lm_head.weight.numel()
        - distributed_backend.get_raw_model(model).transformer.wte.weight.numel()
    )
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of non-embedding parameters: %.2fM" % (nonemb_param_cnt / 1e6,))
    if args.wandb and distributed_backend.is_master_process():
        wandb.log(
            {
                "parameters": params_cnt,
                "non_embedding_parameters": nonemb_param_cnt,
            }
        )

    if args.hyperball_optimization:
        parameters_to_init = matrix_weights + head_weights if args.hyperball_optimization_head else matrix_weights
        for p in parameters_to_init:
            if args.hyperball_init == "lecun":
                # see https://psychedelic-sunstone-851.notion.site/Fantastic-Pretraining-Optimizers-and-Where-to-Find-Them-2-1-Hyperball-Optimization-2e924306e6f280e7a5ffee00eb40a0dd
                # section 2 Empirical Tips: "we randomly initialize each parameters with a standard deviation of 1 / sqrt(d_in)"
                torch.nn.init.normal_(p, mean=0.0, std=(p.shape[1] ** (-0.5)))
            elif args.hyperball_init == "torch":
                # see https://arxiv.org/pdf/2603.28743v1
                # "The PyTorch [default initialization from Kaiming uniform distribution is adopted."
                # see torch.nn.Linear.reset_parameters
                torch.nn.init.kaiming_uniform_(p, a=(5 ** 0.5))
            elif args.hyperball_init == "spectral_mu_p":
                # see https://arxiv.org/pdf/2601.08393
                spectral_mu_p_init_std = ((p.shape[0] / p.shape[1]) ** 0.5) / ((p.shape[0] ** 0.5) + (p.shape[1] ** 0.5))
                torch.nn.init.normal_(p, mean=0.0, std=spectral_mu_p_init_std)
            elif args.hyperball_init == "none":
                pass
            else:
                assert False, f"Unsupported hyperball_init: {args.hyperball_init}"
            if torch.distributed.is_initialized():
                # in distributed training we should make sure all workers start with the same initialization
                print(f"Synchronizing initialization across workers.")
                torch.distributed.broadcast(p.data, src=0)
                
    assert args.opt == "adamw", f"Unsupported optimizer: {args.opt}"
    opt = adamw.AdamW(
        [
            {"params": other_weights_nodecay, "lr": args.lr, "weight_decay": 0.0, "hyperball_optimization": False},
            {"params": wscale_weights, "lr": args.lsqw_lr_scale * args.lr, "weight_decay": 0.0, "hyperball_optimization": False},
            {"params": ascale_weights, "lr": args.lsqa_lr_scale * args.lr, "weight_decay": 0.0, "hyperball_optimization": False},
            {"params": head_weights, "lr": args.matrix_lr if args.hyperball_optimization_head else args.lr, "weight_decay": args.weight_decay, "hyperball_optimization": args.hyperball_optimization_head},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        hyperball_optimization=False,
        constraint="none",
    )
    
    print(f"\nOptimizer:\n{opt}")
        
    if args.matrix_opt == "adamw":
        matrix_opt = adamw.AdamW(
            matrix_weights,
            lr=args.matrix_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            hyperball_optimization=args.hyperball_optimization,
            constraint=args.hyperball_constraint,
            log_blocksize=args.matrix_param_hadamard_log_blocksize,
        )
    elif args.matrix_opt == "muon":
        matrix_opt = muon.Muon(
            matrix_weights,
            lr=args.matrix_lr,
            weight_decay=args.weight_decay,
            momentum=args.muon_momentum,
            adjust_lr_fn="match_rms_adamw",
            hyperball_optimization=args.hyperball_optimization,
            constraint=args.hyperball_constraint,
            log_blocksize=args.matrix_param_hadamard_log_blocksize,
        )
    elif args.matrix_opt == "lion":
        matrix_opt = lion.Lion(
            matrix_weights,
            lr=args.matrix_lr,
            betas=(args.lion_beta1, args.lion_beta2),
            weight_decay=args.weight_decay,
            hyperball_optimization=args.hyperball_optimization,
            constraint=args.hyperball_constraint,
            log_blocksize=args.matrix_param_hadamard_log_blocksize,
        )
    elif args.matrix_opt == "sso":
        matrix_opt = sso.SSO(
            matrix_weights,
            lr=args.matrix_lr,
            constraint=args.hyperball_constraint,
            weight_decay=0.0,
            linear_lr_scale=args.sso_linear_lr_scale,
            radius_scaler=args.sso_radius_scaler,
            post_retract=args.sso_post_retract,
        )
    else:
        assert False, f"Unsupported matrix optimizer: {args.matrix_opt}"
    print(f"\nMatrix Optimizer:\n{matrix_opt}")

    assert args.warmup_steps < args.iterations, "Warmup steps must be < iterations."
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=[args.lr, args.lsqw_lr_scale * args.lr, args.lsqa_lr_scale * args.lr, args.matrix_lr if args.hyperball_optimization_head else args.lr],
        total_steps=args.iterations,
        pct_start=args.warmup_steps / args.iterations,
        anneal_strategy=args.scheduler,
        cycle_momentum=False,
        div_factor=1e2,
        final_div_factor=0.1,
    )
    matrix_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=matrix_opt,
        max_lr=args.matrix_lr,
        total_steps=args.iterations,
        pct_start=args.warmup_steps / args.iterations,
        anneal_strategy=args.matrix_scheduler,
        cycle_momentum=False,
        div_factor=1e2,
        final_div_factor=0.1,
    )

    if (exp_dir / "ckpts" / "latest" / "main.pt").exists():
        if not args.auto_resume:
            raise ValueError(
                f"The experiment dir {exp_dir} already exists. "
                + "To resume training, set auto_resume=True. "
                + "Otherwise, specify a different experiment name. "
            )
        else:
            # Auto resume overwrites resume_from
            args.resume_from = str(exp_dir / "ckpts" / "latest")

    elif distributed_backend.is_master_process():
        exp_dir.mkdir(parents=True, exist_ok=True)

    stats = train(
        model=model,
        opt=opt,
        matrix_opt=matrix_opt,
        datareaders=datareaders,
        scheduler=scheduler,
        matrix_scheduler=matrix_scheduler,
        exp_dir=exp_dir,
        distributed_backend=distributed_backend,
        cfg=args,
    )

    stats["args"] = vars(args)
    if distributed_backend.is_master_process():
        with open(exp_dir / "summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )


def get_exp_name(args, distributed_backend):
    """Returns the name of the experiment, used for saving models and wandb."""
    if args.experiment_name is not None:
        return args.experiment_name

    rank = distributed_backend.rank

    exp_name = (
        f"{args.dataset}_{args.model}_nlayers{args.n_layer}"
        f"_nhead{args.n_head}_lr{args.lr}"
        f"_sched_{args.scheduler}_warmup{args.warmup_steps}"
        f"_decay_{args.decay_type}_{args.wsd_fract_decay}"
        f"_iter{args.iterations}"
        f"_bs{args.batch_size}x{args.acc_steps}_ws{args.world_size}"
    )
    # for mup
    if args.model == "mup_noam":
        exp_name = (
            f"{args.dataset}_{args.model}"
            f"_opt{args.opt}"
            f"_nlayers{args.n_layer}"
            # f"_nhead{args.n_head}"
            f"_lr{args.lr}"
            f"_sched_{args.scheduler}"
            f"_decay_{args.decay_type}"
            # f"_warmup{args.warmup_steps}"
            f"_iter{args.iterations}"
            f"_init{args.init_std}_sce{args.scale_emb}"
            f"_scd{args.scale_depth}"
            # f"_bs{args.batch_size}x{args.acc_steps}_ws{args.world_size}"
        )
    if args.wandb_run_prefix != "none":
        exp_name = args.wandb_run_prefix + "_" + exp_name
    exp_name += f"_seed{args.seed - rank}"
    exp_name += f"_data_seed{args.data_seed}"

    if args.weight_average:
        exp_name += f"_WA"
    if args.opt == "SFAdamW":
        exp_name += f"_beta1_{args.beta1}"
        exp_name += f"_beta2_{args.beta2}"
    return exp_name


def get_data_readers(args, verbose=True):
    data_srcs = get_dataset(args)
    train_reader = DataReader(
        data_src=data_srcs["train"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=args.data_in_ram,
    )
    val_reader = DataReader(
        data_src=data_srcs["val"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,  # NOTE Identical Per Rank
        keep_in_ram=args.data_in_ram,
    )

    if verbose:
        print(f"Num training tokens: {train_reader.num_tokens}")
        print(f"Num validation tokens: {val_reader.num_tokens}")

    return {
        "train": train_reader,
        "val": val_reader,
    }


if __name__ == "__main__":
    args = get_args()
    main(args)
