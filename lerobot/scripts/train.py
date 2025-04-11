#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from threading import Lock
from datetime import datetime

import hydra
import numpy as np
import torch
from deepdiff import DeepDiff
from omegaconf import DictConfig, ListConfig, OmegaConf
from termcolor import colored
from torch import nn
from torch.cuda.amp import GradScaler

from lerobot.common.datasets.factory import make_dataset, resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset
from lerobot.common.datasets.online_buffer import OnlineBuffer, compute_sampler_weights
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import Logger, log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import PolicyWithUpdate
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)
from lerobot.scripts.eval import eval_policy
from torch.serialization import add_safe_globals
from omegaconf.base import ContainerMetadata
import matplotlib.pyplot as plt
import pandas as pd
import json

add_safe_globals([
    ContainerMetadata,  # omegaconf 基础元数据类
    ListConfig,         # omegaconf 列表配置类
    DictConfig,        # omegaconf 字典配置类
])

def make_optimizer_and_scheduler(cfg, policy):
    if cfg.policy.name == "act":
        optimizer_params_dicts = [
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in policy.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": cfg.training.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_params_dicts, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
        lr_scheduler = None
    elif cfg.policy.name == "diffusion":
        optimizer = torch.optim.Adam(
            policy.diffusion.parameters(),
            cfg.training.lr,
            cfg.training.adam_betas,
            cfg.training.adam_eps,
            cfg.training.adam_weight_decay,
        )
        from diffusers.optimization import get_scheduler

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.offline_steps,
        )
    elif policy.name == "tdmpc":
        optimizer = torch.optim.Adam(policy.parameters(), cfg.training.lr)
        lr_scheduler = None
    elif cfg.policy.name == "vqbet":
        from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTOptimizer, VQBeTScheduler

        optimizer = VQBeTOptimizer(policy, cfg)
        lr_scheduler = VQBeTScheduler(optimizer, cfg)
        
    elif cfg.policy.name == "scaledp":
        params = policy.get_optim_params() 
        
        # 为不同层使用不同权重衰减
        param_groups = [
            {'params': [p for n, p in policy.named_parameters() if 'attn' in n], 
             'weight_decay': 1e-4},  # 注意力层使用较大权重衰减
            {'params': [p for n, p in policy.named_parameters() if 'mlp' in n], 
             'weight_decay': 2e-4},  # MLP层使用更大权重衰减
            {'params': [p for n, p in policy.named_parameters() 
                       if not any(x in n for x in ['attn', 'mlp'])], 
             'weight_decay': 5e-5}   # 其他层使用较小权重衰减
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.training.lr)
        # optimizer = torch.optim.Adam(
        #     param_groups,
        #     cfg.training.lr,
        #     cfg.training.adam_betas,
        #     cfg.training.adam_eps,
        #     cfg.training.adam_weight_decay,
        # )

        from diffusers.optimization import get_scheduler

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.offline_steps,
        )
    else:
        raise NotImplementedError()

    return optimizer, lr_scheduler


def update_policy(
    policy,
    batch,
    optimizer,
    grad_clip_norm,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
):
    """Returns a dictionary of items for logging."""
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        output_dict = policy.forward(batch)
        loss = output_dict["loss"]
    
    # 清空之前的梯度
    optimizer.zero_grad()
    
    # 反向传播
    grad_scaler.scale(loss).backward()

    # 在梯度裁剪前记录梯度统计
    grad_stats = {}
    if hasattr(policy, "name") and policy.name == "scale_dp":
        grad_stats = log_gradient_stats(policy, grad_clip_norm)

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    # grad_scaler.scale(loss).backward()

    # # 在梯度裁剪前记录梯度统计
    # grad_stats = {}
    # if hasattr(policy, "name") and policy.name == "scale_dp":
    #     grad_stats = log_gradient_stats(policy, grad_clip_norm)


    # # Unscale the graident of the optimzer's assigned params in-place **prior to gradient clipping**.
    # grad_scaler.unscale_(optimizer)

    # grad_norm = torch.nn.utils.clip_grad_norm_(
    #     policy.parameters(),
    #     grad_clip_norm,
    #     error_if_nonfinite=False,
    # )

    # # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    # with lock if lock is not None else nullcontext():
    #     grad_scaler.step(optimizer)
    # # Updates the scale for next iteration.
    # grad_scaler.update()

    # optimizer.zero_grad()



    if lr_scheduler is not None:
        lr_scheduler.step()

    if isinstance(policy, PolicyWithUpdate):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    info = {
        "loss": loss.item(),
        "grad_norm": float(grad_norm),
        "lr": optimizer.param_groups[0]["lr"],
        "update_s": time.perf_counter() - start_time,
        **{k: v for k, v in output_dict.items() if k != "loss"},
    }
    
    # 将梯度统计添加到info中
    info.update(grad_stats)
    
    # 将其他信息添加到info中
    info.update({k: v for k, v in output_dict.items() if k not in info})

    return info


def log_gradient_stats(model, grad_clip_norm, log_details=False):
    """记录模型各层梯度统计信息"""
    stats = {}
    
    # 1. 基本梯度统计
    layer_stats = {}
    max_norm = 0
    min_norm = float('inf')
    problem_layers = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            norm = grad.norm().item()
            mean = grad.mean().item()
            max_val = grad.abs().max().item()
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()
            
            # 记录每层的统计数据
            layer_stats[name] = {
                "norm": norm,
                "mean": mean,
                "max": max_val,
                "has_nan": has_nan,
                "has_inf": has_inf,
            }
            
            # 更新全局统计
            max_norm = max(max_norm, norm)
            if norm > 0:  # 忽略零梯度
                min_norm = min(min_norm, norm)
            
            # 检查问题
            if has_nan or has_inf or norm > grad_clip_norm * 0.9:
                problem_layers.append(name)
                
            # 检查梯度与权重比例
            if hasattr(param, 'data') and param.data.abs().max() != 0:
                param_max = param.data.abs().max().item()
                ratio = max_val / param_max
                layer_stats[name]["grad_weight_ratio"] = ratio
                
                # 如果比例过大或过小，记录问题
                if ratio > 100 or ratio < 1e-6:
                    if name not in problem_layers:
                        problem_layers.append(name)
    
    # 2. 汇总统计
    stats["max_grad_norm"] = max_norm
    stats["min_grad_norm"] = min_norm if min_norm != float('inf') else 0
    stats["grad_norm_ratio"] = max_norm / min_norm if min_norm > 0 else 0
    stats["problem_layers_count"] = len(problem_layers)
    stats["has_nan_or_inf"] = len([name for name, stat in layer_stats.items() 
                               if stat["has_nan"] or stat["has_inf"]]) > 0
    
    # 3. 记录详细信息到日志
    if log_details and problem_layers:
        logging.warning(f"发现梯度问题的层: {problem_layers}")
        for name in problem_layers:
            logging.warning(f"  {name}: norm={layer_stats[name]['norm']:.4f}, "
                         f"max={layer_stats[name]['max']:.4f}, "
                         f"has_nan={layer_stats[name]['has_nan']}, "
                         f"has_inf={layer_stats[name]['has_inf']}")
            if "grad_weight_ratio" in layer_stats[name]:
                logging.warning(f"  梯度/权重比例: {layer_stats[name]['grad_weight_ratio']:.4f}")
    
    return stats


def log_train_info(logger: Logger, info, step, cfg, dataset, is_online):
    loss = info["loss"]
    grad_norm = info["grad_norm"]
    lr = info["lr"]
    update_s = info["update_s"]
    dataloading_s = info["dataloading_s"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.training.batch_size
    avg_samples_per_ep = dataset.num_samples / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_samples
    log_items = [
        f"step:{format_big_number(step)}",
        # number of samples seen during training
        f"smpl:{format_big_number(num_samples)}",
        # number of episodes seen during training
        f"ep:{format_big_number(num_episodes)}",
        # number of time all unique samples are seen
        f"epch:{num_epochs:.2f}",
        f"loss:{loss:.3f}",
        f"grdn:{grad_norm:.3f}",
        f"lr:{lr:0.1e}",
        # in seconds
        f"updt_s:{update_s:.3f}",
        f"data_s:{dataloading_s:.3f}",  # if not ~0, you are bottlenecked by cpu or io
    ]
    logging.info(" ".join(log_items))

    info["step"] = step
    info["num_samples"] = num_samples
    info["num_episodes"] = num_episodes
    info["num_epochs"] = num_epochs
    info["is_online"] = is_online

    logger.log_dict(info, step, mode="train")


def log_eval_info(logger, info, step, cfg, dataset, is_online):
    eval_s = info["eval_s"]
    avg_sum_reward = info["avg_sum_reward"]
    pc_success = info["pc_success"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.training.batch_size
    avg_samples_per_ep = dataset.num_samples / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_samples
    log_items = [
        f"step:{format_big_number(step)}",
        # number of samples seen during training
        f"smpl:{format_big_number(num_samples)}",
        # number of episodes seen during training
        f"ep:{format_big_number(num_episodes)}",
        # number of time all unique samples are seen
        f"epch:{num_epochs:.2f}",
        f"∑rwrd:{avg_sum_reward:.3f}",
        f"success:{pc_success:.1f}%",
        f"eval_s:{eval_s:.3f}",
    ]
    logging.info(" ".join(log_items))

    info["step"] = step
    info["num_samples"] = num_samples
    info["num_episodes"] = num_episodes
    info["num_epochs"] = num_epochs
    info["is_online"] = is_online

    logger.log_dict(info, step, mode="eval")


def train(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    init_logging()
    logging.info(pformat(OmegaConf.to_container(cfg)))

    if cfg.training.online_steps > 0 and isinstance(cfg.dataset_repo_id, ListConfig):
        raise NotImplementedError("Online training with LeRobotMultiDataset is not implemented.")

    # If we are resuming a run, we need to check that a checkpoint exists in the log directory, and we need
    # to check for any differences between the provided config and the checkpoint's config.
    if cfg.resume:
        if not Logger.get_last_checkpoint_dir(out_dir).exists():
            raise RuntimeError(
                "You have set resume=True, but there is no model checkpoint in "
                f"{Logger.get_last_checkpoint_dir(out_dir)}"
            )
        checkpoint_cfg_path = str(Logger.get_last_pretrained_model_dir(out_dir) / "config.yaml")
        logging.info(
            colored(
                "You have set resume=True, indicating that you wish to resume a run",
                color="yellow",
                attrs=["bold"],
            )
        )
        # Get the configuration file from the last checkpoint.
        checkpoint_cfg = init_hydra_config(checkpoint_cfg_path)
        # Check for differences between the checkpoint configuration and provided configuration.
        # Hack to resolve the delta_timestamps ahead of time in order to properly diff.
        resolve_delta_timestamps(cfg)
        diff = DeepDiff(OmegaConf.to_container(checkpoint_cfg), OmegaConf.to_container(cfg))
        # Ignore the `resume` and parameters.
        if "values_changed" in diff and "root['resume']" in diff["values_changed"]:
            del diff["values_changed"]["root['resume']"]
        # Log a warning about differences between the checkpoint configuration and the provided
        # configuration.
        if len(diff) > 0:
            logging.warning(
                "At least one difference was detected between the checkpoint configuration and "
                f"the provided configuration: \n{pformat(diff)}\nNote that the checkpoint configuration "
                "takes precedence.",
            )
        # Use the checkpoint config instead of the provided config (but keep `resume` parameter).
        cfg = checkpoint_cfg
        cfg.resume = True
        logging.info(pformat(OmegaConf.to_container(cfg)))
    elif Logger.get_last_checkpoint_dir(out_dir).exists():
        raise RuntimeError(
            f"The configured output directory {Logger.get_last_checkpoint_dir(out_dir)} already exists. If "
            "you meant to resume training, please use `resume=true` in your command or yaml configuration."
        )

    if cfg.eval.batch_size > cfg.eval.n_episodes:
        raise ValueError(
            "The eval batch size is greater than the number of eval episodes "
            f"({cfg.eval.batch_size} > {cfg.eval.n_episodes}). As a result, {cfg.eval.batch_size} "
            f"eval environments will be instantiated, but only {cfg.eval.n_episodes} will be used. "
            "This might significantly slow down evaluation. To fix this, you should update your command "
            f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={cfg.eval.batch_size}`), "
            f"or lower the batch size (e.g. `eval.batch_size={cfg.eval.n_episodes}`)."
        )

    # log metrics to terminal and wandb
    logger = Logger(cfg, out_dir, wandb_job_name=job_name)

    set_global_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_dataset")
    offline_dataset = make_dataset(cfg)
    if isinstance(offline_dataset, MultiLeRobotDataset):
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(offline_dataset.repo_id_to_index , indent=2)}"
        )

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.training.eval_freq > 0:
        logging.info("make_env")
        eval_env = make_env(cfg)

    logging.info("make_policy")
    policy = make_policy(
        hydra_cfg=cfg,
        dataset_stats=offline_dataset.stats if not cfg.resume else None,
        pretrained_policy_name_or_path=str(logger.last_pretrained_model_dir) if cfg.resume else None,
    )
    assert isinstance(policy, nn.Module)
    # Create optimizer and scheduler
    # Temporary hack to move optimizer out of policy
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(enabled=cfg.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step = logger.load_last_training_state(optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())
    # cfg.training.offline_steps = 300000
    log_output_dir(out_dir)
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.training.offline_steps=} ({format_big_number(cfg.training.offline_steps)})")
    logging.info(f"{cfg.training.online_steps=}")
    logging.info(f"{offline_dataset.num_samples=} ({format_big_number(offline_dataset.num_samples)})")
    logging.info(f"{offline_dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Note: this helper will be used in offline and online training loops.
    def evaluate_and_checkpoint_if_needed(step, is_online):
        _num_digits = max(6, len(str(cfg.training.offline_steps + cfg.training.online_steps)))
        step_identifier = f"{step:0{_num_digits}d}"

        if cfg.training.eval_freq > 0 and step % cfg.training.eval_freq == 0:
            logging.info(f"Eval policy at step {step}")
            with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
                assert eval_env is not None
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=Path(out_dir) / "eval" / f"videos_step_{step_identifier}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )
            log_eval_info(logger, eval_info["aggregated"], step, cfg, offline_dataset, is_online=is_online)
            if cfg.wandb.enable:
                logger.log_video(eval_info["video_paths"][0], step, mode="eval")
            logging.info("Resume training")

        if cfg.training.save_checkpoint and (
            step % cfg.training.save_freq == 0
            or step == cfg.training.offline_steps + cfg.training.online_steps
        ):
            logging.info(f"Checkpoint policy after step {step}")
            # Note: Save with step as the identifier, and format it to have at least 6 digits but more if
            # needed (choose 6 as a minimum for consistency without being overkill).
            logger.save_checkpoint(
                step,
                policy,
                optimizer,
                lr_scheduler,
                identifier=step_identifier,
                max_checkpoints=cfg.training.max_checkpoints,
            )
            logging.info("Resume training")

    # create dataloader for offline training
    if cfg.training.get("drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            offline_dataset.episode_data_index,
            drop_n_last_frames=cfg.training.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    dataloader = torch.utils.data.DataLoader(
        offline_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()
    offline_step = 0
    # 添加梯度跟踪
    grad_tracking = {
        "step": [],
        "loss": [],
        "grad_norm": [],
        "max_grad_norm": [],
        "min_grad_norm": [],
        "grad_norm_ratio": [],
        "problem_layers_count": [],
    }
    for _ in range(step, cfg.training.offline_steps):
        if offline_step == 0:
            logging.info("Start offline training on a fixed dataset")

        start_time = time.perf_counter()
        batch = next(dl_iter)
        dataloading_s = time.perf_counter() - start_time

        for key in batch:
            batch[key] = batch[key].to(device, non_blocking=True)

        train_info = update_policy(
            policy,
            batch,
            optimizer,
            cfg.training.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.use_amp,
        )

        train_info["dataloading_s"] = dataloading_s

        # 收集梯度数据
        for k in ["max_grad_norm", "min_grad_norm", "grad_norm_ratio", "problem_layers_count"]:
            if k in train_info:
                grad_tracking[k].append(train_info[k])
            else:
                grad_tracking[k].append(0)  # 默认值
        
        grad_tracking["step"].append(step)
        grad_tracking["loss"].append(train_info["loss"])
        grad_tracking["grad_norm"].append(train_info["grad_norm"])

        # 每10000步进行一次梯度分析
        if step % 200 == 0 and step > 0:
            analyze_and_visualize_gradients(grad_tracking, out_dir, step)
            
            # 如果发现严重问题，可以提供警告
            recent_problems = grad_tracking["problem_layers_count"][-100:]
            if sum(recent_problems) > 50:  # 如果最近100步中有超过50步出现问题
                logging.warning(f"训练不稳定! 最近100步中有{sum(recent_problems)}步出现梯度问题")
                logging.warning("建议降低学习率或增加梯度裁剪强度")

        if step % cfg.training.log_freq == 0:
            log_train_info(logger, train_info, step, cfg, offline_dataset, is_online=False)

        # 添加梯度分布分析，建议每1000步执行一次
        if step % 200 == 0 and step > 0:
            analyze_gradient_distribution(policy, out_dir, step)

        # Note: evaluate_and_checkpoint_if_needed happens **after** the `step`th training update has completed,
        # so we pass in step + 1.
        evaluate_and_checkpoint_if_needed(step + 1, is_online=False)

        step += 1
        offline_step += 1

        if step % 1000 == 0:
            param_norms = {name: param.norm().item() 
                          for name, param in policy.named_parameters()}
            with open(f"{out_dir}/param_norms_step_{step}.json", "w") as f:
                json.dump(param_norms, f, indent=2)

    if cfg.training.online_steps == 0:
        if eval_env:
            eval_env.close()
        logging.info("End of training")
        return

    # Online training.

    # Create an env dedicated to online episodes collection from policy rollout.
    online_env = make_env(cfg, n_envs=cfg.training.online_rollout_batch_size)
    resolve_delta_timestamps(cfg)
    online_buffer_path = logger.log_dir / "online_buffer"
    if cfg.resume and not online_buffer_path.exists():
        # If we are resuming a run, we default to the data shapes and buffer capacity from the saved online
        # buffer.
        logging.warning(
            "When online training is resumed, we load the latest online buffer from the prior run, "
            "and this might not coincide with the state of the buffer as it was at the moment the checkpoint "
            "was made. This is because the online buffer is updated on disk during training, independently "
            "of our explicit checkpointing mechanisms."
        )
    online_dataset = OnlineBuffer(
        online_buffer_path,
        data_spec={
            **{k: {"shape": v, "dtype": np.dtype("float32")} for k, v in policy.config.input_shapes.items()},
            **{k: {"shape": v, "dtype": np.dtype("float32")} for k, v in policy.config.output_shapes.items()},
            "next.reward": {"shape": (), "dtype": np.dtype("float32")},
            "next.done": {"shape": (), "dtype": np.dtype("?")},
            "next.success": {"shape": (), "dtype": np.dtype("?")},
        },
        buffer_capacity=cfg.training.online_buffer_capacity,
        fps=online_env.unwrapped.metadata["render_fps"],
        delta_timestamps=cfg.training.delta_timestamps,
    )

    # If we are doing online rollouts asynchronously, deepcopy the policy to use for online rollouts (this
    # makes it possible to do online rollouts in parallel with training updates).
    online_rollout_policy = deepcopy(policy) if cfg.training.do_online_rollout_async else policy

    # Create dataloader for online training.
    concat_dataset = torch.utils.data.ConcatDataset([offline_dataset, online_dataset])
    sampler_weights = compute_sampler_weights(
        offline_dataset,
        offline_drop_n_last_frames=cfg.training.get("drop_n_last_frames", 0),
        online_dataset=online_dataset,
        # +1 because online rollouts return an extra frame for the "final observation". Note: we don't have
        # this final observation in the offline datasets, but we might add them in future.
        online_drop_n_last_frames=cfg.training.get("drop_n_last_frames", 0) + 1,
        online_sampling_ratio=cfg.training.online_sampling_ratio,
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        sampler_weights,
        num_samples=len(concat_dataset),
        replacement=True,
    )
    dataloader = torch.utils.data.DataLoader(
        concat_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)

    # Lock and thread pool executor for asynchronous online rollouts. When asynchronous mode is disabled,
    # these are still used but effectively do nothing.
    lock = Lock()
    # Note: 1 worker because we only ever want to run one set of online rollouts at a time. Batch
    # parallelization of rollouts is handled within the job.
    executor = ThreadPoolExecutor(max_workers=1)

    online_step = 0
    online_rollout_s = 0  # time take to do online rollout
    update_online_buffer_s = 0  # time taken to update the online buffer with the online rollout data
    # Time taken waiting for the online buffer to finish being updated. This is relevant when using the async
    # online rollout option.
    await_update_online_buffer_s = 0
    rollout_start_seed = cfg.training.online_env_seed

    while True:
        if online_step == cfg.training.online_steps:
            break

        if online_step == 0:
            logging.info("Start online training by interacting with environment")

        def sample_trajectory_and_update_buffer():
            nonlocal rollout_start_seed
            with lock:
                online_rollout_policy.load_state_dict(policy.state_dict())
            online_rollout_policy.eval()
            start_rollout_time = time.perf_counter()
            with torch.no_grad():
                eval_info = eval_policy(
                    online_env,
                    online_rollout_policy,
                    n_episodes=cfg.training.online_rollout_n_episodes,
                    max_episodes_rendered=min(10, cfg.training.online_rollout_n_episodes),
                    videos_dir=logger.log_dir / "online_rollout_videos",
                    return_episode_data=True,
                    start_seed=(
                        rollout_start_seed := (rollout_start_seed + cfg.training.batch_size) % 1000000
                    ),
                )
            online_rollout_s = time.perf_counter() - start_rollout_time

            with lock:
                start_update_buffer_time = time.perf_counter()
                online_dataset.add_data(eval_info["episodes"])

                # Update the concatenated dataset length used during sampling.
                concat_dataset.cumulative_sizes = concat_dataset.cumsum(concat_dataset.datasets)

                # Update the sampling weights.
                sampler.weights = compute_sampler_weights(
                    offline_dataset,
                    offline_drop_n_last_frames=cfg.training.get("drop_n_last_frames", 0),
                    online_dataset=online_dataset,
                    # +1 because online rollouts return an extra frame for the "final observation". Note: we don't have
                    # this final observation in the offline datasets, but we might add them in future.
                    online_drop_n_last_frames=cfg.training.get("drop_n_last_frames", 0) + 1,
                    online_sampling_ratio=cfg.training.online_sampling_ratio,
                )
                sampler.num_samples = len(concat_dataset)

                update_online_buffer_s = time.perf_counter() - start_update_buffer_time

            return online_rollout_s, update_online_buffer_s

        future = executor.submit(sample_trajectory_and_update_buffer)
        # If we aren't doing async rollouts, or if we haven't yet gotten enough examples in our buffer, wait
        # here until the rollout and buffer update is done, before proceeding to the policy update steps.
        if (
            not cfg.training.do_online_rollout_async
            or len(online_dataset) <= cfg.training.online_buffer_seed_size
        ):
            online_rollout_s, update_online_buffer_s = future.result()

        if len(online_dataset) <= cfg.training.online_buffer_seed_size:
            logging.info(
                f"Seeding online buffer: {len(online_dataset)}/{cfg.training.online_buffer_seed_size}"
            )
            continue

        policy.train()
        for _ in range(cfg.training.online_steps_between_rollouts):
            with lock:
                start_time = time.perf_counter()
                batch = next(dl_iter)
                dataloading_s = time.perf_counter() - start_time

            for key in batch:
                batch[key] = batch[key].to(cfg.device, non_blocking=True)

            train_info = update_policy(
                policy,
                batch,
                optimizer,
                cfg.training.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.use_amp,
                lock=lock,
            )

            train_info["dataloading_s"] = dataloading_s
            train_info["online_rollout_s"] = online_rollout_s
            train_info["update_online_buffer_s"] = update_online_buffer_s
            train_info["await_update_online_buffer_s"] = await_update_online_buffer_s
            with lock:
                train_info["online_buffer_size"] = len(online_dataset)

            if step % cfg.training.log_freq == 0:
                log_train_info(logger, train_info, step, cfg, online_dataset, is_online=True)

            # Note: evaluate_and_checkpoint_if_needed happens **after** the `step`th training update has completed,
            # so we pass in step + 1.
            evaluate_and_checkpoint_if_needed(step + 1, is_online=True)

            step += 1
            online_step += 1

        # If we're doing async rollouts, we should now wait until we've completed them before proceeding
        # to do the next batch of rollouts.
        if future.running():
            start = time.perf_counter()
            online_rollout_s, update_online_buffer_s = future.result()
            await_update_online_buffer_s = time.perf_counter() - start

        if online_step >= cfg.training.online_steps:
            break

    if eval_env:
        eval_env.close()
    logging.info("End of training")


@hydra.main(version_base="1.2", config_name="default", config_path="../configs")
def train_cli(cfg: dict):
    train(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )


def train_notebook(out_dir=None, job_name=None, config_name="default", config_path="../configs"):
    from hydra import compose, initialize

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)
    train(cfg, out_dir=out_dir, job_name=job_name)


# 添加梯度分析和可视化函数
def analyze_and_visualize_gradients(grad_tracking, out_dir, step):
    """分析梯度趋势并生成可视化图表
        结果保存到路径 gradient_analysis 下
        保存gradient_analysis_step_{step}.png 图片
        保存gradient_data_step_{step}.csv
        保存analysis_summary_step_{step}.txt 分析结果


    """
    # 创建保存目录
    vis_dir = Path(out_dir) / "gradient_analysis"
    vis_dir.mkdir(exist_ok=True, parents=True)
    
    # 转换为DataFrame进行分析
    df = pd.DataFrame(grad_tracking)
    
    # 绘制多个指标
    plt.figure(figsize=(16, 12))
    
    # 1. 绘制损失和梯度范数
    plt.subplot(2, 2, 1)
    plt.plot(df["step"], df["loss"], label="Loss")
    plt.title("Loss vs Training Steps")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(df["step"], df["grad_norm"], label="Gradient Norm")
    plt.title("Gradient Norm vs Training Steps")
    plt.xlabel("Steps")
    plt.ylabel("Gradient Norm")
    plt.grid(True)
    
    # 2. 绘制最大和最小梯度范数
    plt.subplot(2, 2, 3)
    plt.plot(df["step"], df["max_grad_norm"], label="Max Grad Norm")
    plt.plot(df["step"], df["min_grad_norm"], label="Min Grad Norm")
    plt.title("Max and Min Gradient Norms")
    plt.xlabel("Steps")
    plt.ylabel("Gradient Norm")
    plt.legend()
    plt.grid(True)
    
    # 3. 绘制梯度范数比率和问题层数量
    plt.subplot(2, 2, 4)
    plt.plot(df["step"], df["grad_norm_ratio"], label="Grad Norm Ratio")
    plt.plot(df["step"], df["problem_layers_count"], label="Problem Layers Count")
    plt.title("Gradient Issues Indicators")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(vis_dir / f"gradient_analysis_step_{step}.png")
    plt.close()
    
    # 保存数据
    df.to_csv(vis_dir / f"gradient_data_step_{step}.csv", index=False)
    
    # 基本分析和统计
    analysis = {
        "recent_loss_trend": "上升" if df["loss"].iloc[-100:].mean() > df["loss"].iloc[-200:-100].mean() else "下降",
        "recent_grad_norm_trend": "上升" if df["grad_norm"].iloc[-100:].mean() > df["grad_norm"].iloc[-200:-100].mean() else "下降",
        "problem_frequency": df["problem_layers_count"].mean(),
        "max_grad_norm_stats": df["max_grad_norm"].describe().to_dict(),
        "grad_norm_ratio_stats": df["grad_norm_ratio"].describe().to_dict()
    }
    
    # 保存分析结果
    with open(vis_dir / f"analysis_summary_step_{step}.txt", "w") as f:
        f.write(f"梯度分析报告 (步骤: {step})\n")
        f.write("-" * 50 + "\n")
        f.write(f"最近的损失趋势: {analysis['recent_loss_trend']}\n")
        f.write(f"最近的梯度范数趋势: {analysis['recent_grad_norm_trend']}\n")
        f.write(f"问题层出现频率: {analysis['problem_frequency']:.2f}\n")
        f.write("-" * 50 + "\n")
        f.write("最大梯度范数统计:\n")
        for k, v in analysis["max_grad_norm_stats"].items():
            f.write(f"  {k}: {v}\n")
        f.write("-" * 50 + "\n")
        f.write("梯度范数比率统计:\n")
        for k, v in analysis["grad_norm_ratio_stats"].items():
            f.write(f"  {k}: {v}\n")
    
    logging.info(f"梯度分析完成，结果保存到 {vis_dir}")


def analyze_gradient_distribution(model, out_dir, step):
    """
    Analyze gradient distribution across model layers
    """
    # 确保out_dir是字符串或路径对象
    if isinstance(out_dir, int):
        logging.error(f"Invalid out_dir type: {type(out_dir)}, must be str or Path")
        return
    
    # 创建保存目录
    vis_dir = Path(out_dir) / "gradient_analysis"
    vis_dir.mkdir(exist_ok=True, parents=True)
    
    # 打印调试信息
    logging.info(f"Starting gradient analysis for step {step}")
    
    # 收集梯度数据 - 直接从模型参数收集
    layer_names = []
    grad_norms = []
    param_norms = []
    grad_param_ratios = []
    
    # 直接从模型参数收集梯度数据
    named_parameters = list(model.named_parameters())
    logging.info(f"Model has {len(named_parameters)} named parameters")
    
    collected_count = 0
    for name, param in named_parameters:
        # 只处理需要梯度且梯度不为None的参数
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item() 
            
            # 避免除零
            if param_norm < 1e-10:
                param_norm = 1e-10
                
            ratio = grad_norm / param_norm
            
            layer_names.append(name)
            grad_norms.append(grad_norm)
            param_norms.append(param_norm)
            grad_param_ratios.append(ratio)
            collected_count += 1
    
    logging.info(f"Collected gradient data for {collected_count} parameters")
    
    # 如果没有收集到任何数据，添加虚拟数据
    if not layer_names:
        logging.warning("No gradient data collected. Using dummy data for visualization.")
        layer_names = ["dummy"]
        grad_norms = [0.001]
        param_norms = [1.0]
        grad_param_ratios = [0.001]
    
    # 保存收集到的原始数据(用于调试)
    with open(vis_dir / f"grad_raw_data_{step}.txt", "w") as f:
        f.write(f"=== Raw Gradient Data (Step {step}) ===\n\n")
        f.write(f"Total parameters with gradient: {len(layer_names)}\n\n")
        for i, (name, grad, param, ratio) in enumerate(zip(layer_names, grad_norms, param_norms, grad_param_ratios)):
            if i < 50 or i > len(layer_names) - 50:  # 只记录前50和后50个，避免文件过大
                f.write(f"{name}: grad_norm={grad:.6f}, param_norm={param:.6f}, ratio={ratio:.6f}\n")
            elif i == 50:
                f.write("...(omitted)...\n")
    
    # 创建图表
    plt.figure(figsize=(20, 16))
    
    # 顶部子图 - 左侧：梯度范数分布
    plt.subplot(2, 2, 1)
    plt.hist(grad_norms, bins=50, alpha=0.7)
    plt.title(f"Gradient Norm Distribution (Step {step})", fontsize=14)
    plt.xlabel("Gradient Norm", fontsize=12)
    plt.ylabel("Layer Count", fontsize=12)
    try:
        plt.yscale('log')
    except:
        plt.yscale('linear')
    plt.grid(True)
    
    # 顶部子图 - 右侧：梯度/参数比率分布
    plt.subplot(2, 2, 2)
    plt.hist(grad_param_ratios, bins=50, alpha=0.7)
    plt.title("Gradient/Parameter Ratio Distribution", fontsize=14)
    plt.xlabel("Gradient/Parameter Ratio", fontsize=12)
    plt.ylabel("Layer Count", fontsize=12)
    try:
        plt.yscale('log')
    except:
        plt.yscale('linear')
    plt.grid(True)
    
    # 底部子图：梯度分布对比
    plt.subplot(2, 1, 2)
    
    # 如果只有虚拟数据，显示提示信息
    if len(layer_names) == 1 and layer_names[0] == "dummy":
        plt.text(0.5, 0.5, "Insufficient gradient data", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=20)
    else:
        # 准备排序数据
        sorted_indices = np.argsort(grad_norms)
        sorted_norms = np.array(grad_norms)[sorted_indices]
        sorted_names = np.array(layer_names)[sorted_indices]
        
        # 减少显示的层数以提高可读性
        max_display = 15  # 每组最多显示15个，减少显示数量以留出更多空间给标签
        num_display = min(max_display, len(sorted_names)//2)
        if num_display == 0:
            num_display = min(1, len(sorted_names))
        
        # 更详细的层名称处理函数 - 保留更多信息
        def format_layer_name(name, max_length=45):
            """格式化层名称，保留更多关键信息同时确保长度合适"""
            # 移除前缀路径
            if "/" in name:
                name = name.split("/")[-1]
            
            # 识别关键组件
            components = []
            if "." in name:
                parts = name.split(".")
                
                # 优先保留的关键字
                key_terms = [
                    "transformer_block", "encoder", "decoder", "backbone", 
                    "attn", "mlp", "embedder", "head", "layer", "blocks",
                    "norm", "weight", "bias", "adaLN", "vision", "final"
                ]
                
                # 找出关键组件
                for i, part in enumerate(parts):
                    # 检查当前部分是否包含关键字
                    if any(term in part for term in key_terms):
                        # 获取当前部分和它的索引(如果有的话)
                        current_part = part
                        
                        # 查找它的前一部分用于上下文(如果存在)
                        prev_part = parts[i-1] if i > 0 else ""
                        
                        # 如果前部分是数字索引或简短字符，添加一起
                        if prev_part and (prev_part.isdigit() or len(prev_part) <= 3):
                            components.append(f"{prev_part}.{current_part}")
                        else:
                            components.append(current_part)
            
            # 如果没有找到任何关键组件，直接使用原始名称
            if not components:
                components = [name]
            
            # 将组件合并为一个名称，确保不超过最大长度
            formatted_name = ".".join(components)
            if len(formatted_name) > max_length:
                # 截断并添加省略号
                return formatted_name[:max_length-3] + "..."
            
            return formatted_name
        
        # 创建详细的y轴标签
        y_labels_min = [format_layer_name(name) for name in sorted_names[:num_display]]
        y_labels_max = [format_layer_name(name) for name in sorted_names[-num_display:]]
        
        # 绘制条形图
        y_pos_min = np.arange(num_display)
        plt.barh(y_pos_min, sorted_norms[:num_display], 
                alpha=0.7, color='blue', label='min grad')
        
        if len(sorted_norms) > num_display:
            y_pos_max = np.arange(num_display, 2*num_display)
            plt.barh(y_pos_max, sorted_norms[-num_display:], 
                    alpha=0.7, color='red', label='max grad')
            
            # 设置Y轴标签
            plt.yticks(np.concatenate([y_pos_min, y_pos_max]), 
                      y_labels_min + y_labels_max, fontsize=10)
        else:
            plt.yticks(y_pos_min, y_labels_min, fontsize=10)
        
        plt.title("Layer Gradient Distribution Comparison", fontsize=14)
        plt.xlabel("Gradient Norm (Log Scale)", fontsize=12)
        try:
            plt.xscale('log')
        except:
            plt.xscale('linear')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend(loc='upper right', fontsize=12)
        
        # 增加左边距给标签留更多空间
        plt.subplots_adjust(left=0.3)  # 增加到0.3，给更长的标签留出空间
    
    # 添加总标题
    plt.suptitle(f"Gradient Analysis - Step {step}", fontsize=18, y=0.98)
    
    # 保存图表
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为顶部标题留出空间
    plt.savefig(vis_dir / f"grad_distribution_step_{step}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 创建详细的梯度分析报告
    with open(vis_dir / f"grad_summary_step_{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(f"=== Gradient Distribution Analysis (Step {step}) ===\n\n")
        
        if len(layer_names) == 1 and layer_names[0] == "dummy":
            f.write("Insufficient gradient data for analysis.\n")
            logging.warning("Insufficient gradient data for analysis at step %d", step)
            return
        
        # 基本统计
        f.write("Basic Statistics:\n")
        f.write(f"  Total layers analyzed: {len(layer_names)}\n")
        small_grad_count = sum(1 for g in grad_norms if g < 1e-6)
        large_grad_count = sum(1 for g in grad_norms if g > 10.0)
        f.write(f"  Layers with small gradients (<1e-6): {small_grad_count} ({small_grad_count/len(grad_norms)*100:.1f}%)\n")
        f.write(f"  Layers with large gradients (>10.0): {large_grad_count} ({large_grad_count/len(grad_norms)*100:.1f}%)\n\n")
        
        # 问题模式分析
        pattern_analysis = {}
        for name, norm in zip(layer_names, grad_norms):
            if norm < 1e-6:  # 梯度过小
                # 识别模块类型
                for pattern in ["transformer_block", "mlp", "attn", 
                              "adaLN_modulation", "vision_encoder", "x_embedder", 
                              "final_layer", "weight", "bias"]:
                    if pattern in name:
                        pattern_analysis[pattern] = pattern_analysis.get(pattern, 0) + 1
        
        # 写入模式分析
        f.write("Problem Pattern Analysis:\n")
        f.write("  Small Gradient Pattern:\n")
        for pattern, count in pattern_analysis.items():
            f.write(f"    {pattern}: {count} layers\n")
        
        # 最小梯度层
        f.write("\nLayers with Smallest Gradients:\n")
        smallest_indices = np.argsort(grad_norms)[:10]  # 前10个最小梯度
        for idx in smallest_indices:
            f.write(f"  {layer_names[idx]}: {grad_norms[idx]:.10f}\n")
        
        # 最大梯度层
        f.write("\nLayers with Largest Gradients:\n")
        largest_indices = np.argsort(grad_norms)[-10:]  # 后10个最大梯度
        for idx in reversed(largest_indices):
            f.write(f"  {layer_names[idx]}: {grad_norms[idx]:.6f}\n")
    
    logging.info(f"Gradient distribution analysis saved to {vis_dir}")


def extract_gradient_statistics(grad_tracking):
    """Extract gradient statistics from tracked gradient data
    
    Args:
        grad_tracking: Dictionary or list containing gradient tracking information
    
    Returns:
        Dictionary containing:
            - layer_names: List of layer names
            - grad_norms: List of gradient norms for each layer
            - param_norms: List of parameter norms for each layer
            - grad_param_ratios: Ratio of gradient norm to parameter norm for each layer
    """
    # 初始化存储统计信息的字典
    stats = {
        "layer_names": [],
        "grad_norms": [],
        "param_norms": [],
        "grad_param_ratios": []
    }
    
    # 处理跟踪的梯度数据
    # 如果是列表，取最后一个元素(最新步骤)
    if isinstance(grad_tracking, list):
        if len(grad_tracking) > 0:
            grad_data = grad_tracking[-1]
        else:
            # 确保返回至少有一条记录的统计信息，避免空数组
            return {
                "layer_names": ["dummy"],
                "grad_norms": [0.001],  # 非零值以避免log scale错误
                "param_norms": [1.0],
                "grad_param_ratios": [0.001]
            }
    else:
        grad_data = grad_tracking  # 否则直接使用
    
    # 提取层名称和梯度信息
    for name, param in grad_data.items():
        # 跳过非参数项
        if not isinstance(param, dict) or 'grad_norm' not in param:
            continue
        
        grad_norm = param.get('grad_norm', 0.0)
        param_norm = param.get('param_norm', 1.0)
        
        # 确保梯度范数至少为一个很小的正数，避免对数缩放问题
        grad_norm = max(grad_norm, 1e-10)
        param_norm = max(param_norm, 1e-10)
        
        # 计算梯度与参数的比率
        grad_param_ratio = grad_norm / param_norm
        
        # 添加到统计信息
        stats["layer_names"].append(name)
        stats["grad_norms"].append(grad_norm)
        stats["param_norms"].append(param_norm)
        stats["grad_param_ratios"].append(grad_param_ratio)
    
    # 如果没有收集到任何有效数据，添加一个虚拟记录
    if len(stats["layer_names"]) == 0:
        stats["layer_names"].append("dummy")
        stats["grad_norms"].append(0.001)  # 非零值以避免log scale错误
        stats["param_norms"].append(1.0)
        stats["grad_param_ratios"].append(0.001)
    
    return stats


if __name__ == "__main__":
    train_cli()
