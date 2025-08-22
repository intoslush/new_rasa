import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from .cluster import cluster_begin_epoch
import ruamel.yaml as YAML 
import utils.optimizer as utils
from optim import create_optimizer
from scheduler import create_scheduler
from .eval import itm_eval, evaluation
import os
import json
import torch.distributed as dist

# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler

from math import cos, pi

def _interp(base_v: float, target_v: float, t: float, mode: str = "linear") -> float:
    t = max(0.0, min(1.0, t))
    if mode == "cosine":
        # t=0 时为 base，t=1 时为 target，余弦缓入缓出
        return target_v + (base_v - target_v) * 0.5 * (1.0 + cos(pi * t))
    # 默认线性
    return base_v + (target_v - base_v) * t

def compute_dynamic_weights(epoch: int,
                            num_epoch: int,
                            base_weights: dict,
                            schedule: dict | None = None) -> dict:
    """
    base_weights: 例如 {"loss_cl":0.5, "loss_pitm":1, "loss_mlm":1, "loss_prd":0.5, "loss_mrtd":0.5}
    schedule 示例见下方 config 片段。
    """
    if schedule is None:
        schedule = {}

    freeze = int(schedule.get("freeze_epochs", 20))
    mode = schedule.get("mode", "linear")          # "linear" 或 "cosine"

    # 冻结阶段：直接返回初始权重
    if epoch <= freeze:
        return dict(base_weights)

    # 进度 t：从冻结结束到训练结束线性映射到 [0,1]
    denom = max(1, (num_epoch - freeze))
    t = (epoch - freeze) / denom
    t = max(0.0, min(1.0, t))

    w = dict(base_weights)

    # loss_cl：下降到最小值
    cl_cfg = schedule.get("loss_cl", {})
    if "min" in cl_cfg:
        w["loss_cl"] = _interp(base_weights.get("loss_cl", 0.0),
                               float(cl_cfg["min"]), t, mode)

    # loss_mlm：上升到最大值
    mlm_cfg = schedule.get("loss_mlm", {})
    if "max" in mlm_cfg:
        w["loss_mlm"] = _interp(base_weights.get("loss_mlm", 0.0),
                                float(mlm_cfg["max"]), t, mode)
    # loss_pitm：下降到最小值
    pitm_cfg = schedule.get("loss_pitm", {})
    if "min" in pitm_cfg:
        w["loss_pitm"] = _interp(base_weights.get("loss_pitm", 0.0),
                               float(cl_cfg["min"]), t, mode)

    # （可选）保持权重和不变：让总权重与初始总和一致
    if schedule.get("normalize_sum", False):
        base_sum = sum(base_weights.values())
        new_sum = sum(w.values())
        if new_sum > 0:
            scale = base_sum / new_sum
            for k in w:
                w[k] *= scale

    return w


def do_train(start_epoch, args, model, train_loader, evaluator, checkpointer, cluster_loader, test_loader):
    log_period = args.log_period
    eval_period = args.eval_period
    device = torch.device("cuda")
    num_epoch = args.num_epoch

    # distributed / main process flags
    is_distributed = args.distributed
    is_main = (not is_distributed) or dist.get_rank() == 0
    rank = dist.get_rank() if is_distributed else 0

    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger(args.name)
    logger.info("start training (rank %s)", rank)

    # TensorBoard only on main
    if is_main:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'), flush_secs=10)
        global_step = 0

    yaml = YAML.YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))

    # Optimizer and scheduler
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    scheduler, _ = create_scheduler(arg_sche, optimizer)

    # train config
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_epochs = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0
    best_log = ''
    step_size = 100
    warmup_iterations = warmup_epochs * step_size

    # AMP setup: 仅在 CUDA 上启用
    use_amp = getattr(args, "use_amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    if is_distributed:
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # make output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoint"), exist_ok=True)

    for epoch in range(start_epoch, num_epoch + 1):
        # === pseudo-label generation (only on main) ===
        
        if epoch<38:
            cluster_loader.dataset.mode = 'cluster'
            if is_main:
                with torch.no_grad():
                    image_pseudo_labels_np = cluster_begin_epoch(cluster_loader, model, args, config, None, logger)
                    image_num_cluster = len(set(image_pseudo_labels_np)) - (1 if -1 in image_pseudo_labels_np else 0)
                    logger.info("==> Statistics for epoch [{}]: {} image clusters, total {}".format(
                        epoch, image_num_cluster, len(image_pseudo_labels_np)))
                    image_pseudo_labels = torch.tensor(image_pseudo_labels_np, dtype=torch.long).to(device, non_blocking=True)
                    noise_frac = (image_pseudo_labels == -1).float().mean().item()
                    tb_writer.add_scalar("Cluster/num_clusters", image_num_cluster, epoch)
                    tb_writer.add_scalar("Cluster/noise_frac", noise_frac, epoch)
                    # free the numpy copy early
                    del image_pseudo_labels_np
            else:
                print(f"[Rank {rank}] 等待主进程生成伪标签")
                image_pseudo_labels = torch.empty(len(cluster_loader.dataset), dtype=torch.long, device=device)

            if is_distributed:
                dist.broadcast(image_pseudo_labels, src=0)
                dist.barrier()
            # apply pseudo labels
            train_loader.dataset.mode = 'train'
            train_loader.dataset.set_pseudo_labels(image_pseudo_labels.cpu())
        
        # scheduler step per epoch (after warmup epoch 0)
        if epoch > 0:
            try:
                scheduler.step(epoch)
            except Exception:
                # some schedulers expect no args
                scheduler.step()

        if is_distributed:
            dist.barrier()
            # update sampler state
            train_loader.sampler.set_valid_indices(train_loader.dataset.valid_indices)
            train_loader.sampler.set_epoch(epoch)

        model.train()

        logger.info(f"[Rank {rank}] 开始 epoch {epoch} mini-batch 循环")
        dynamic_weights = compute_dynamic_weights(
            epoch=epoch,
            num_epoch=num_epoch,
            base_weights=config["weights"],
            schedule=config.get("weights_schedule", None)
        )
        if is_main:
            # 记录到 TensorBoard 方便对齐实验
            for k, v in dynamic_weights.items():
                tb_writer.add_scalar(f"Weights/{k}", float(v), epoch)
            logger.info(f"[Rank {rank}] epoch {epoch} 使用的 loss 权重: {dynamic_weights}")
        # training loop
        for n_iter, batch in enumerate(train_loader):
            # move batch to device (non_blocking if pinned)
            batch = {
                k: v.to(device, non_blocking=True) if hasattr(v, 'to') else v
                for k, v in batch.items()
            }

            # compute alpha (warmup for first epoch if configured)
            if epoch > 0 or not config.get('warm_up', False):
                alpha = config['alpha']
            else:
                alpha = config['alpha'] * min(1.0, (n_iter + 1) / len(train_loader))

            if n_iter % log_period == 0:
                logger.info(f"开始 epoch {epoch} 的第 {n_iter}/{len(train_loader)} 个 batch 的 loss 计算")

            # forward + loss under autocast (torch.amp requires device_type)
            with autocast(device_type=device.type, enabled=use_amp):
                loss_dict = model(batch, alpha, config,epoch)
                loss = 0.0
                for k, v in loss_dict.items():
                    w = dynamic_weights.get(k, config["weights"].get(k, 1.0))
                    loss += w * v

            # zero grads
            optimizer.zero_grad(set_to_none=True)

            # backward with scaling if AMP
            if use_amp:
                scaler.scale(loss).backward()
                # unscale for clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # iteration-level warmup (only if originally in warmup phase)
            if epoch == 0 and n_iter % step_size == 0 and n_iter <= warmup_iterations:
                try:
                    scheduler.step(n_iter // step_size)
                except Exception:
                    pass  # some schedulers might not support this

            # logging
            if is_main:
                global_step += 1
                tb_writer.add_scalars("LossGroup", {k: v.item() for k, v in loss_dict.items()}, global_step)
                current_lr = optimizer.param_groups[0]['lr']
                tb_writer.add_scalars("Meta", {
                    "LearningRate": current_lr,
                    "Epoch": epoch,
                }, global_step)

            # clean up per-iteration temporaries
            del loss_dict, loss

        logger.info(f"---------- epoch {epoch} 训练完成 -------------")

        # evaluation + checkpoint logic
        with torch.no_grad():
            if epoch >= config.get('eval_epoch', 0) or args.evaluate:
                score_test_t2i = evaluation(model_without_ddp, test_loader, model_without_ddp.tokenizer, device, config, args)
                if utils.is_main_process():
                    test_result = itm_eval(score_test_t2i, test_loader.dataset.img2person, test_loader.dataset.txt2person, args.eval_mAP)
                    # print('Test:', test_result, '\n')
                    logger.info(f"Test result: {test_result}")
                    if args.evaluate:
                        log_stats = {'epoch': epoch,
                                    **{f'test_{k}': v for k, v in test_result.items()}
                                    }
                        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                            f.write(json.dumps(log_stats) + "\n")
                    else:
                        log_stats = {'epoch': epoch,
                                    **{f'test_{k}': v for k, v in test_result.items()},
                                    }
                        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                            f.write(json.dumps(log_stats) + "\n")

                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': scheduler.state_dict(),
                            'config': config,
                            'epoch': epoch,
                            'best': best,
                            'best_epoch': best_epoch
                        }
                        if test_result['r1'] > best:
                            torch.save(save_obj, os.path.join(args.output_dir, "checkpoint", 'checkpoint_best.pth'))
                            best = test_result['r1']
                            best_epoch = epoch
                            best_log = log_stats
                    if is_main:
                        for key, value in test_result.items():
                            tb_writer.add_scalar(f"Eval/{key}", value, epoch)

        # synchronize end of epoch
        if is_distributed:
            dist.barrier()

        # clear cache to reduce fragmentation
        torch.cuda.empty_cache()

    # final logging of best
    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
        f.write(json.dumps(best_log) + "\n")

    if is_main:
        tb_writer.close()
        




