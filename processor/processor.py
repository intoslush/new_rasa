import os
import logging
import json
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from typing import Dict, Any

from ruamel.yaml import YAML

# 来自你的工程
import utils.optimizer as utils
from optim import create_optimizer
from scheduler import create_scheduler

# 拆分后的模块（同包内相对导入）
from .weights import compute_dynamic_weights
from .pseudo import generate_and_broadcast_pseudo_labels
from .eval_hooks import evaluate_and_checkpoint
from io import StringIO
import pprint

def _setup_optim_sched(config: Dict[str, Any], model):
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    scheduler, _ = create_scheduler(arg_sche, optimizer)
    return optimizer, scheduler


def _tb(writer: SummaryWriter, scalars: Dict[str, float], tag_prefix: str, step: int):
    for k, v in scalars.items():
        writer.add_scalar(f"{tag_prefix}/{k}", float(v), step)


def do_train(start_epoch, args, model, train_loader, evaluator, checkpointer, cluster_loader, test_loader):
    device = torch.device("cuda")
    num_epoch = args.num_epoch

    # 分布式标志位
    is_distributed = args.distributed
    rank = dist.get_rank() if is_distributed else 0
    is_main = (not is_distributed) or (rank == 0)

    logger = logging.getLogger(args.name)
    logger.info("start training (rank %s)", rank)

    # 仅主进程写 TB
    tb_writer = None
    global_step = 0
    if is_main:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'), flush_secs=60)
    # 写入频率（默认每 20 步；也可通过 args.tb_every 覆盖）
    tb_every = getattr(args, "tb_every", 50)
    if is_main:
        logger.info(f"TensorBoard scalars will be logged every {tb_every} steps")

    # 读取 YAML 配置
    yaml = YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))

    # Optimizer & Scheduler
    optimizer, scheduler = _setup_optim_sched(config, model)

    # 训练日历
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_epochs = config['schedular']['warmup_epochs']
    step_size = 100
    warmup_iterations = warmup_epochs * step_size

    # AMP
    use_amp = getattr(args, "use_amp", True) and device.type == "cuda"
    logger.info(f"使用 AMP: {use_amp}")
    scaler = GradScaler(enabled=use_amp)

    # 获取 DDP 包裹的实际模型
    model_without_ddp = model.module if is_distributed else model

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoint"), exist_ok=True)

    best = 0.0
    best_epoch = 0
    best_log = {}

    for epoch in range(start_epoch, num_epoch + 1):
        # ====== 第七个 epoch dump args + config 到日志 ======
        dump_epoch_1based = 7
        target_epoch = start_epoch + dump_epoch_1based - 1  # start_epoch=0 -> target_epoch=6

        if is_main and epoch == target_epoch:
            logger.info("========== [Dump args & config @ epoch=%d] ==========", epoch)
            try:
                logger.info("args (vars):\n%s", pprint.pformat(vars(args), width=120, sort_dicts=False))
            except Exception:
                logger.info("args:\n%s", str(args))
            try:
                buf = StringIO()
                yaml.dump(config, buf)
                logger.info("config (yaml):\n%s", buf.getvalue())
            except Exception:
                logger.info("config (repr):\n%s", pprint.pformat(config, width=120))

            logger.info("========== [End dump] ==========")
        # =======================================================
        if epoch<5 or epoch%2==1:
            image_pseudo_labels = generate_and_broadcast_pseudo_labels(
                epoch=epoch,
                device=device,
                is_main=is_main,
                is_distributed=is_distributed,
                rank=rank,
                cluster_loader=cluster_loader,
                model=model,
                args=args,
                config=config,
                logger=logger,
                tb_writer=tb_writer,
                enable_nmi_ari=True,
                cluster_until_epoch=50,
            )
            train_loader.dataset.mode = 'train'
            train_loader.dataset.set_pseudo_labels(image_pseudo_labels.cpu())

            if bool(config.get('reset_queue_each_epoch', True)):
                if is_distributed:
                    dist.barrier()
                model.reset_queues(random_init=bool(config.get('queue_random_reinit', False)))
                if is_main:
                    logger.info(
                        f"[Rank {rank}] 已清空对比队列 (random_init={bool(config.get('queue_random_reinit', False))})"
                    )
                if is_distributed:
                    dist.barrier()
        if is_distributed:
            dist.barrier()
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_valid_indices'):
                train_loader.sampler.set_valid_indices(train_loader.dataset.valid_indices)
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
                
        if epoch > 0:
            try:
                scheduler.step(epoch)
            except Exception:
                scheduler.step()

        

        # ========== 4) 进入训练态 ==========
        model.train()
        logger.info(f"[Rank {rank}] 开始 epoch {epoch} mini-batch 循环")

        dynamic_weights = compute_dynamic_weights(
            epoch=epoch,
            num_epoch=num_epoch,
            base_weights=config["weights"],
            schedule=config.get("weights_schedule", None),
        )
        if tb_writer is not None and is_main:
            _tb(tb_writer, dynamic_weights, "Weights", epoch)
            logger.info(f"[Rank {rank}] epoch {epoch} 使用的 loss 权重: {dynamic_weights}")
        for n_iter, batch in enumerate(train_loader):
            # move to device
            batch = {k: (v.to(device, non_blocking=True) if hasattr(v, 'to') else v) for k, v in batch.items()}
            if epoch > 0 or not config.get('warm_up', False):
                alpha = config['alpha']
            else:
                alpha = config['alpha'] * min(1.0, (n_iter + 1) / len(train_loader))

            if n_iter % args.log_period == 0:
                logger.info(f"开始 epoch {epoch} 的第 {n_iter}/{len(train_loader)} 个 batch 的 loss 计算")
            batch["global_step"]=global_step
            with autocast(device_type=device.type, enabled=use_amp):
                loss_dict: Dict[str, torch.Tensor] = model(batch, alpha, config, epoch)
                loss = 0.0
                for k, v in loss_dict.items():
                    w = dynamic_weights.get(k, config["weights"].get(k, 0.5)) #默认 0.5
                    loss = loss + w * v

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # iteration-level warmup
            if epoch == 0 and n_iter % step_size == 0 and n_iter <= warmup_iterations:
                try:
                    scheduler.step(n_iter // step_size)
                except Exception:
                    pass

            # TB: loss & meta
            if tb_writer is not None and is_main:
                global_step += 1
                if (global_step % tb_every) == 0:
                    tb_writer.add_scalars("LossGroup", {k: v.item() for k, v in loss_dict.items()}, global_step)
                    current_lr = optimizer.param_groups[0]['lr']
                    tb_writer.add_scalars("Meta", {"LearningRate": current_lr, "Epoch": epoch}, global_step)

            # 释放临时变量
            del loss_dict, loss

        logger.info(f"---------- epoch {epoch} 训练完成 -------------")

        # ========== 5) 评估与保存 ==========
        with torch.no_grad():
            if epoch >= config.get('eval_epoch', 0) or args.evaluate or (epoch == 0):
                best, best_epoch, best_log_epoch = evaluate_and_checkpoint(
                    model_without_ddp=model_without_ddp,
                    test_loader=test_loader,
                    device=device,
                    config=config,
                    args=args,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best=best,
                    best_epoch=best_epoch,
                    output_dir=args.output_dir,
                    logger=logger,
                    tb_writer=tb_writer,
                )
                if best_log_epoch:
                    best_log = best_log_epoch

        if is_distributed:
            dist.barrier()

        torch.cuda.empty_cache()

    # 写入最终最优
    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
        f.write(json.dumps(best_log) + "\n")

    if tb_writer is not None:
        tb_writer.close()