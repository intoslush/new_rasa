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

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from utils.grounding_viz import render_token_grounding_folder_one

def run_analysis_and_vis(args, model, data_loader, tokenizer, config):
    model.train()  # 确保模型处于训练模式, 如果需要梯度的话，train模式可以启用 Dropout 等操作
    device = torch.device("cuda")
    
    # === Fig.A 统计容器 ===
    # 存储格式: word -> {'sum_prob': float, 'count': int, 'sum_ground': float}
    token_stats = defaultdict(lambda: {'sum_prob': 0.0, 'count': 0, 'sum_ground': 0.0})
    
    # === Fig.B 可视化计数 ===
    vis_count = 0
    vis_limit = 20  # 限制输出图片的数量，只看几张效果
    vis_output_dir = os.path.join("./pic", "vis_output")
    os.makedirs(vis_output_dir, exist_ok=True)

    print("Start Analysis...")

    for i, batch in enumerate(data_loader):
        # 将数据移到 GPU
        batch = {k: (v.to(device, non_blocking=True) if hasattr(v, 'to') else v) for k, v in batch.items()}
        
        # 调用新增的 analysis forward
        ret = model.forward_analysis(batch, config)
        
        input_ids = ret['input_ids']          # [B, L]
        probs = ret['probability_matrix']     # [B, L]
        saliency = ret['saliency_scores']     # [B, L]
        cross_attentions = ret['cross_attentions'] 
        # cross_attentions 是 tuple (layers), 每层 [B, Heads, L, Img_Tokens]
        
        # 获取 Batch 大小
        B_size = input_ids.size(0)
        
        # ================= Fig.A: 统计 Token Mask 概率 =================
        input_ids_cpu = input_ids.cpu().numpy()
        probs_cpu = probs.cpu().numpy()
        saliency_cpu = saliency.cpu().numpy()
        
        for b in range(B_size):
            # 获取当前句子的 tokens (去除 pad)
            tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu[b])
            
            # WordPiece 合并逻辑：将 ##abc 归并到前一个词
            current_word = ""
            current_prob_sum = 0.0
            current_ground_sum = 0.0
            subtoken_count = 0
            
            for t_idx, token in enumerate(tokens):
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                
                p_val = probs_cpu[b, t_idx]
                g_val = saliency_cpu[b, t_idx]
                
                if token.startswith("##"):
                    current_word += token[2:]
                    current_prob_sum += p_val
                    current_ground_sum += g_val
                    subtoken_count += 1
                else:
                    # 结算上一个词
                    if current_word != "":
                        avg_p = current_prob_sum / max(1, subtoken_count)
                        avg_g = current_ground_sum / max(1, subtoken_count)
                        token_stats[current_word]['sum_prob'] += avg_p
                        token_stats[current_word]['sum_ground'] += avg_g
                        token_stats[current_word]['count'] += 1
                    
                    # 开启新词
                    current_word = token
                    current_prob_sum = p_val
                    current_ground_sum = g_val
                    subtoken_count = 1
            
            # 结算句尾最后一个词
            if current_word != "":
                avg_p = current_prob_sum / max(1, subtoken_count)
                avg_g = current_ground_sum / max(1, subtoken_count)
                token_stats[current_word]['sum_prob'] += avg_p
                token_stats[current_word]['sum_ground'] += avg_g
                token_stats[current_word]['count'] += 1

        # ================= Fig.B: 生成热力图 =================
        if vis_count < vis_limit:
            img_res = config.get('image_res', 224) 
            grid_size = img_res // 16 

            last_layers = cross_attentions[-3:] 
            avg_attn = torch.stack(last_layers, dim=0).mean(dim=0) # [B, Heads, L, P+1]
            avg_attn = avg_attn.mean(dim=1) # [B, L, P+1]
            patch_attn = avg_attn[:, :, 1:] # 去掉 CLS token -> [B, L, P]
            
            for b in range(B_size):
                if vis_count >= vis_limit: break
                
                seq_len = (input_ids[b] != tokenizer.pad_token_id).sum()
                if seq_len < 10 or seq_len > 35:
                    continue
                
                # --- 1. 图像反归一化 ---
                raw_image = batch['image1'][b].cpu().permute(1, 2, 0).numpy()
                
                mean = np.array([0.48145466, 0.4578275, 0.40821073])
                std =  np.array([0.26862954, 0.26130258, 0.27577711])
                
                raw_image = (raw_image * std + mean) * 255.0
                raw_image = np.clip(raw_image, 0, 255).astype(np.uint8)
                
                raw_image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
                
                img_h, img_w = raw_image_bgr.shape[:2]

                # --- 2. 选词与绘图 ---
                valid_indices = []
                tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu[b])
                
                for idx, t in enumerate(tokens):
                    if t not in ['[CLS]', '[SEP]', '[PAD]', '.', ',', 'a', 'the', 'is'] and not t.startswith("##"):
                        valid_indices.append(idx)
                
                if len(valid_indices) < 3: continue

                valid_scores = saliency[b, valid_indices]
                k_val = min(3, len(valid_indices))
                topk_vals, topk_ind_in_valid = torch.topk(valid_scores, k=k_val)
                target_token_indices = [valid_indices[i] for i in topk_ind_in_valid.cpu().numpy()]
                
                for t_idx in target_token_indices:
                    token_word = tokens[t_idx]
                    
                    # 修复问题：detach()后再转换为numpy
                    att_map = patch_attn[b, t_idx, :].reshape(grid_size, grid_size).detach().cpu().numpy()
                    
                    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
                    
                    heatmap = cv2.resize(att_map, (img_w, img_h))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    
                    overlay = cv2.addWeighted(raw_image_bgr, 0.6, heatmap, 0.4, 0)
                    
                    pid = batch['person_id'][b].item() if 'person_id' in batch else b
                    save_name = f"id{pid}_word_{token_word}_score{saliency[b, t_idx]:.2f}.jpg"
                    cv2.imwrite(os.path.join(vis_output_dir, save_name), overlay)
                
                vis_count += 1

    # === 保存 Fig.A 统计数据 ===
    final_stats = []
    for word, data in token_stats.items():
        if data['count'] > 0:
            final_stats.append({
                'word': word,
                'freq': data['count'],
                'avg_prob': data['sum_prob'] / data['count'],
                'avg_ground': data['sum_ground'] / data['count']
            })
            
    with open(os.path.join("./pic", "token_stats_fig_a.json"), "w") as f:
        json.dump(final_stats, f, indent=4)
        
    print(f"Analysis Done. Stats saved to ./pic/token_stats_fig_a.json")
    print(f"Vis saved to {vis_output_dir}")



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
    # if args.resume:
    #     # 确保加载了预训练权重
    #     checkpoint = torch.load(args.resume_ckpt_file, map_location='cpu', weights_only=False)
    #     model.load_state_dict(checkpoint['model'], strict=False)
        
    #     run_analysis_and_vis(
    #         args=args, 
    #         model=model.to(device), 
    #         data_loader=train_loader, # 使用训练集做 mask 统计
    #         tokenizer=model.tokenizer, # 假设 tokenizer 在 model 里，或者从外部传
    #         config=config
    #     )
    # return

    # Optimizer & Scheduler
    optimizer, scheduler = _setup_optim_sched(config, model)
    checkpoint = torch.load(args.resume_ckpt_file, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
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

    for epoch in range(0,  1):
        epoch=30
        if is_distributed:
            dist.barrier()
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_valid_indices'):
                train_loader.sampler.set_valid_indices(train_loader.dataset.valid_indices)
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
                


        

        # ========== 4) 进入训练态 ==========
        model.train()


        dynamic_weights = compute_dynamic_weights(
            epoch=epoch,
            num_epoch=num_epoch,
            base_weights=config["weights"],
            schedule=config.get("weights_schedule", None),
        )
        if tb_writer is not None and is_main:
            _tb(tb_writer, dynamic_weights, "Weights", epoch)
            logger.info(f"[Rank {rank}] epoch {epoch} 使用的 loss 权重: {dynamic_weights}")
        cou=1
        # for n_iter, batch in enumerate(train_loader):
        #     # move to device
        #     batch = {k: (v.to(device, non_blocking=True) if hasattr(v, 'to') else v) for k, v in batch.items()}
            
        #     if epoch > 0 or not config.get('warm_up', False):
        #         alpha = config['alpha']
        #     else:
        #         alpha = config['alpha'] * min(1.0, (n_iter + 1) / len(train_loader))

            # # ====== 可视化分支（不影响训练）======
        
            # if config.get("viz_enable", True):
            #     period = int(config.get("viz_period", 1))
            #     if True:
            #         do_viz = True
            #         if "RANK" in os.environ:
            #             do_viz = (int(os.environ["RANK"]) == 0)

            #         if do_viz:
            #             model_ = model.module if hasattr(model, "module") else model
            #             prev_mode = model_.training

            #             out_dir = config.get("viz_out_dir", "./viz_figB")
            #             num_samples = int(config.get("viz_num_samples", 50))   # 50句
            #             num_tokens  = int(config.get("viz_num_tokens", 10))    # 每句10token
            #             layers = int(config.get("viz_layers", 3))
            #             cap_key = config.get("viz_use_caption", "caption1")

            #             image_paths = batch["image_path"]
            #             captions = batch[cap_key]

            #             B = len(image_paths)
            #             take = min(num_samples, B)

            #             # 每次触发可视化，单独建一个 run 目录，避免覆盖
            #             run_dir = os.path.join(out_dir, f"e{epoch}_step{global_step}_iter{n_iter}")
            #             os.makedirs(run_dir, exist_ok=True)

            #             for b in range(take):
            #                 img_name = os.path.splitext(os.path.basename(image_paths[b]))[0]
            #                 sent_dir = os.path.join(run_dir, f"b{b:03d}_{img_name}")

            #                 render_token_grounding_folder_one(
            #                     model=model_,
            #                     tokenizer=model_.tokenizer,
            #                     image_path=image_paths[b],
            #                     caption=captions[b],
            #                     image_res=int(config.get("image_res", 384)),
            #                     num_tokens=num_tokens,
            #                     layers=layers,
            #                     skip_wordpiece=True,
            #                     seed=global_step + b,
            #                     out_dir=sent_dir,              # 关键：输出到文件夹
            #                     overview_name="overview.png",  # 总览图名字
            #                     write_caption_txt=True,        # 写 caption.txt
            #                 )

            #             # 如果你希望恢复模式（一般不需要改，保持你原逻辑）
            #             # model_.train(prev_mode)
                            
                        
            #     if cou<5:
            #         cou+=1
            #         print("正在可视化中，跳过训练步骤")
            #         continue
            #     else:
            #         return
            
            # if n_iter % args.log_period == 0:
            #     logger.info(f"开始 epoch {epoch} 的第 {n_iter}/{len(train_loader)} 个 batch 的 loss 计算")
            # batch["global_step"]=global_step
            # with autocast(device_type=device.type, enabled=use_amp):
            #     loss_dict: Dict[str, torch.Tensor] = model(batch, alpha, config, epoch)
            #     loss = 0.0
            #     for k, v in loss_dict.items():
            #         w = dynamic_weights.get(k, config["weights"].get(k, 0.5)) #默认 0.5
            #         loss = loss + w * v

            # optimizer.zero_grad(set_to_none=True)
            # return
            # if use_amp:
            #     scaler.scale(loss).backward()
            #     scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #     optimizer.step()

            # # iteration-level warmup
            # if epoch == 0 and n_iter % step_size == 0 and n_iter <= warmup_iterations:
            #     try:
            #         scheduler.step(n_iter // step_size)
            #     except Exception:
            #         pass

            # # TB: loss & meta
            # if tb_writer is not None and is_main:
            #     global_step += 1
            #     if (global_step % tb_every) == 0:
            #         tb_writer.add_scalars("LossGroup", {k: v.item() for k, v in loss_dict.items()}, global_step)
            #         current_lr = optimizer.param_groups[0]['lr']
            #         tb_writer.add_scalars("Meta", {"LearningRate": current_lr, "Epoch": epoch}, global_step)

            # # 释放临时变量
            # del loss_dict, loss

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


