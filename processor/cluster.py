import os
import gc
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.cluster import DBSCAN  # 可选：如果内存允许、想用更快的 C 实现，可以开启下面的 sklearn 分支

@torch.no_grad()
def _infer_cluster_feat_dim(model_no_ddp, config, mode: str) -> int:
    mode = mode.lower()
    if mode == "image":
        return int(model_no_ddp.vision_proj.out_features)  # embed_dim
    elif mode == "text":
        use_proj = bool(config.get("cluster_text_use_proj", False))
        if use_proj:
            return int(model_no_ddp.text_proj.out_features)  # embed_dim
        return int(model_no_ddp.text_encoder.config.hidden_size)  # text_width
    elif mode == "fusion":
        use_proj = bool(config.get("cluster_fusion_use_proj", False))
        if use_proj:
            return int(model_no_ddp.text_proj.out_features)  # embed_dim
        return int(model_no_ddp.text_encoder.config.hidden_size)  # text_width
    else:
        raise ValueError(f"Unknown cluster_feature_mode: {mode}")


@torch.no_grad()
def extract_cluster_features(batch, model_no_ddp, tokenizer, config, device: torch.device) -> torch.Tensor:
    """
    返回 [B, D] 的 L2-normalized 特征，用于后续 jaccard + DBSCAN。
    由 config['cluster_feature_mode'] 控制:
      - 'image' : vision_proj(image_cls)
      - 'text'  : text_cls (可选 text_proj)
      - 'fusion': fusion_cls (可选 text_proj)
    """
    mode = str(config.get("cluster_feature_mode", "image")).lower()

    # ---------------- image-only ----------------
    if mode == "image":
        image = batch["image1"].to(device, non_blocking=True)
        image_embeds = model_no_ddp.visual_encoder(image)
        feat = model_no_ddp.vision_proj(image_embeds[:, 0, :])
        return F.normalize(feat, dim=-1)

    # text / fusion 需要 tokenizer
    if tokenizer is None:
        raise ValueError("tokenizer is required when cluster_feature_mode is 'text' or 'fusion'.")

    # 选用哪个 caption 做聚类（默认 caption2；你也可以切 caption1）
    text_key = str(config.get("cluster_text_key", "caption2"))
    texts = batch[text_key]
    text = tokenizer(
        texts,
        padding="longest",
        max_length=int(config["max_words"]),
        return_tensors="pt",
        truncation=True,
    ).to(device)

    # ---------------- text-only ----------------
    if mode == "text":
        out = model_no_ddp.text_encoder.bert(
            text["input_ids"],
            attention_mask=text["attention_mask"],
            return_dict=True,
            mode="text",
        )
        cls = out.last_hidden_state[:, 0, :]  # [B, text_width]

        if bool(config.get("cluster_text_use_proj", False)):
            cls = model_no_ddp.text_proj(cls)  # -> [B, embed_dim]
        return F.normalize(cls, dim=-1)

    # ---------------- fusion CLS ----------------
    if mode == "fusion":
        image = batch["image1"].to(device, non_blocking=True)
        image_embeds = model_no_ddp.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

        # 先跑一遍 text mode 得到 encoder_embeds（保持和你 forward 一致）
        text_out = model_no_ddp.text_encoder.bert(
            text["input_ids"],
            attention_mask=text["attention_mask"],
            return_dict=True,
            mode="text",
        )
        text_embeds = text_out.last_hidden_state  # [B, L, text_width]

        fusion_out = model_no_ddp.text_encoder.bert(
            encoder_embeds=text_embeds,
            attention_mask=text["attention_mask"],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode="fusion",
        )
        cls = fusion_out.last_hidden_state[:, 0, :]  # [B, text_width]  <-- 你要的“融合层最后一层 CLS”

        if bool(config.get("cluster_fusion_use_proj", False)):
            cls = model_no_ddp.text_proj(cls)  # -> [B, embed_dim]
        return F.normalize(cls, dim=-1)

    raise ValueError(f"Unknown cluster_feature_mode: {mode}")


def compute_jaccard_to_memmap(
    features: torch.Tensor,
    out_path: str,
    k1: int = 30,
    k2: int = 6,
    use_float16: bool = True,
    row_chunk: int = 1024,
    search_option=0
) -> str:
    """
    逐行计算 re-ranking 的 jaccard distance 并写入 disk-backed memmap，避免在 CPU 一次性占用 O(N^2) 内存。
    返回 memmap 文件路径（out_path）。
    features: Tensor[N, D] (will be moved to GPU inside)
    """
    device = torch.device("cuda")
    dtype = torch.float16 if use_float16 else torch.float32

    feats = features.to(device=device, dtype=dtype)  # [N, D]
    N, D = feats.shape
    feats_t = feats.t()  # reuse for similarity

    # 1) initial rank (k1 + 1 neighbors)
    initial_rank = torch.empty((N, k1 + 1), dtype=torch.long, device=device)
    for start in range(0, N, row_chunk):
        end = min(start + row_chunk, N)
        sim = torch.matmul(feats[start:end], feats_t)  # [b, N]
        _, idx = torch.topk(sim, k=k1 + 1, dim=1, largest=True, sorted=True)
        initial_rank[start:end] = idx
        del sim, idx

    # 2) reciprocal neighbors (在 CPU 上做 set 运算)
    init_cpu = initial_rank.cpu()
    nn_k1 = []
    nn_k1_half = []
    half_k1 = (k1 + 1) // 2
    for i in range(N):
        neigh = init_cpu[i]  # (k1+1,)
        back = init_cpu[neigh][:, : k1 + 1]  # [k1+1, k1+1]
        mask = (back == i).any(dim=1)
        k_recip = neigh[mask]
        nn_k1.append(k_recip)

        neigh2 = init_cpu[i][: half_k1 + 1]
        back2 = init_cpu[neigh2][:, : half_k1 + 1]
        mask2 = (back2 == i).any(dim=1)
        nn_k1_half.append(neigh2[mask2])

    # 3) 构造 V（dense，放在 GPU 上），然后逐行计算 jaccard 并 flush 到 disk
    V = torch.zeros((N, N), dtype=dtype, device=device)  # 可能 ~9GB float16 for N=68000

    for i in range(N):
        k_recip_cpu = nn_k1[i]
        if k_recip_cpu.numel() == 0:
            continue
        exp_set = set(int(x) for x in k_recip_cpu.tolist())

        for cand in list(exp_set):
            neigh2_cpu = nn_k1_half[cand]
            if neigh2_cpu.numel() == 0:
                continue
            common = len(set(neigh2_cpu.tolist()) & set(k_recip_cpu.tolist()))
            if common > (2 / 3) * neigh2_cpu.numel():
                exp_set.update(int(x) for x in neigh2_cpu.tolist())

        exp = torch.tensor(sorted(exp_set), device=device, dtype=torch.long)
        if exp.numel() == 0:
            continue
        dist = 2.0 - 2.0 * torch.matmul(feats[i].unsqueeze(0), feats[exp].t())  # [1, M]
        weights = F.softmax(-dist, dim=1).view(-1)  # [M]
        V[i, exp] = weights  # sparse fill

    # 4) query expansion
    if k2 != 1:
        for i in range(N):
            idx = initial_rank[i, :k2]
            V[i] = V[idx].mean(dim=0)

    # 5) 准备 memmap 输出
    dtype_np = np.float16 if use_float16 else np.float32
    dirname = os.path.dirname(out_path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    jaccard_mmap = np.memmap(out_path, dtype=dtype_np, mode="w+", shape=(N, N))

    # 6) 逐行计算 jaccard 并写入 disk
    for i in range(N):
        v_i = V[i]  # [N] on GPU
        nz = torch.nonzero(v_i, as_tuple=False).view(-1)
        if nz.numel() == 0:
            row_gpu = torch.ones((N,), dtype=dtype, device=device)
        else:
            V_sub = V[:, nz]  # [N, K]
            v_i_sub = v_i[nz].unsqueeze(0)  # [1, K]
            tmp_min = torch.min(V_sub, v_i_sub).sum(dim=1)  # [N]
            row_gpu = 1.0 - tmp_min / (2.0 - tmp_min)  # [N]
        row_cpu = row_gpu.to("cpu").to(dtype if use_float16 else dtype).numpy()
        jaccard_mmap[i, :] = row_cpu  # 写一行

    jaccard_mmap.flush()

    # 清理 GPU 资源
    del V, feats, feats_t, initial_rank
    torch.cuda.empty_cache()
    gc.collect()

    return out_path  # memmap 路径


def dbscan_memmap(jaccard_path: str, eps: float = 0.6, min_samples: int = 4):
    """
    在磁盘-backed 的 jaccard memmap 上做 DBSCAN（不一次性读入整个矩阵）。
    返回 numpy labels。
    """
    jaccard = np.memmap(jaccard_path, dtype=np.float16, mode="r")
    N = int(np.sqrt(jaccard.size))
    jaccard = jaccard.reshape((N, N))

    core_mask = np.zeros(N, dtype=bool)
    # 1. core 判断（按行读）
    for i in range(N):
        row = jaccard[i]
        if np.count_nonzero(row <= eps) >= min_samples:
            core_mask[i] = True

    labels = -1 * np.ones(N, dtype=int)
    visited = np.zeros(N, dtype=bool)
    cluster_id = 0

    # 2. 聚类扩展（经典 DBSCAN BFS）
    for i in range(N):
        if visited[i] or not core_mask[i]:
            continue
        labels[i] = cluster_id
        visited[i] = True
        queue = [i]
        while queue:
            curr = queue.pop()
            curr_row = jaccard[curr]
            nbrs = np.nonzero(curr_row <= eps)[0]
            for nb in nbrs:
                if not visited[nb]:
                    visited[nb] = True
                    if core_mask[nb]:
                        queue.append(nb)
                if labels[nb] == -1:
                    labels[nb] = cluster_id
        cluster_id += 1

    return labels  # numpy array


@torch.no_grad()
def cluster_begin_epoch(train_loader, model, args, config, tokenizer=None, logger=None):
    device = torch.device("cuda")

    model = model.to(device)
    model.eval()
    if args.distributed and hasattr(model, "module"):
        model_no_ddp = model.module
    else:
        model_no_ddp = model

    max_size = len(train_loader.dataset)
    feat_dim = _infer_cluster_feat_dim(model_no_ddp, config, mode=str(config.get("cluster_feature_mode", "image")))
    # bank 放 GPU，float16 足够
    feat_bank = torch.empty((max_size, feat_dim), device=device, dtype=torch.float16)

    index = 0

    # 输出路径按模式区分，避免互相覆盖
    mode = str(config.get("cluster_feature_mode", "image")).lower()
    save_path = f"./logs/pseudo_labels_{mode}.pt"
    save_feats = f"./logs/feats_{mode}.pt"
    jaccard_path = f"./tmp/{mode}_rerank_jaccard.memmap"
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./tmp", exist_ok=True)

    test = False  # 保留你的缓存开关

    logger.info(f"开始计算伪标签 | cluster_feature_mode={mode} | feat_dim={feat_dim}")

    # 1) load cached features if exist
    if test and os.path.exists(save_feats):
        bank_cpu = torch.load(save_feats, weights_only=False)
        feat_bank = bank_cpu.to(device=device, dtype=torch.float16)
        index = feat_bank.size(0)
        logger.info(f"检测到已保存的特征，加载 {save_feats}")
    else:
        for i, batch in enumerate(train_loader):
            # 关键：按模式抽取 [B, D] 特征
            feats = extract_cluster_features(batch, model_no_ddp, model.tokenizer, config, device)  # float32/16
            feats = feats.to(torch.float16)

            bs = feats.size(0)
            feat_bank[index:index + bs] = feats
            index += bs

        # 可选保存（仅 rank0）
        if test and (not args.distributed or (args.distributed and torch.distributed.get_rank() == 0)):
            torch.save(feat_bank[:index].cpu(), save_feats)
            logger.info(f"特征已保存至 {save_feats}")

    if index != max_size:
        # 重要：如果你在 DDP 下用 DistributedSampler，这里通常会触发（每张卡只看一部分数据）
        logger.warning(f"注意：feature bank 填充长度 index={index} != dataset_len={max_size}。"
                       f"如果你在分布式聚类，请确保用全量数据/或先 gather 特征。")

    # 2) 计算 jaccard 并写 memmap（只用已经填充的部分）
    feat_bank_used = feat_bank[:index].to(torch.float32)

    if args.distributed:
        search_option = 2
        logger.info(f"Rank {torch.distributed.get_rank()} | 开始计算距离")
    else:
        search_option = 3
        logger.info("单卡 | 开始计算距离")

    compute_jaccard_to_memmap(
        feat_bank_used,
        out_path=jaccard_path,
        k1=int(config.get("cluster_k1", 30)),
        k2=int(config.get("cluster_k2", 6)),
        use_float16=True,
        row_chunk=int(config.get("cluster_row_chunk", 1024)),
        search_option=search_option
    )

    # 释放 GPU bank
    del feat_bank, feat_bank_used
    torch.cuda.empty_cache()
    gc.collect()

    # 3) DBSCAN on memmap
    logger.info("开始基于 memmap 的 DBSCAN 聚类")
    if args.distributed:
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    image_pseudo_labels = None
    if (not args.distributed) or (args.distributed and rank == 0):
        image_pseudo_labels = dbscan_memmap(
            jaccard_path,
            eps=float(config.get("cluster_eps", 0.6)),
            min_samples=int(config.get("cluster_min_samples", 4))
        )
        logger.info("聚类完成（主节点）")
        if not os.path.exists(save_path):
            torch.save(image_pseudo_labels, save_path)
            logger.info(f"伪标签已保存至 {save_path}")

    # 4) broadcast labels
    if args.distributed:
        labels_list = [None]
        if rank == 0:
            labels_list[0] = image_pseudo_labels
        torch.distributed.broadcast_object_list(labels_list, src=0)
        image_pseudo_labels = labels_list[0]

    # 5) stats
    num_noise = int((image_pseudo_labels == -1).sum())
    unique_labels = set(image_pseudo_labels.tolist())
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    logger.info(f"Dataset 总长度: {len(train_loader.dataset)}, 输出伪标签长度: {len(image_pseudo_labels)}")
    logger.info(f"聚类数（不含 -1）: {num_clusters}")
    logger.info(f"-1 数量: {num_noise}\n")

    return image_pseudo_labels
