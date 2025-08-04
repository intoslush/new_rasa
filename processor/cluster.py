import os
import gc
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.cluster import DBSCAN  # 可选：如果内存允许、想用更快的 C 实现，可以开启下面的 sklearn 分支


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
    """
    计算 image features -> re-ranking jaccard (写 memmap) -> DBSCAN 聚类 -> 返回 pseudo labels（numpy array）。
    支持分布式：只有 rank 0 做保存和聚类，其他 rank 通过 broadcast_object_list 拿到结果。
    """
    device = torch.device("cuda")
    feature_size = 256  # cuhk 是 256，融合后可能更大
    max_size = len(train_loader.dataset)
    # 以 float16 在 GPU 上建 bank，训练后保存为 CPU 便于复用
    image_bank = torch.empty((max_size, feature_size), device=device, dtype=torch.float16)
    index = 0

    model = model.to(device)
    model.eval()
    if args.distributed and hasattr(model, "module"):
        model_no_ddp = model.module
    else:
        model_no_ddp = model

    test = False  # 保留原来测试加载开关
    save_path = "./logs/pseudo_labels.pt"
    save_feats = "./logs/feats.pt"

    logger.info("开始计算伪标签")

    # 1. load cached image features if exist
    if test and os.path.exists(save_feats):
        image_bank_cpu = torch.load(save_feats, weights_only=False)
        image_bank = image_bank_cpu.to(device=device, dtype=torch.float16)
        logger.info(f"检测到已保存的图像特征，加载 {save_feats}")
    else:
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                image1 = batch['image1'].to(device, non_blocking=True)
                batch_size = image1.size(0)

                # 提取视觉 embedding
                image_embeds = model_no_ddp.visual_encoder(image1)
                image_feat = F.normalize(model_no_ddp.vision_proj(image_embeds[:, 0, :]), dim=-1)  # cls token
                image_bank[index: index + batch_size] = image_feat.to(torch.float16)
                index += batch_size

        # 保存特征（仅 rank 0）
        if test and (not args.distributed or (args.distributed and torch.distributed.get_rank() == 0)):
            torch.save(image_bank.cpu(), save_feats)
            logger.info(f"图像特征已保存至 {save_feats}")

    # 2. 计算 re-ranking jaccard 并 dump 到 memmap
    if args.distributed:
        search_option = 2
        logger.info(f"Rank {torch.distributed.get_rank()} | 开始计算不同类之间的距离")
    else:
        search_option = 3
        logger.info("单卡 | 开始计算不同类之间的距离")

    jaccard_path = "./tmp/image_rerank_jaccard.memmap"
    # 保证目录
    os.makedirs(os.path.dirname(jaccard_path), exist_ok=True)

    # 注意：这里传 float32 使数值稳定，内部会转 dtype（use_float16 控制最终 jaccard 存储精度）
    compute_jaccard_to_memmap(
        image_bank.to(torch.float32),
        out_path=jaccard_path,
        k1=30,
        k2=6,
        use_float16=True,
        row_chunk=1024,
        search_option=search_option
    )

    # 释放 image_bank 以腾出 GPU
    del image_bank
    torch.cuda.empty_cache()
    gc.collect()

    # 3. 聚类：用 memmap 版本（不吃大量 RAM）
    logger.info("开始基于 memmap 的 DBSCAN 聚类")
    image_pseudo_labels = None
    if args.distributed:
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    if (not args.distributed) or (args.distributed and rank == 0):
        # 实际聚类只有主节点做
        image_pseudo_labels = dbscan_memmap(jaccard_path, eps=0.6, min_samples=4)
        logger.info("聚类完成（主节点）")

        # 保存伪标签
        if not os.path.exists(save_path):
            torch.save(image_pseudo_labels, save_path)
            logger.info(f"伪标签已保存至 {save_path}")

    # 4. 分布式广播 pseudo labels 给其他 rank（如果有）
    if args.distributed:
        # 使用 broadcast_object_list 来同步 numpy array
        labels_list = [None]
        if rank == 0:
            labels_list[0] = image_pseudo_labels
        torch.distributed.broadcast_object_list(labels_list, src=0)
        image_pseudo_labels = labels_list[0]

    # 5. 打印统计
    dataset_len = len(train_loader.dataset)
    num_noise = int((image_pseudo_labels == -1).sum())
    unique_labels = set(image_pseudo_labels.tolist())
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    logger.info(f"Dataset 总长度: {dataset_len}, 最终输出的伪标签长度: {len(image_pseudo_labels)}")
    logger.info(f"聚类数（不含 -1）: {num_clusters}")
    logger.info(f"-1 (未归入任何簇) 的数量: {num_noise}\n")

    return image_pseudo_labels
