import os
import torch
import torch.distributed as dist
import numpy as np
from typing import Optional
from collections import defaultdict

try:
    from .cluster import cluster_begin_epoch
except Exception:
    # 兼容直接放在同级目录的情形
    from .cluster import cluster_begin_epoch  # type: ignore


def compute_pseudo_stats(pseudo_labels: np.ndarray, true_person_ids: np.ndarray):
    """
    pseudo_labels: [N]，聚类结果，含 -1（噪声）
    true_person_ids: [N]，真实 person 索引
    返回：覆盖率、正确数、纯度等
    """
    assert len(pseudo_labels) == len(true_person_ids), (
        f"长度不一致: pseudo={len(pseudo_labels)}, gt={len(true_person_ids)}"
    )

    mask = (pseudo_labels != -1)
    num_total = len(pseudo_labels)
    num_assigned = int(mask.sum())
    num_noise = num_total - num_assigned

    # 按簇聚合 -> 多数表决
    correct = 0
    if num_assigned > 0:
        cluster2idx = defaultdict(list)
        for i, cid in enumerate(pseudo_labels):
            if cid != -1:
                cluster2idx[cid].append(i)
        for _, idxs in cluster2idx.items():
            gt = true_person_ids[idxs]
            values, counts = np.unique(gt, return_counts=True)
            correct += int(counts.max())

    return {
        "num_total": num_total,
        "num_assigned": num_assigned,
        "num_noise": num_noise,
        "coverage": num_assigned / num_total if num_total else 0.0,
        "correct": correct,
        "purity_non_noise": correct / num_assigned if num_assigned else 0.0,
        "purity_overall": correct / num_total if num_total else 0.0,
    }


def generate_and_broadcast_pseudo_labels(
    *,
    epoch: int,
    device: torch.device,
    is_main: bool,
    is_distributed: bool,
    rank: int,
    cluster_loader,
    model,
    args,
    config,
    logger,
    tb_writer=None,
    enable_nmi_ari: bool = True,
    cluster_until_epoch: int = 40,
):
    """
    仅在 epoch < cluster_until_epoch 时进行聚类，并将结果广播到所有 rank。
    返回：torch.LongTensor[dataset_size] (设备 device)
    """
    dataset_size = len(cluster_loader.dataset)
    if epoch >= cluster_until_epoch:
        # 直接创建占位（-1）张量
        pseudo_labels = torch.full((dataset_size,), -1, dtype=torch.long, device=device)
        return pseudo_labels

    cluster_loader.dataset.mode = 'cluster'

    if is_main:
        with torch.no_grad():
            image_pseudo_labels_np = cluster_begin_epoch(
                cluster_loader, model, args, config, None, logger
            )
            image_num_cluster = len(set(image_pseudo_labels_np)) - (
                1 if -1 in image_pseudo_labels_np else 0
            )
            logger.info(
                "==> [epoch %d] clusters=%d, total=%d",
                epoch, image_num_cluster, len(image_pseudo_labels_np)
            )

            # 统计信息
            image_pseudo_labels = torch.tensor(
                image_pseudo_labels_np, dtype=torch.long
            ).to(device, non_blocking=True)
            noise_frac = (image_pseudo_labels == -1).float().mean().item()

            # 可选：伪标签质量监控
            if hasattr(cluster_loader.dataset, 'pairs'):
                gt_persons = np.array([p for _, _, p in cluster_loader.dataset.pairs], dtype=np.int64)
                pseudo_np = image_pseudo_labels.detach().cpu().numpy()
                stats = compute_pseudo_stats(pseudo_np, gt_persons)

                logger.info(
                    "[ClusterEval][epoch %d] assigned=%d/%d (coverage=%.4f), noise=%d, "
                    "correct=%d, purity_non_noise=%.4f, overall=%.4f",
                    epoch, stats["num_assigned"], stats["num_total"], stats["coverage"],
                    stats["num_noise"], stats["correct"], stats["purity_non_noise"], stats["purity_overall"],
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("Cluster/coverage", stats["coverage"], epoch)
                    tb_writer.add_scalar("Cluster/correct_count", stats["correct"], epoch)
                    tb_writer.add_scalar("Cluster/purity_non_noise", stats["purity_non_noise"], epoch)
                    tb_writer.add_scalar("Cluster/purity_overall", stats["purity_overall"], epoch)
                    tb_writer.add_scalar("Cluster/num_clusters", image_num_cluster, epoch)
                    tb_writer.add_scalar("Cluster/noise_frac", noise_frac, epoch)

                if enable_nmi_ari:
                    try:
                        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
                        m = (pseudo_np != -1)
                        if m.any():
                            nmi = normalized_mutual_info_score(gt_persons[m], pseudo_np[m])
                            ari = adjusted_rand_score(gt_persons[m], pseudo_np[m])
                            logger.info("[ClusterEval][epoch %d] NMI=%.4f, ARI=%.4f", epoch, nmi, ari)
                            if tb_writer is not None:
                                tb_writer.add_scalar("Cluster/NMI_non_noise", nmi, epoch)
                                tb_writer.add_scalar("Cluster/ARI_non_noise", ari, epoch)
                    except Exception as e:
                        logger.warning(f"计算 NMI/ARI 失败：{e}")

            # 提前释放 numpy 内存
            del image_pseudo_labels_np
    else:
        print(f"[Rank {rank}] 等待主进程生成伪标签")
        image_pseudo_labels = torch.empty(dataset_size, dtype=torch.long, device=device)

    if is_distributed:
        dist.broadcast(image_pseudo_labels, src=0)
        dist.barrier()

    # 还原为 train 模式交由上层设置
    cluster_loader.dataset.mode = 'cluster'

    return image_pseudo_labels
