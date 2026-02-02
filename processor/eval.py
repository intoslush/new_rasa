import os
import time
import datetime
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import utils.optimizer as utils
from typing import Optional



def _ensure_tmp_dir(base_dir: str) -> str:
    tmp_dir = os.path.join(base_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def _create_memmap(path: str, shape, dtype):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return np.memmap(path, mode="w+", dtype=dtype, shape=shape)


@torch.no_grad()
def itm_eval(scores_t2i, img2person, txt2person, eval_mAP=True):
    """
    scores_t2i: Tensor [num_text, num_image], larger is better (only ranking matters)
    img2person: list/1d array length num_image
    txt2person: list/1d array length num_text
    """
    device = scores_t2i.device
    img2person = torch.as_tensor(img2person, device=device)
    txt2person = torch.as_tensor(txt2person, device=device)

    # sort gallery per query
    index = torch.argsort(scores_t2i, dim=-1, descending=True)   # [T, I]
    pred_person = img2person[index]                              # [T, I]
    matches = (txt2person.view(-1, 1).eq(pred_person)).long()    # [T, I]

    def acc_k(matches, k=1):
        k = min(k, matches.size(1))
        hits = matches[:, :k].sum(dim=-1) > 0
        return 100.0 * hits.float().mean()

    # Recall@K
    ir1 = acc_k(matches, k=1).item()
    ir5 = acc_k(matches, k=5).item()
    ir10 = acc_k(matches, k=10).item()

    # positives per query
    real_num = matches.sum(dim=-1)          # [T]
    valid = real_num > 0                   # queries with at least one positive

    # mAP
    tmp_cmc = matches.cumsum(dim=-1).float()
    order = torch.arange(1, matches.size(1) + 1, device=device).view(1, -1).float()
    tmp_cmc = (tmp_cmc / order) * matches.float()

    AP = torch.zeros(matches.size(0), device=device)
    if valid.any():
        AP[valid] = tmp_cmc[valid].sum(dim=-1) / real_num[valid].float()
        mAP = (AP[valid].mean() * 100.0).item()
    else:
        mAP = 0.0

    # mINP: INP(q) = (#positives) / (rank of last positive), 1-based rank
    ranks = torch.arange(1, matches.size(1) + 1, device=device).view(1, -1)
    last_pos_rank = (matches * ranks).max(dim=-1).values         # 0 if no positive
    INP = torch.zeros(matches.size(0), device=device)
    if valid.any():
        INP[valid] = real_num[valid].float() / last_pos_rank[valid].float()
        mINP = (INP[valid].mean() * 100.0).item()
    else:
        mINP = 0.0

    return {
        'r1': ir1,
        'r5': ir5,
        'r10': ir10,
        'mAP': mAP,
        'mINP': mINP,
    }


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, args):
    """
    改造点：
    - num_img > 4000 时：
      - image_feats: memmap(fp16) 落盘
      - sims: 按 query 行计算，避免 sims_matrix 常驻显存
      - score_matrix_t2i: memmap(fp32) 落盘，distributed 下 chunk all_reduce 后写回
    """
    model.eval()
    logger = logging.getLogger(args.name)
    header = 'Evaluation:'
    logger.info(f"{header} Start")
    start_time = time.time()

    # -------------------------
    # Basic sizes / distributed
    # -------------------------
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    distributed = bool(getattr(args, "distributed", False))

    # -------------------------
    # Text features (same as before)
    # -------------------------
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256

    logger.info(f"{header} Extract text feats (T={num_text})")
    text_feats, text_embeds, text_atts = [], [], []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=config['max_words'],
            return_tensors="pt"
        ).to(device)

        text_output = model.text_encoder.bert(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            mode='text'
        )
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))

        text_embeds.append(text_embed)                 # [bs, D] on GPU
        text_feats.append(text_feat)                   # [bs, L, D] on GPU
        text_atts.append(text_input.attention_mask)    # [bs, L] on GPU

    text_embeds = torch.cat(text_embeds, dim=0)   # [T, D] on GPU
    text_feats  = torch.cat(text_feats, dim=0)    # [T, L, D] on GPU
    text_atts   = torch.cat(text_atts, dim=0)     # [T, L] on GPU

    # -------------------------
    # Image features (memmap if large)
    # -------------------------
    num_img = len(data_loader.dataset)
    k = min(int(config['k_test']), num_img)

    use_memmap = num_img > 4000
    base_dir = os.getcwd()
    tmp_dir = _ensure_tmp_dir(base_dir)

    img_feat_mmap: Optional[np.memmap] = None
    score_mmap: Optional[np.memmap] = None

    # use rank-specific files to avoid collisions
    img_feat_path = os.path.join(tmp_dir, f"image_feats.fp16.rank{rank}.mmap")
    score_path    = os.path.join(tmp_dir, f"score_matrix.fp32.rank{rank}.mmap")

    logger.info(f"{header} Extract image feats (I={num_img}, use_memmap={use_memmap})")

    image_embeds = []
    image_feats_list = []  # only used when not memmap
    img_ptr = 0

    for batch in data_loader:
        image = batch["image"].to(device)

        # vision encoder can be fp16 safely
        with torch.cuda.amp.autocast(enabled=True):
            image_feat = model.visual_encoder(image)  # [B, L, D]
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        bsz, L, D = image_feat.shape

        if img_feat_mmap is None and use_memmap:
            img_feat_mmap = _create_memmap(
                img_feat_path,
                shape=(num_img, L, D),
                dtype=np.float16
            )

        if use_memmap:
            # fp16 on disk to reduce disk footprint and IO
            img_feat_mmap[img_ptr:img_ptr + bsz] = image_feat.detach().cpu().half().numpy()
        else:
            image_feats_list.append(image_feat.detach().cpu())

        image_embeds.append(image_embed)
        img_ptr += bsz

    image_embeds = torch.cat(image_embeds, dim=0)  # [I, D] on GPU

    if not use_memmap:
        image_feats_cpu = torch.cat(image_feats_list, dim=0)  # [I, L, D] on CPU
    else:
        # flush to ensure data is written
        assert img_feat_mmap is not None
        img_feat_mmap.flush()

    # -------------------------
    # Score matrix (memmap if large)
    # -------------------------
    if use_memmap:
        score_mmap = _create_memmap(
            score_path,
            shape=(num_text, num_img),
            dtype=np.float32
        )
        # initialize to -1e9
        score_mmap[:] = -1e9
        score_mmap.flush()
        # score on CPU memmap; we will write rows into it during loop (from GPU computations)
    else:
        score_matrix_t2i = torch.full((num_text, num_img), -1e9, device=device)

    # rank_scores: strictly decreasing scores -> argsort(desc) recovers final_order
    rank_scores = torch.arange(num_img, 0, -1, device=device).float()  # [I]

    # -------------------------
    # Distributed split by query
    # -------------------------
    step = num_text // num_tasks + 1
    start = rank * step
    end = min(num_text, start + step)

    logger.info(f"{header} Rerank queries in range [{start}, {end}) k={k}")

    for i in range(start, end):
        if (i - start) % 1000 == 0:
            logger.info(f"{header} [{i - start}/{end - start}]")

        # compute sims for this query only (no full sims_matrix)
        sims = text_embeds[i] @ image_embeds.t()  # [I] on GPU

        # 1) Full ITC order
        itc_order = torch.argsort(sims, descending=True)  # [I]
        topk_idx_gpu = itc_order[:k]                      # [k]

        # 2) Load topk image feats (from memmap or RAM)
        if use_memmap:
            assert img_feat_mmap is not None
            topk_idx_cpu_np = topk_idx_gpu.detach().cpu().numpy()
            feats_np = img_feat_mmap[topk_idx_cpu_np]  # [k, L, D] float16
            encoder_output = torch.from_numpy(feats_np).to(device)  # fp16 on GPU
        else:
            topk_idx_cpu = topk_idx_gpu.detach().cpu()
            encoder_output = image_feats_cpu[topk_idx_cpu].to(device)  # fp32 on GPU

        encoder_att = torch.ones(
            encoder_output.size()[:-1],
            dtype=torch.long,
            device=device
        )

        # 3) ITM fusion on topk (autocast fp16); head in fp32
        with torch.cuda.amp.autocast(enabled=True):
            output = model.text_encoder.bert(
                encoder_embeds=text_feats[i].repeat(k, 1, 1),
                attention_mask=text_atts[i].repeat(k, 1),
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
                mode='fusion'
            )

        itm_score = model.itm_head(output.last_hidden_state[:, 0, :].float())[:, 1]  # [k], fp32

        # 4) Rerank within topk by ITM score
        rerank_order_in_topk = torch.argsort(itm_score, descending=True)     # [k]
        reranked_topk = topk_idx_gpu[rerank_order_in_topk]                   # [k]

        # 5) Final order = reranked topk + remaining ITC order
        final_order = torch.cat([reranked_topk, itc_order[k:]], dim=0)       # [I]

        # 6) Encode final order as scores
        row_scores = torch.full((num_img,), -1e9, device=device)
        row_scores[final_order] = rank_scores

        if use_memmap:
            assert score_mmap is not None
            # write this row to memmap (CPU)
            score_mmap[i, :] = row_scores.detach().cpu().numpy()
        else:
            score_matrix_t2i[i, :] = row_scores

    if use_memmap and score_mmap is not None:
        score_mmap.flush()

    # -------------------------
    # Sync across processes if distributed
    # - if score is memmap: chunk all_reduce -> write back
    # - else: original all_reduce on full tensor
    # -------------------------
    if distributed:
        dist.barrier()

        if use_memmap:
            assert score_mmap is not None
            # chunk by rows to keep GPU peak small
            # choose chunk rows so that chunk tensor is not too large
            # rows_per_chunk can be tuned; 256 is conservative for 4000^2 scale
            rows_per_chunk = 256
            for r0 in range(0, num_text, rows_per_chunk):
                r1 = min(num_text, r0 + rows_per_chunk)
                chunk = torch.from_numpy(np.array(score_mmap[r0:r1, :], copy=False)).to(device)
                dist.all_reduce(chunk, op=dist.ReduceOp.MAX)
                # write reduced chunk back to memmap
                score_mmap[r0:r1, :] = chunk.detach().cpu().numpy()
            score_mmap.flush()
        else:
            dist.all_reduce(score_matrix_t2i, op=dist.ReduceOp.MAX)

        dist.barrier()

    # -------------------------
    # Finalize / return
    # -------------------------
    total_time = time.time() - start_time
    logger.info(f'{header} time {str(datetime.timedelta(seconds=int(total_time)))}')

    if use_memmap:
        assert score_mmap is not None
        # return CPU tensor backed by a normal ndarray (detach from file so we can cleanup)
        score_cpu = torch.from_numpy(np.array(score_mmap, copy=True))
    else:
        score_cpu = score_matrix_t2i.detach().cpu()

    # cleanup tmp files (each rank removes its own files)
    if use_memmap:
        try:
            # make sure everyone finished reading/writing
            if distributed:
                dist.barrier()
            if os.path.exists(img_feat_path):
                os.remove(img_feat_path)
            if os.path.exists(score_path):
                os.remove(score_path)
            # rank0尝试清理空目录
            if rank == 0:
                # only remove tmp_dir if empty
                if os.path.isdir(tmp_dir) and len(os.listdir(tmp_dir)) == 0:
                    os.rmdir(tmp_dir)
        except Exception:
            # 不影响评估结果
            pass

    return score_cpu
