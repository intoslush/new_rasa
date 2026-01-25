import torch
import utils.optimizer as utils
import time
import torch.nn.functional as F
import datetime
import torch.distributed as dist
import  logging

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
    model.eval()
    logger = logging.getLogger(args.name)
    header = 'Evaluation:'
    logger.info(f"{header} Start")
    start_time = time.time()

    # -------- Text features --------
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256

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

        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)   # [T, D]
    text_feats  = torch.cat(text_feats, dim=0)    # [T, L, D]
    text_atts   = torch.cat(text_atts, dim=0)     # [T, L]

    # -------- Image features --------
    image_feats, image_embeds = [], []
    for batch in data_loader:
        image = batch["image"].to(device)

        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat.cpu())      # keep feats on CPU to save GPU mem
        image_embeds.append(image_embed)

    image_feats  = torch.cat(image_feats, dim=0)  # [I, L, D] on CPU
    image_embeds = torch.cat(image_embeds, dim=0) # [I, D] on GPU

    # -------- ITC similarity --------
    sims_matrix = text_embeds @ image_embeds.t()  # [T, I] on GPU
    num_img = sims_matrix.size(1)

    # This score matrix will encode FINAL ranking (topk reranked by ITM, rest by ITC).
    # Use a very small default so MAX-reduce works in distributed setting.
    score_matrix_t2i = torch.full((num_text, num_img), -1e9, device=device)

    # rank_scores: unique strictly decreasing scores -> argsort(desc) recovers final_order
    rank_scores = torch.arange(num_img, 0, -1, device=device).float()  # [I]

    # -------- Distributed split by query --------
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    k = min(int(config['k_test']), num_img)

    for i in range(start, end):
        if (i - start) % 1000 == 0:
            logger.info(f"{header} [{i - start}/{end - start}]")

        sims = sims_matrix[i]  # [I]

        # 1) Full ITC order (so tail keeps ITC ordering)
        itc_order = torch.argsort(sims, descending=True)  # [I] on GPU

        topk_idx_gpu = itc_order[:k]                      # [k] on GPU
        topk_idx_cpu = topk_idx_gpu.to(image_feats.device)

        # 2) Compute ITM only on topk candidates
        encoder_output = image_feats[topk_idx_cpu]        # [k, L, D] on CPU
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long, device=device)

        output = model.text_encoder.bert(
            encoder_embeds=text_feats[i].repeat(k, 1, 1),
            attention_mask=text_atts[i].repeat(k, 1),
            encoder_hidden_states=encoder_output.to(device),
            encoder_attention_mask=encoder_att,
            return_dict=True,
            mode='fusion'
        )

        itm_score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]  # [k]

        # 3) Rerank within topk by ITM score
        rerank_order_in_topk = torch.argsort(itm_score, descending=True)     # [k]
        reranked_topk = topk_idx_gpu[rerank_order_in_topk]                   # [k]

        # 4) Final order = reranked topk + remaining ITC order
        final_order = torch.cat([reranked_topk, itc_order[k:]], dim=0)       # [I]

        # 5) Encode final order as scores so downstream argsort recovers it
        score_matrix_t2i[i, final_order] = rank_scores

    # -------- Sync across processes if distributed --------
    if getattr(args, "distributed", False):
        dist.barrier()
        dist.all_reduce(score_matrix_t2i, op=dist.ReduceOp.MAX)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Evaluation time {}'.format(total_time_str))

    return score_matrix_t2i.cpu()