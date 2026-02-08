# ──────────────────────────────────────────────────────────────────────────────
# Project layout (proposed)
# ──────────────────────────────────────────────────────────────────────────────
# my_model/
#   vit.py
#   xbert.py
#   albef/
#     __init__.py
#     model.py                  # ALBEF main module (forward intact)
#     mixins/
#       __init__.py
#       vision.py               # _build_vit
#       momentum.py             # copy_params, _momentum_update
#       queues.py               # _init_queues, _dequeue_and_enqueue, reset_queues, concat_all_gather
#       mlm.py                  # mask
#       saliency.py             # compute_cross_modal_saliency, build_curriculum_mask_probs
#       debug_utils.py          # debug_render_mask_diff
#       infmask.py              # compute_infmask_loss
# ──────────────────────────────────────────────────────────────────────────────
from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch import nn
from my_model.xbert import BertConfig, BertForMaskedLM
from .mixins import (
    VisionBuilderMixin,
    MomentumMixin,
    QueueMixin,
    MLMMixin,
    SaliencyMixin,
    DebugMaskMixin,
    concat_all_gather,
    SoftMaskITMMixin,
)
from .mixins.infmask import InfMaskMixin


class ALBEF(VisionBuilderMixin, MomentumMixin, QueueMixin, MLMMixin, SaliencyMixin, DebugMaskMixin, InfMaskMixin,SoftMaskITMMixin, nn.Module):
    def __init__(self, text_encoder=None, tokenizer=None, config: Dict[str, Any] = None):
        super().__init__()
        if config is None:
            config = {}

        self.tokenizer = tokenizer
        self.mlm_probability = config['mlm_probability']
        self.mrtd_mask_probability = config['mrtd_mask_probability']
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']
        image_res = config['image_res']

        # Vision Encoder
        self.visual_encoder = self._build_vit(image_res)
        self.vision_proj = nn.Linear(vision_width, embed_dim)

        # Text Encoder
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        self.text_width = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(self.text_width, embed_dim)

        # Heads
        self.itm_head = nn.Linear(self.text_width, 2)
        # ✱ 新增：ITM 动量头
        self.itm_head_m = nn.Linear(self.text_width, 2)

        # Temperature parameter
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])

        # Momentum models
        self.visual_encoder_m = self._build_vit(image_res)
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        self.text_proj_m = nn.Linear(self.text_width, embed_dim)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
            # ✱ 新增：ITM 线性头也做 EMA
            [self.itm_head, self.itm_head_m],
        ]
        self.copy_params()
        # Queues
        self._init_queues(embed_dim)

    def forward(self, batch, alpha, config, epoch):  # text2 是概率同一个 id 的其他图片描述, img1/img2 同一图不同增广
        loss_dict = {}
        image1 = batch['image1']
        image2 = batch['image2']
        text1 = self.tokenizer(batch['caption1'], padding='longest', max_length=config['max_words'], return_tensors="pt").to(image1.device)
        text2 = self.tokenizer(batch['caption2'], padding='longest', max_length=config['max_words'], return_tensors="pt").to(image1.device)
        text_atts = text2['attention_mask']
        idx = batch['person_id']
        replace = batch['replace_flag']
        idx = batch['pseudo_label']  # 覆盖为伪标签，保持原逻辑

        # extract image features
        image_embeds = self.visual_encoder(image1,register_blk=-1)
        # === 图像显著性：从 CLS→patch attention 中抽 ===
        attn = self.visual_encoder.blocks[-1].attn.get_attention_map()
        attn_mean = attn.mean(dim=1)              # [B, N, N]
        patch_scores = attn_mean[:, 0, 1:]        # [B, P] CLS→所有 patch 的权重

        B, P = patch_scores.shape
        min_v = patch_scores.view(B, -1).min(dim=-1, keepdim=True)[0]
        max_v = patch_scores.view(B, -1).max(dim=-1, keepdim=True)[0]
        patch_scores = (patch_scores - min_v) / (max_v - min_v + 1e-6)  # [B, P] ∈ [0,1]

        # 拼 CLS 的显著性（简单置 1），并 detach，防止梯度回流到 attn
        saliency_image = torch.cat(
            [
                torch.ones(B, 1, device=image1.device, dtype=patch_scores.dtype),
                patch_scores,
            ],
            dim=1,      # [B, 1+P]，和 image_embeds 的 token 数对齐
        ).detach()
        
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image1.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        # extract text features
        text_output = self.text_encoder.bert(text2['input_ids'], attention_mask=text2['attention_mask'], return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        
        # ===== Contrastive loss (ITC) =====
        image_embeds_m = None
        image_feat_m   = None
        text_embeds_m  = None
        text_feat_m    = None
        itm_pos_prob_m = None
        enable_cl_loss = bool(config.get('enable_cl_loss', True))
        use_momentum   = bool(config.get('use_momentum', True))
        use_queue      = bool(config.get('use_queue', True))

        # --- 新增：ITM->ITC 的配置 ---
        use_itm_affinity = bool(config.get('itc_use_itm_affinity', True))     # 是否把 teacher ITM affinity 融入 ITC targets
        itm_topk         = int(config.get('itc_itm_topk', 8))                 # top-k 候选（batch 内）
        itm_beta         = float(config.get('itc_itm_beta', 0.35))            # affinity 融合权重
        gate_tau_low     = float(config.get('itc_itm_gate_tau_low', 0.3))     # gating 低阈
        gate_tau_high    = float(config.get('itc_itm_gate_tau_high', 0.7))    # gating 高阈
        eps = 1e-8

        if enable_cl_loss:
            idx = idx.view(-1, 1)  # [B,1]
            bs = idx.size(0)

            # -------- 1) 构造 hard cluster targets（支持 queue） --------
            if use_queue:
                idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)  # [1, B+Q]
            else:
                idx_all = idx.t()  # [1, B]
            pos_idx = torch.eq(idx, idx_all).float()  # [B, B(+Q)]
            sim_targets_hard = pos_idx / (pos_idx.sum(1, keepdim=True) + eps)  # [B, B(+Q)]

            # -------- 2) teacher/momentum features & 蒸馏式对比 targets --------
            with torch.no_grad():
                if use_momentum:
                    self._momentum_update()

                    # momentum enc
                    image_embeds_m = self.visual_encoder_m(image2)
                    image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)

                    text_output_m = self.text_encoder_m.bert(
                        text2['input_ids'],
                        attention_mask=text2['attention_mask'],
                        return_dict=True,
                        mode='text'
                    )
                    text_embeds_m = text_output_m.last_hidden_state
                    text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)

                    # keys: momentum current + queue
                    if use_queue:
                        text_feat_all_k  = torch.cat([text_feat_m.t(),  self.text_queue.clone().detach()],  dim=1)  # [D, B+Q]
                        image_feat_all_k = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)  # [D, B+Q]
                    else:
                        text_feat_all_k  = text_feat_m.t()
                        image_feat_all_k = image_feat_m.t()

                    sim_i2t_m = image_feat_m @ text_feat_all_k / self.temp
                    sim_t2i_m = text_feat_m @ image_feat_all_k / self.temp

                    # 原 ALBEF 风格蒸馏 targets（cluster hard + teacher softmax）
                    sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets_hard
                    sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets_hard

                else:
                    # no momentum: 只用 hard cluster targets
                    sim_i2t_targets = sim_targets_hard
                    sim_t2i_targets = sim_targets_hard

            # -------- 3) 计算 online logits：query=online, key=momentum/queue（或 batch-only） --------
            if use_queue:
                if use_momentum:
                    # key 用 momentum（MoCo/ALBEF 常见做法）
                    text_feat_all  = torch.cat([text_feat_m.t(),  self.text_queue.clone().detach()],  dim=1)  # [D, B+Q]
                    image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)  # [D, B+Q]
                else:
                    # 没动量时，退化为 online(detach) + queue
                    text_feat_all  = torch.cat([text_feat.detach().t(),  self.text_queue.clone().detach()],  dim=1)
                    image_feat_all = torch.cat([image_feat.detach().t(), self.image_queue.clone().detach()], dim=1)
            else:
                # batch-only
                text_feat_all  = text_feat.t()
                image_feat_all = image_feat.t()

            sim_i2t = image_feat @ text_feat_all / self.temp     # [B, B(+Q)]
            sim_t2i = text_feat @ image_feat_all / self.temp     # [B, B(+Q)]

            # -------- 4) 新增：teacher ITM affinity (batch 内 top-k) + 正样本 gating --------
            # 只在 use_momentum=True 且 use_itm_affinity=True 时启用（否则退化为原逻辑）
            if use_momentum and use_itm_affinity:
                with torch.no_grad():
                    # 4.1 正配对 teacher ITM prob：p_m(i, t_i)
                    image_atts_m = torch.ones(image_embeds_m.size()[:-1], dtype=torch.long, device=image1.device)
                    out_pos_m = self.text_encoder_m.bert(
                        encoder_embeds=text_embeds_m,
                        attention_mask=text_atts,
                        encoder_hidden_states=image_embeds_m,
                        encoder_attention_mask=image_atts_m,
                        return_dict=True,
                        mode='fusion',
                    )
                    logits_pos_m = self.itm_head_m(out_pos_m.last_hidden_state[:, 0, :])  # [B,2]
                    prob_pos_m = F.softmax(logits_pos_m, dim=-1)[:, 1]                    # [B]
                    itm_pos_prob_m = prob_pos_m.detach()

                    # gating 权重 w_pos in [0,1]
                    w_pos = (itm_pos_prob_m - gate_tau_low) / max(gate_tau_high - gate_tau_low, 1e-6)
                    w_pos = w_pos.clamp_(0.0, 1.0)  # [B]

                    # 4.2 batch 内 Top-k 候选（用 momentum CL 相似度先筛候选）
                    sim_batch_i2t = (image_feat_m @ text_feat_m.t()) / self.temp  # [B,B]
                    k = min(itm_topk, bs)
                    topk_idx = torch.topk(sim_batch_i2t, k=k, dim=1).indices      # [B,k]

                    # 强制把对角线加进候选（避免极端情况下 diag 不在 topk）
                    diag = torch.arange(bs, device=image1.device).view(bs, 1)
                    topk_idx = torch.cat([topk_idx, diag], dim=1)                 # [B, k+1]

                    # flatten pairs
                    i_flat = torch.arange(bs, device=image1.device).view(bs, 1).expand_as(topk_idx).reshape(-1)  # [B*(k+1)]
                    j_flat = topk_idx.reshape(-1)

                    # teacher fusion on selected pairs
                    out_pairs_m = self.text_encoder_m.bert(
                        encoder_embeds=text_embeds_m[j_flat],
                        attention_mask=text_atts[j_flat],
                        encoder_hidden_states=image_embeds_m[i_flat],
                        encoder_attention_mask=image_atts_m[i_flat],
                        return_dict=True,
                        mode='fusion',
                    )
                    logits_pairs_m = self.itm_head_m(out_pairs_m.last_hidden_state[:, 0, :])  # [B*(k+1),2]
                    prob_pairs_m = F.softmax(logits_pairs_m, dim=-1)[:, 1]                    # [B*(k+1)]

                    # scatter -> P (batch 内稀疏 affinity)
                    P = torch.zeros(bs, bs, device=image1.device, dtype=prob_pairs_m.dtype)  # [B,B]
                    P[i_flat, j_flat] = prob_pairs_m

                    # row-normalize P
                    P = P / (P.sum(dim=1, keepdim=True) + eps)  # [B,B]

                    # pad 到 [B, B(+Q)]
                    if use_queue:
                        Q = self.text_queue.shape[1]
                        P_pad = torch.cat([P, torch.zeros(bs, Q, device=image1.device, dtype=P.dtype)], dim=1)
                    else:
                        P_pad = P

                # 4.3 用 affinity 融合 ITC targets（对 i2t 和 t2i 都做）
                # 注意：P_pad 是 image->text 的 affinity；t2i 用转置（batch 部分转置，queue 部分为 0）
                sim_i2t_targets = (1 - itm_beta) * sim_i2t_targets + itm_beta * P_pad
                sim_i2t_targets = sim_i2t_targets / (sim_i2t_targets.sum(dim=1, keepdim=True) + eps)

                if use_queue:
                    # t2i target 的 batch 部分用 P^T，queue 依旧 0
                    Q = self.image_queue.shape[1]
                    P_t = P.t()
                    P_t_pad = torch.cat([P_t, torch.zeros(bs, Q, device=image1.device, dtype=P_t.dtype)], dim=1)
                else:
                    P_t_pad = P.t()

                sim_t2i_targets = (1 - itm_beta) * sim_t2i_targets + itm_beta * P_t_pad
                sim_t2i_targets = sim_t2i_targets / (sim_t2i_targets.sum(dim=1, keepdim=True) + eps)

            else:
                # 没启用 affinity 时，gating 退化为全 1
                w_pos = torch.ones(bs, device=image1.device)

            # -------- 5) ITC loss（带 gating 加权） --------
            loss_i2t_vec = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1)  # [B]
            loss_t2i_vec = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1)  # [B]
            loss_cl_vec = 0.5 * (loss_i2t_vec + loss_t2i_vec)

            # 加权平均：sum(w*loss)/sum(w)
            loss_dict['loss_cl'] = (w_pos.detach() * loss_cl_vec).sum() / (w_pos.detach().sum() + eps)

            # -------- 6) 队列更新 --------
            if use_queue:
                if use_momentum:
                    self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)
                else:
                    self._dequeue_and_enqueue(image_feat.detach(), text_feat.detach(), idx)

        # ===== Saliency compute =====
        probability_matrix = None 
        saliency_compute_epoch = config.get('saliency_compute_epoch', 5)
        if epoch > saliency_compute_epoch :#and bool(config.get('enable_mlm_loss', False))
            with torch.no_grad():
                saliency = self.compute_cross_modal_groundedness(
                    text_ids=text1['input_ids'],
                    attention_mask=text1['attention_mask'],
                    image_embeds=image_embeds,
                    image_atts=image_atts,
                    saliency_image=None,  # 用已有的 patch 显著性
                    layers=int(config.get('saliency_layers', 3)),
                    use_entropy=bool(config.get("grounded_use_entropy", True)),
                    use_patch_saliency=bool(config.get("grounded_use_patch_saliency", False)),
                )

            probability_matrix = self.build_curriculum_mask_probs(
                saliency=saliency,
                attention_mask=text1['attention_mask'],
                input_ids=text1['input_ids'],
                base_prob=float(config.get('mlm_probability', self.mlm_probability)),
                focus_top_p=float(config.get('mlm_focus_top_p', 0.3)),
                p_strong=float(config.get('mlm_p_strong', 0.95)),
                p_min=float(config.get('mlm_prob_min', 0.05)),
                p_max=float(config.get('mlm_prob_max', 0.95)),
            )            
        else:
            probability_matrix = None
        
        # ===== Masked Language Modeling =====
        enable_mlm_loss = bool(config.get('enable_mlm_loss', False))
        enable_soft_label = bool(config.get('mlm_soft_label', False))
        image_embeds_m = None      # ensure defined if used below
        if enable_mlm_loss:
            input_ids = text1.input_ids.clone()
            labels = input_ids.clone()
            ids_before_debug = input_ids.clone()
            input_ids, labels = self.mask(
                input_ids,
                self.text_encoder.config.vocab_size,
                targets=labels,
                probability_matrix=probability_matrix  # 显著性引导的 mask 概率
            ) 
            if enable_soft_label:
                with torch.no_grad():
                    # ensure image_embeds_m is available if CL disabled
                    if image_embeds_m is None:
                        image_embeds_m = self.visual_encoder_m(image2) if enable_cl_loss else image_embeds
                    logits_m = self.text_encoder_m(
                        input_ids,
                        attention_mask=text1.attention_mask,
                        encoder_hidden_states=image_embeds_m,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                        return_logits=True,
                    )
                mlm_output = self.text_encoder(
                    input_ids,
                    attention_mask=text1.attention_mask,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                    labels=labels,
                    soft_labels=F.softmax(logits_m, dim=-1),
                    alpha=alpha,
                )
            else:
                mlm_output = self.text_encoder(
                    input_ids,
                    attention_mask=text1.attention_mask,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                    labels=labels,
                )
            debug_epoch = int(config.get('debug_mask_epoch', 6))
            if epoch > debug_epoch and bool(config.get("debug_log_saliency", True)):
                
                # 注意力方案的debug
                self.debug_render_mask_with_norms(
                    epoch=int(epoch),
                    step=int(batch.get("global_step", 0)),   # 没有就传 n_iter/全局计数
                    input_ids_before=ids_before_debug,
                    input_ids_after=input_ids,
                    targets=labels,
                    attention_mask=text1['attention_mask'],
                    probability_matrix=(probability_matrix if probability_matrix is not None else None),
                    saliency_norm=saliency,
                    raw_texts=batch.get('caption1', None),
                    out_path=str(config.get("debug_mask_file", "./mask_output2.txt")),
                    limit_per_epoch=int(config.get("debug_mask_limit_per_epoch", 30)),
                    sample_per_step=int(config.get("debug_sample_per_step", 2)),
                    topk_tokens=int(config.get("debug_topk_tokens", 8)),
                    step_prob=float(config.get("debug_step_prob", 0.15)),
                )
            loss_dict['loss_mlm'] = mlm_output.loss

        # ===== ITM (matched/unmatched) =====
        enable_itm_loss = bool(config.get('enable_itm_loss', False))
        enable_itm_softmask = bool(config.get('enable_itm_softmask', False))

        # --- 新增：ITM 蒸馏/降权配置 ---
        itm_use_distill     = bool(config.get('itm_use_distill', True))
        itm_distill_lambda  = float(config.get('itm_distill_lambda', 0.5))     # soft teacher vs hard CE
        itm_neg_filter_tau  = float(config.get('itm_neg_filter_tau', 0.6))     # teacher 认为 match 概率太高的“假负”降权起点
        itm_gate_tau_low    = float(config.get('itm_gate_tau_low', 0.3))       # 正样本 gating
        itm_gate_tau_high   = float(config.get('itm_gate_tau_high', 0.7))
        eps = 1e-8

        if enable_itm_loss:
            # --- 学生：正样本 (text2, image1) ---
            output_pos = self.text_encoder.bert(
                encoder_embeds=text_embeds,
                attention_mask=text_atts,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                mode='fusion',
                output_attentions=enable_itm_softmask,
                output_hidden_states=False,
            )

            with torch.no_grad():
                bs = image1.size(0)
                itm_neg_sampling = str(config.get("itm_neg_sampling", "cl")).lower()
                if itm_neg_sampling not in ("cl", "random"):
                    raise ValueError(f"config['itm_neg_sampling'] must be 'cl' or 'random', got {itm_neg_sampling}")

                idx_1d = idx.view(-1)  # [B]
                mask_same = torch.eq(idx_1d.view(bs, 1), idx_1d.view(1, bs))  # [B,B]

                if itm_neg_sampling == "cl":
                    if not enable_cl_loss:
                        sim_i2t_tmp = image_feat @ text_feat.t()
                        sim_t2i_tmp = text_feat @ image_feat.t()
                    else:
                        # 使用 ITC 已算的 sim_i2t/sim_t2i（若 use_queue=True，这里只取 batch 部分）
                        sim_i2t_tmp = sim_i2t[:, :bs]
                        sim_t2i_tmp = sim_t2i[:, :bs]

                    weights_i2t = F.softmax(sim_i2t_tmp, dim=1)
                    weights_t2i = F.softmax(sim_t2i_tmp, dim=1)
                    weights_i2t.masked_fill_(mask_same, 0.0)
                    weights_t2i.masked_fill_(mask_same, 0.0)

                    # 兜底：若某行全 0，退化为排除自身随机
                    if (weights_i2t.sum(dim=1) == 0).any():
                        w = (~torch.eye(bs, device=image1.device, dtype=torch.bool)).float()
                        weights_i2t = w / (w.sum(dim=1, keepdim=True) + eps)
                    else:
                        weights_i2t = weights_i2t / (weights_i2t.sum(dim=1, keepdim=True) + eps)

                    if (weights_t2i.sum(dim=1) == 0).any():
                        w = (~torch.eye(bs, device=image1.device, dtype=torch.bool)).float()
                        weights_t2i = w / (w.sum(dim=1, keepdim=True) + eps)
                    else:
                        weights_t2i = weights_t2i / (weights_t2i.sum(dim=1, keepdim=True) + eps)

                    image_neg_idx = torch.multinomial(weights_t2i, 1).squeeze(1)  # [B]
                    text_neg_idx  = torch.multinomial(weights_i2t, 1).squeeze(1)  # [B]
                else:
                    valid = (~mask_same).float()
                    row_sum = valid.sum(dim=1, keepdim=True)
                    if (row_sum == 0).any():
                        valid_fallback = (~torch.eye(bs, device=image1.device, dtype=torch.bool)).float()
                        valid = torch.where(row_sum > 0, valid, valid_fallback)
                        row_sum = valid.sum(dim=1, keepdim=True)
                    probs = valid / (row_sum + eps)
                    image_neg_idx = torch.multinomial(probs, 1).squeeze(1)
                    text_neg_idx  = torch.multinomial(probs, 1).squeeze(1)

            # --- 构造负样本对 ---
            image_embeds_neg = image_embeds[image_neg_idx]
            text_embeds_neg  = text_embeds[text_neg_idx]
            text_atts_neg    = text_atts[text_neg_idx]

            # pairs: (text_pos, image_neg) + (text_neg, image_pos)
            text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
            text_atts_all   = torch.cat([text_atts,   text_atts_neg],  dim=0)
            image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
            image_atts_all   = torch.cat([image_atts,       image_atts],   dim=0)

            output_neg_cross = self.text_encoder.bert(
                encoder_embeds=text_embeds_all,
                attention_mask=text_atts_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                return_dict=True,
                mode='fusion',
            )

            # student logits
            vl_embeddings = torch.cat([
                output_pos.last_hidden_state[:, 0, :],          # [B, D]
                output_neg_cross.last_hidden_state[:, 0, :],    # [2B, D]
            ], dim=0)                                          # [3B, D]
            vl_output = self.itm_head(vl_embeddings)           # [3B, 2]

            # hard labels
            itm_labels = torch.cat([
                torch.ones(bs, dtype=torch.long),
                torch.zeros(2 * bs, dtype=torch.long)
            ], dim=0).to(image1.device)

            # -------- 新增：teacher soft labels + 假负降权 --------
            if itm_use_distill:
                with torch.no_grad():
                    # 确保 teacher enc 已计算
                    if image_embeds_m is None or text_embeds_m is None:
                        self._momentum_update()
                        image_embeds_m = self.visual_encoder_m(image2)
                        text_output_m = self.text_encoder_m.bert(
                            text2['input_ids'],
                            attention_mask=text2['attention_mask'],
                            return_dict=True,
                            mode='text'
                        )
                        text_embeds_m = text_output_m.last_hidden_state

                    image_atts_m = torch.ones(image_embeds_m.size()[:-1], dtype=torch.long, device=image1.device)

                    # teacher pairs 对齐 student 的 3B 顺序：
                    # pos: (t_i, img_i)
                    # neg1: (t_i, img_neg_i)
                    # neg2: (t_neg_i, img_i)
                    t_pos_m   = text_embeds_m
                    a_pos_m   = text_atts
                    img_pos_m = image_embeds_m

                    t_neg1_m   = text_embeds_m
                    a_neg1_m   = text_atts
                    img_neg1_m = image_embeds_m[image_neg_idx]

                    t_neg2_m   = text_embeds_m[text_neg_idx]
                    a_neg2_m   = text_atts[text_neg_idx]
                    img_neg2_m = image_embeds_m

                    t_teacher   = torch.cat([t_pos_m,  t_neg1_m,  t_neg2_m],  dim=0)
                    a_teacher   = torch.cat([a_pos_m,  a_neg1_m,  a_neg2_m],  dim=0)
                    img_teacher = torch.cat([img_pos_m, img_neg1_m, img_neg2_m], dim=0)
                    img_atts_teacher = torch.cat([image_atts_m, image_atts_m[image_neg_idx], image_atts_m], dim=0)

                    out_teacher = self.text_encoder_m.bert(
                        encoder_embeds=t_teacher,
                        attention_mask=a_teacher,
                        encoder_hidden_states=img_teacher,
                        encoder_attention_mask=img_atts_teacher,
                        return_dict=True,
                        mode='fusion',
                    )
                    logits_teacher = self.itm_head_m(out_teacher.last_hidden_state[:, 0, :])  # [3B,2]
                    probs_teacher  = F.softmax(logits_teacher, dim=-1).detach()               # [3B,2]
                    p_match_teacher = probs_teacher[:, 1]                                     # [3B]

                    # 正样本 gating
                    w_pos = (p_match_teacher[:bs] - itm_gate_tau_low) / max(itm_gate_tau_high - itm_gate_tau_low, 1e-6)
                    w_pos = w_pos.clamp_(0.0, 1.0)  # [B]

                    # 负样本“疑似假负”降权：p_match 越大权重越小
                    p_neg = p_match_teacher[bs:]  # [2B]
                    w_neg = 1.0 - ((p_neg - itm_neg_filter_tau) / max(1.0 - itm_neg_filter_tau, 1e-6)).clamp(0.0, 1.0)

                    weights = torch.cat([w_pos, w_neg], dim=0).detach()  # [3B]

                # hard CE（逐样本）
                hard_ce = F.cross_entropy(vl_output, itm_labels, reduction='none')  # [3B]
                # soft CE（teacher probs）
                soft_ce = -torch.sum(probs_teacher * F.log_softmax(vl_output, dim=-1), dim=-1)  # [3B]

                loss_vec = (1.0 - itm_distill_lambda) * hard_ce + itm_distill_lambda * soft_ce
                loss_itm = (weights * loss_vec).sum() / (weights.sum() + eps)

            else:
                # 纯硬标签 CE（原逻辑）
                loss_itm = F.cross_entropy(vl_output, itm_labels)

            loss_dict['loss_itm'] = loss_itm

            # 正样本 logits（保留你原来的输出接口）
            vl_output_pos_full = vl_output[:bs].detach()


        return loss_dict