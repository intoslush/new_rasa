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
        # --- 新增：queue 的置信度与 image_id 队列 ---
        self.register_buffer("conf_queue", torch.ones(1, self.queue_size))
        self.register_buffer("imgid_queue", torch.full((1, self.queue_size), -1, dtype=torch.long))

    def forward(self, batch, alpha, config, epoch):  # text2 是概率同一个 id 的其他图片描述, img1/img2 同一图不同增广
        loss_dict = {}
        image1 = batch['image1']
        image2 = batch['image2']
        text1 = self.tokenizer(batch['caption1'], padding='longest', max_length=config['max_words'], return_tensors="pt").to(image1.device)
        text2 = self.tokenizer(batch['caption2'], padding='longest', max_length=config['max_words'], return_tensors="pt").to(image1.device)
        text_atts = text2['attention_mask']
        idx = batch['person_id']
        # --- labels ---
        pseudo_idx = batch["pseudo_label"].view(-1, 1)  # weak label (noisy)
        use_image_id_pos = bool(config.get("use_image_id_pos", True))
        img_idx = batch["image_id"].view(-1, 1) if use_image_id_pos and ("image_id" in batch) else None
        

        # extract image features
        image_embeds = self.visual_encoder(image1,register_blk=-1)
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image1.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        # extract text features
        text_output = self.text_encoder.bert(text2['input_ids'], attention_mask=text2['attention_mask'], return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        
        ##################################
        itm_prob_m = None
        p_diag_m = None

        enable_itm_gate = bool(config.get("enable_itm_gate", True))
        gate_start = int(config.get("itm_gate_start_epoch", 10))

        # 只有在 epoch>=gate_start 且确实需要时才算（CL 或 ITM 开启）
        need_gate = enable_itm_gate and (epoch >= gate_start) and (bool(config.get("enable_cl_loss", True)) or bool(config.get("enable_itm_loss", False)))

        if need_gate:
            with torch.no_grad():
                # teacher 特征（用 image1 + text2 对齐当前 batch 的真实配对）
                self._momentum_update()

                image_embeds_m_gate = self.visual_encoder_m(image1)
                image_atts_gate = torch.ones(image_embeds_m_gate.size()[:-1], dtype=torch.long, device=image1.device)

                itm_prob_m = self.compute_itm_prob_matrix_m(
                    image_embeds_m=image_embeds_m_gate,
                    image_atts=image_atts_gate,
                    text_ids=text2["input_ids"],
                    text_atts=text2["attention_mask"],
                    chunk_size=int(config.get("itm_gate_chunk", 128)),
                )  # [B,B]
                p_diag_m = itm_prob_m.diag().clamp(0.0, 1.0)  # [B]
        
        # ===== Contrastive loss =====
        enable_cl_loss = bool(config.get('enable_cl_loss', True))
        use_momentum   = bool(config.get('use_momentum', True))   # 是否用动量分支
        use_queue      = bool(config.get('use_queue', True))      # <-- 新增：是否使用队列做对比
        

        if enable_cl_loss:
            bs = image1.size(0)

            # --- 1) 候选集合：batch-only 或 batch+queue ---
            if use_queue:
                pseudo_all = torch.cat([pseudo_idx.t(), self.idx_queue.clone().detach()], dim=1)  # [1, B+Q]
                if img_idx is not None:
                    img_all = torch.cat([img_idx.t(), self.imgid_queue.clone().detach()], dim=1)  # [1, B+Q]
                else:
                    img_all = None
            else:
                pseudo_all = pseudo_idx.t()  # [1,B]
                img_all = img_idx.t() if img_idx is not None else None

            # --- 2) 先构造 pseudo positives ---
            pos_pseudo = torch.eq(pseudo_idx, pseudo_all).float()  # [B, B(+Q)]

            # batch 内 pseudo positives：用 ITM gate 过滤/软权重
            if (itm_prob_m is not None):
                margin_pos = float(config.get("itm_gate_margin_pos", 0.10))
                tau_pos = float(config.get("itm_gate_tau_pos", 0.30))
                thr = torch.clamp(p_diag_m - margin_pos, min=tau_pos).view(bs, 1)  # [B,1]

                gate_hard = (itm_prob_m >= thr).float()  # [B,B]
                same_pseudo_batch = torch.eq(pseudo_idx.view(bs,1), pseudo_idx.view(1,bs)).float()
                gate_pos_batch = gate_hard * same_pseudo_batch  # [B,B]

                pos_pseudo[:, :bs] = pos_pseudo[:, :bs] * gate_pos_batch

            # --- 3) image_id 强正样：无条件加入（不需要 gate） ---
            if img_all is not None:
                pos_img = torch.eq(img_idx, img_all).float()
                pos_idx = torch.clamp(pos_pseudo + pos_img, 0.0, 1.0)
            else:
                pos_idx = pos_pseudo

            # --- 4) queue positives 用 conf 加权（3.1） ---
            if use_queue and bool(config.get("queue_use_conf", True)):
                conf_min = float(config.get("conf_min", 0.05))
                conf_q = self.conf_queue.clone().detach().clamp(min=conf_min)  # [1,Q]
                pos_idx[:, bs:] = pos_idx[:, bs:] * conf_q  # [B,Q] broadcast

                # 同时对 anchor 侧也用对角置信度缩放（可选但推荐）
                if p_diag_m is not None:
                    conf_a = p_diag_m.clamp(min=conf_min).view(bs, 1)  # [B,1]
                    pos_idx = pos_idx * conf_a

            sim_targets = pos_idx / (pos_idx.sum(1, keepdim=True) + 1e-8)

            # --- 5) 动量蒸馏 targets（保持你原始 ALBEF 逻辑） ---
            with torch.no_grad():
                if use_momentum:
                    # 你原本这里用 image2 做 m 分支对比特征，我保留
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image2)
                    image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)

                    text_output_m = self.text_encoder_m.bert(
                        text2["input_ids"],
                        attention_mask=text2["attention_mask"],
                        return_dict=True,
                        mode="text"
                    )
                    text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)

                    if use_queue:
                        image_feat_all_m = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)  # [D,B+Q]
                        text_feat_all_m  = torch.cat([text_feat_m.t(),  self.text_queue.clone().detach()],  dim=1)  # [D,B+Q]
                    else:
                        image_feat_all_m = image_feat_m.t()
                        text_feat_all_m  = text_feat_m.t()

                    sim_i2t_m = image_feat_m @ text_feat_all_m / self.temp
                    sim_t2i_m = text_feat_m @ image_feat_all_m / self.temp

                    sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                    sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
                else:
                    sim_i2t_targets = sim_targets
                    sim_t2i_targets = sim_targets

           # logits key set 对齐
            if use_momentum:
                if use_queue:
                    text_feat_all  = torch.cat([text_feat_m.t(), self.text_queue.detach()], dim=1)
                    image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.detach()], dim=1)
                else:
                    text_feat_all  = text_feat_m.t()
                    image_feat_all = image_feat_m.t()
            else:
                if use_queue:
                    text_feat_all  = torch.cat([text_feat.t(), self.text_queue.detach()], dim=1)
                    image_feat_all = torch.cat([image_feat.t(), self.image_queue.detach()], dim=1)
                else:
                    text_feat_all  = text_feat.t()
                    image_feat_all = image_feat.t()

            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ image_feat_all / self.temp

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            loss_dict["loss_cl"] = (loss_i2t + loss_t2i) / 2

            # --- 7) 入队：同时入 conf（对角 p_diag_m）和 image_id ---
            if use_queue:
                if (p_diag_m is not None):
                    conf_batch = p_diag_m.detach()
                else:
                    conf_batch = torch.ones(bs, device=image1.device, dtype=torch.float32)

                if use_momentum:
                    self._dequeue_and_enqueue(image_feat_m, text_feat_m, pseudo_idx, conf=conf_batch, imgid=img_idx if img_idx is not None else None)
                else:
                    self._dequeue_and_enqueue(image_feat.detach(), text_feat.detach(), pseudo_idx, conf=conf_batch, imgid=img_idx if img_idx is not None else None)
        # ===== Saliency compute =====
        probability_matrix = None 
        saliency_compute_epoch = config.get('saliency_compute_epoch', 5)
        if epoch > saliency_compute_epoch :#and bool(config.get('enable_mlm_loss', False))
            was_train = self.text_encoder.training
            self.text_encoder.eval()
            try:
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
            finally:
                self.text_encoder.train(was_train)
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

        if enable_itm_loss:
            bs = image1.size(0)

            # --- 学生：正样本 (text2, image1) ---
            output_pos = self.text_encoder.bert(
                encoder_embeds=text_embeds,              # 来自 text2 的 text_embeds
                attention_mask=text_atts,                # text2 attention
                encoder_hidden_states=image_embeds,      # 来自 image1 的 image_embeds
                encoder_attention_mask=image_atts,
                return_dict=True,
                mode='fusion',
                output_attentions=enable_itm_softmask,
                output_hidden_states=False,
            )

            # ========= 关键：构造“禁止负采样”的掩码 mask_forbid =========
            # 规则：
            # 1) 同 image_id：永远禁止当负样本（强保护，干净约束）
            # 2) 同 pseudo_label：仅当动量 ITM 判为 likely positive 才禁止；否则允许成为 hard negative

            with torch.no_grad():
                itm_neg_sampling = str(config.get("itm_neg_sampling", "cl")).lower()
                if itm_neg_sampling not in ("cl", "random"):
                    raise ValueError(f"config['itm_neg_sampling'] must be 'cl' or 'random', got {itm_neg_sampling}")

                # --- pseudo 同簇掩码 ---
                idx_1d = pseudo_idx.view(-1)  # [B]
                mask_same_pseudo = torch.eq(idx_1d.view(bs, 1), idx_1d.view(1, bs))  # [B,B] bool

                # --- image_id 强保护掩码（同图多描述永远不做负样本） ---
                if ('image_id' in batch) and (batch['image_id'] is not None):
                    # 这里假设你 forward 前面已经准备好了 img_idx；否则你可以直接用 batch['image_id']
                    if 'img_idx' in locals() and (img_idx is not None):
                        img_1d = img_idx.view(-1)
                    else:
                        img_1d = batch['image_id'].view(-1).to(image1.device)
                    mask_same_img = torch.eq(img_1d.view(bs, 1), img_1d.view(1, bs))  # [B,B] bool
                else:
                    mask_same_img = torch.zeros((bs, bs), device=image1.device, dtype=torch.bool)

                # --- 动量 ITM gate：likely_pos ---
                enable_itm_gate = bool(config.get("enable_itm_gate", True))
                gate_start = int(config.get("itm_gate_start_epoch", 10))
                itm_prob_m = None
                p_diag_m = None

                if enable_itm_gate and (epoch >= gate_start):
                    # 只对 batch 内做 BxB gate，成本很小（B=16）
                    self._momentum_update()
                    image_embeds_m_gate = self.visual_encoder_m(image1)  # 用 image1 和 text2 对齐
                    image_atts_gate = torch.ones(image_embeds_m_gate.size()[:-1], dtype=torch.long, device=image1.device)

                    itm_prob_m = self.compute_itm_prob_matrix_m(
                        image_embeds_m=image_embeds_m_gate,
                        image_atts=image_atts_gate,
                        text_ids=text2["input_ids"],
                        text_atts=text2["attention_mask"],
                        chunk_size=int(config.get("itm_gate_chunk", 128)),
                    )  # [B,B]
                    p_diag_m = itm_prob_m.diag().clamp(0.0, 1.0)  # [B]

                    margin_pos = float(config.get("itm_gate_margin_pos", 0.10))
                    tau_pos = float(config.get("itm_gate_tau_pos", 0.30))
                    thr = torch.clamp(p_diag_m - margin_pos, min=tau_pos).view(bs, 1)  # [B,1]
                    likely_pos = (itm_prob_m >= thr)  # [B,B] bool

                    # 同 pseudo 且 likely_pos -> 禁止作为负样本
                    mask_forbid = mask_same_img | (mask_same_pseudo & likely_pos)
                else:
                    # 没有 gate 时：退化为原始规则（同 pseudo 禁止）+ image_id 强保护
                    mask_forbid = mask_same_img | mask_same_pseudo

                # ========= 根据采样策略产生 neg index =========
                if itm_neg_sampling == "cl":
                    # 使用 CL 相似度分布采样（batch 内）
                    # 若 enable_cl_loss=False，现算 sim
                    if not enable_cl_loss:
                        sim_i2t_batch = image_feat @ text_feat.t()  # [B,B]
                        sim_t2i_batch = text_feat @ image_feat.t()  # [B,B]
                    else:
                        # enable_cl_loss=True 时你可能有 sim_i2t/sim_t2i，但它可能含 queue
                        # 这里稳妥起见强制用 batch 内相似度
                        sim_i2t_batch = image_feat @ text_feat.t()
                        sim_t2i_batch = text_feat @ image_feat.t()

                    weights_i2t = F.softmax(sim_i2t_batch, dim=1)  # [B,B]
                    weights_t2i = F.softmax(sim_t2i_batch, dim=1)  # [B,B]

                    # 应用“禁止负采样”掩码：mask_forbid=True 的位置置 0
                    weights_i2t = weights_i2t.masked_fill(mask_forbid, 0.0)
                    weights_t2i = weights_t2i.masked_fill(mask_forbid, 0.0)

                    # 兜底：若某行全 0（极端：整批都被 forbid），退化为排除自身随机
                    if (weights_i2t.sum(dim=1) == 0).any():
                        w = (~torch.eye(bs, device=image1.device, dtype=torch.bool)).float()
                        weights_i2t = w / (w.sum(dim=1, keepdim=True) + 1e-12)
                    else:
                        weights_i2t = weights_i2t / (weights_i2t.sum(dim=1, keepdim=True) + 1e-12)

                    if (weights_t2i.sum(dim=1) == 0).any():
                        w = (~torch.eye(bs, device=image1.device, dtype=torch.bool)).float()
                        weights_t2i = w / (w.sum(dim=1, keepdim=True) + 1e-12)
                    else:
                        weights_t2i = weights_t2i / (weights_t2i.sum(dim=1, keepdim=True) + 1e-12)

                    image_neg_idx = torch.multinomial(weights_t2i, 1).squeeze(1)  # [B]
                    text_neg_idx  = torch.multinomial(weights_i2t, 1).squeeze(1)  # [B]

                else:
                    # random：在允许集合上均匀采样
                    valid = (~mask_forbid).float()  # [B,B]
                    row_sum = valid.sum(dim=1, keepdim=True)

                    if (row_sum == 0).any():
                        valid_fallback = (~torch.eye(bs, device=image1.device, dtype=torch.bool)).float()
                        valid = torch.where(row_sum > 0, valid, valid_fallback)
                        row_sum = valid.sum(dim=1, keepdim=True)

                    probs = valid / (row_sum + 1e-12)
                    image_neg_idx = torch.multinomial(probs, 1).squeeze(1)
                    text_neg_idx  = torch.multinomial(probs, 1).squeeze(1)

            # ========= 构造 neg batch，做 ITM 分类 =========
            image_embeds_neg = image_embeds[image_neg_idx]        # [B, N, D]
            text_embeds_neg  = text_embeds[text_neg_idx]          # [B, L, D]
            text_atts_neg    = text_atts[text_neg_idx]            # [B, L]

            # 你原来的拼接方式保持不变（注意标签对应）
            text_embeds_all  = torch.cat([text_embeds, text_embeds_neg], dim=0)      # [2B, L, D]
            text_atts_all    = torch.cat([text_atts, text_atts_neg], dim=0)          # [2B, L]
            image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)    # [2B, N, D]
            image_atts_all   = torch.cat([image_atts, image_atts], dim=0)            # [2B, N]

            output_neg_cross = self.text_encoder.bert(
                encoder_embeds=text_embeds_all,
                attention_mask=text_atts_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                return_dict=True,
                mode='fusion',
            )

            vl_embeddings = torch.cat([
                output_pos.last_hidden_state[:, 0, :],            # [B, D]
                output_neg_cross.last_hidden_state[:, 0, :],      # [2B, D]
            ], dim=0)                                             # [3B, D]

            vl_output = self.itm_head(vl_embeddings)              # [3B, 2]

            itm_labels = torch.cat([
                torch.ones(bs, dtype=torch.long),
                torch.zeros(2 * bs, dtype=torch.long)
            ], dim=0).to(image1.device)

            loss_dict['loss_itm'] = F.cross_entropy(vl_output, itm_labels)

        return loss_dict
    @torch.no_grad()
    def compute_itm_prob_matrix_m(
        self,
        image_embeds_m: torch.Tensor,   # [B, N, Dv]
        image_atts: torch.Tensor,       # [B, N]
        text_ids: torch.Tensor,         # [B, L]
        text_atts: torch.Tensor,        # [B, L]
        chunk_size: int = 128,
    ) -> torch.Tensor:
        """
        返回 P_match: [B, B], P[i,j] = Pr(match | image_i, text_j)  (softmax(logits)[...,1])
        用动量分支 text_encoder_m + itm_head_m，no_grad 作为 teacher gate。
        """
        device = text_ids.device
        B = text_ids.size(0)
        total = B * B

        probs = torch.empty(total, device=device, dtype=torch.float32)

        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            idx_flat = torch.arange(start, end, device=device)

            i = torch.div(idx_flat, B, rounding_mode="floor")  # image index
            j = idx_flat % B                                   # text index

            text_ids_ij  = text_ids[j]
            text_atts_ij = text_atts[j]
            img_ij       = image_embeds_m[i]
            img_atts_ij  = image_atts[i]

            out = self.text_encoder_m.bert(
                text_ids_ij,
                attention_mask=text_atts_ij,
                encoder_hidden_states=img_ij,
                encoder_attention_mask=img_atts_ij,
                return_dict=True,
                mode="fusion",
            )
            logits = self.itm_head_m(out.last_hidden_state[:, 0, :])  # [chunk, 2]
            probs[start:end] = logits.softmax(dim=-1)[:, 1].float()

        return probs.view(B, B)
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx, conf=None, imgid=None):
        # gather across GPUs if using DDP
        image_feats = concat_all_gather(image_feat)  # [bs_all, D]
        text_feats  = concat_all_gather(text_feat)   # [bs_all, D]
        idxs        = concat_all_gather(idx)         # [bs_all, 1] or [bs_all]
        if idxs.dim() == 1:
            idxs = idxs.view(-1, 1)

        if conf is None:
            confs = torch.ones((idxs.size(0),), device=idxs.device, dtype=torch.float32)
        else:
            conf = conf.view(-1, 1)
            confs = concat_all_gather(conf).view(-1).float()

        if imgid is None:
            imgids = torch.full((idxs.size(0), 1), -1, device=idxs.device, dtype=torch.long)
        else:
            imgid = imgid.view(-1, 1).long()
            imgids = concat_all_gather(imgid).view(-1, 1).long()

        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)

        # 经典 ALBEF 要求 queue_size 能被 batch_size 整除（否则你自己改成循环写入）
        assert self.queue_size % batch_size == 0, f"queue_size {self.queue_size} must be divisible by batch_size {batch_size}"

        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:,  ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:,   ptr:ptr + batch_size] = idxs.T
        self.conf_queue[:,  ptr:ptr + batch_size] = confs.view(1, -1)
        self.imgid_queue[:, ptr:ptr + batch_size] = imgids.view(1, -1)

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr