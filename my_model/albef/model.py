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
import os


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
        self.register_buffer('pos_mu', torch.tensor(0.0))
        self.register_buffer('pos_var', torch.tensor(0.0))
        self.register_buffer('rectify_initialized', torch.tensor(0))
        ######新增
        self.fov_stat_ks = [1, 2, 3]
        self._last_rectify_train_k = 1
        self.reset_fov_epoch_stats()

    def forward(self, batch, alpha, config, epoch):# text1/text2相同, img1/img2 同一图不同增广
        loss_dict = {}

        image1 = batch['image1']
        image2 = batch['image2']

        text1 = self.tokenizer(
            batch['caption1'],
            padding='longest',
            max_length=config['max_words'],
            return_tensors="pt"
        ).to(image1.device)

        text2 = self.tokenizer(
            batch['caption2'],
            padding='longest',
            max_length=config['max_words'],
            return_tensors="pt"
        ).to(image1.device)

        text_atts = text2['attention_mask']
        idx = batch['pseudo_label']  # 聚类伪标签 (ID)

        bs = image1.size(0)

        # ========= Online encoder =========
        image_embeds = self.visual_encoder(image1, register_blk=-1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image1.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text_output = self.text_encoder.bert(
            text2['input_ids'],
            attention_mask=text2['attention_mask'],
            return_dict=True,
            mode='text'
        )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # 给 ITM 用：可能被重整后的正样本矩阵（B x (B+Q)）
        rectified_pos_idx = None

        # 给 MLM 复用（避免重复算动量分支）
        image_embeds_m = None
        image_feat_m = None
        text_output_m = None
        text_feat_m = None

        # ========= ITC / Contrastive loss =========
        enable_cl_loss = bool(config.get('enable_cl_loss', True))  # 你说 ITC 永远开启，这里仍保留开关以防配置误传
        use_momentum = bool(config.get('use_momentum', True))
        use_queue = bool(config.get('use_queue', True))

        if enable_cl_loss:
            idx = idx.view(-1, 1)  # [B,1]
            if use_queue:
                idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)  # [1, B+Q]
            else:
                idx_all = idx.t()  # [1,B]

            pos_idx = torch.eq(idx, idx_all).float()  # [B, B(+Q)]
            sim_targets_hard = pos_idx / (pos_idx.sum(1, keepdim=True) + 1e-8)
            rectified_pos_idx = pos_idx.clone()

            # 下面这两个是“student logits 的 keys 集合”，必须与 teacher 生成 targets 时一致
            image_feat_all = None
            text_feat_all = None
            sim_i2t_targets = None
            sim_t2i_targets = None

            with torch.no_grad():
                if use_momentum:
                    # 动量更新（teacher）
                    self._momentum_update()

                    # keys 用 image2/text2（与你原逻辑一致）
                    image_embeds_m = self.visual_encoder_m(image2)
                    image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)

                    text_output_m = self.text_encoder_m.bert(
                        text2['input_ids'],
                        attention_mask=text2['attention_mask'],
                        return_dict=True,
                        mode='text'
                    )
                    text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)

                    # keys 集合：动量 batch keys (+ queue)
                    if use_queue:
                        image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)  # [D, B+Q]
                        text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)    # [D, B+Q]
                    else:
                        image_feat_all = image_feat_m.t()  # [D,B]
                        text_feat_all = text_feat_m.t()    # [D,B]

                    # teacher 分布（support 与 image_feat_all/text_feat_all 一致）
                    sim_i2t_m = image_feat_m @ text_feat_all / self.temp  # [B, B(+Q)]
                    sim_t2i_m = text_feat_m @ image_feat_all / self.temp  # [B, B(+Q)]

                    # ========= Dynamic Label Rectification =========
                    rectify_epoch = int(config.get('rectify_epoch', 5))
                    if epoch >= rectify_epoch:
                        # ------------------------------
                        # batch-local GT relation for analysis only
                        # 排除对角线 self-pair，不纳入统计
                        # ------------------------------
                        gt_id = batch['person_id'].view(-1, 1)
                        gt_pos_local_bool = torch.eq(gt_id, gt_id.t())  # [B,B], bool

                        diag_mask = torch.eye(bs, dtype=torch.bool, device=image1.device)

                        pos_idx_local_bool_all = pos_idx[:, :bs].bool()            # 含对角线，仅供训练逻辑使用
                        coarse_pos_local_bool = pos_idx_local_bool_all & (~diag_mask)  # 统计口径：off-diagonal coarse positives
                        nonpos_local_bool = (~pos_idx_local_bool_all) & (~diag_mask)   # 统计口径：off-diagonal non-positives

                        # 1) 统计 batch 内“当前认为的正样本”的相似度分布
                        current_pos_sims = sim_i2t_m[:, :bs][pos_idx_local_bool_all]
                        if current_pos_sims.numel() > 0:
                            batch_mu = current_pos_sims.mean()
                            batch_var = current_pos_sims.var() if current_pos_sims.numel() > 1 else torch.tensor(0.0, device=image1.device)

                            if self.rectify_initialized.item() == 0:
                                self.pos_mu.copy_(batch_mu)
                                self.pos_var.copy_(batch_var)
                                self.rectify_initialized.fill_(1)
                            else:
                                m_ema = 0.99
                                self.pos_mu.copy_(m_ema * self.pos_mu + (1 - m_ema) * batch_mu)
                                self.pos_var.copy_(m_ema * self.pos_var + (1 - m_ema) * batch_var)

                        sigma = torch.sqrt(self.pos_var + 1e-5)
                        gamma = float(config.get('rectify_gamma', 1.5))
                        thresh = self.pos_mu - gamma * sigma

                        # 2) 假阳性抑制：伪正但分数显著低
                        # 训练逻辑仍然保持原样：在 B x (B+Q) 上做
                        fp_mask_i2t = rectified_pos_idx.bool() & (sim_i2t_m < thresh)
                        fp_mask_t2i = rectified_pos_idx.bool() & (sim_t2i_m < thresh)
                        fp_mask = fp_mask_i2t | fp_mask_t2i
                        rectified_pos_idx[fp_mask] = 0.0

                        # keep diagonal positive
                        if bool(config.get("rectify_keep_diag", True)):
                            rectified_pos_idx.fill_diagonal_(1.0)

                        # ------------------------------
                        # Pruning stats: 只统计 batch-local, off-diagonal
                        # effective pruned = 原 coarse positive，但在 prune 后 local 矩阵中变成 0
                        # ------------------------------
                        post_prune_local_bool = rectified_pos_idx[:, :bs].bool().clone()
                        effective_pruned_local_bool = coarse_pos_local_bool & (~post_prune_local_bool)

                        coarse_fp_local_bool = coarse_pos_local_bool & (~gt_pos_local_bool)
                        pruned_true_fp_local_bool = effective_pruned_local_bool & (~gt_pos_local_bool)
                        pruned_false_kill_local_bool = effective_pruned_local_bool & gt_pos_local_bool

                        # 3) 假阴性打捞：一次前向同时算 max(train_topk, 3)
                        train_topk = int(config.get('rectify_topk', 1))
                        self._last_rectify_train_k = train_topk

                        stats_topk_max = int(config.get('fov_stats_topk_max', 3))  # 默认统计到 topk=3
                        max_topk_needed = max(train_topk, stats_topk_max)
                        max_topk_needed = min(max_topk_needed, bs)

                        tau = float(config.get('rectify_tau', 0.85))

                        cand_masks_by_k = {}
                        rescued_masks_by_k = {}

                        if max_topk_needed > 0:
                            neg_sim_i2t = sim_i2t_m[:, :bs].clone()
                            neg_sim_i2t[pos_idx_local_bool_all] = -1e4  # 屏蔽原始正样本（含对角）

                            _, topk_idx_all = neg_sim_i2t.topk(max_topk_needed, dim=1)   # [B, maxk]
                            topk_valid_all = (~pos_idx_local_bool_all).gather(1, topk_idx_all)  # 防止极端情况下选到被 mask 的位置

                            # ITM 动量头：一次性算完 max_topk_needed 个候选
                            text_embeds_m_rep = (
                                text_output_m.last_hidden_state
                                .unsqueeze(1)
                                .repeat(1, max_topk_needed, 1, 1)
                                .view(bs * max_topk_needed, -1, self.text_width)
                            )
                            text_atts_rep = (
                                text2['attention_mask']
                                .unsqueeze(1)
                                .repeat(1, max_topk_needed, 1)
                                .view(bs * max_topk_needed, -1)
                            )

                            image_embeds_m_gat = image_embeds_m[topk_idx_all.reshape(-1)]
                            image_atts_gat = image_atts[topk_idx_all.reshape(-1)]

                            output_m_cross = self.text_encoder_m.bert(
                                encoder_embeds=text_embeds_m_rep,
                                attention_mask=text_atts_rep,
                                encoder_hidden_states=image_embeds_m_gat,
                                encoder_attention_mask=image_atts_gat,
                                return_dict=True,
                                mode='fusion'
                            )
                            itm_m_logits = self.itm_head_m(output_m_cross.last_hidden_state[:, 0, :])
                            itm_m_probs_all = F.softmax(itm_m_logits, dim=1)[:, 1].view(bs, max_topk_needed)

                            # ---- 实际训练只使用 config['rectify_topk'] ----
                            actual_k = min(train_topk, max_topk_needed)
                            if actual_k > 0:
                                actual_rescue_prefix = topk_valid_all[:, :actual_k] & (itm_m_probs_all[:, :actual_k] > tau)
                                actual_rescued_mask = torch.zeros((bs, bs), dtype=torch.bool, device=image1.device)
                                actual_rescued_mask.scatter_(1, topk_idx_all[:, :actual_k], actual_rescue_prefix)
                                rectified_pos_idx[:, :bs][actual_rescued_mask] = 1.0

                            # ---- 统计 top-k=1,2,3 ----
                            for stat_k in self.fov_stat_ks:
                                eff_k = min(int(stat_k), max_topk_needed)

                                cand_mask_k = torch.zeros((bs, bs), dtype=torch.bool, device=image1.device)
                                rescued_mask_k = torch.zeros((bs, bs), dtype=torch.bool, device=image1.device)

                                if eff_k > 0:
                                    cand_prefix = topk_valid_all[:, :eff_k]
                                    cand_mask_k.scatter_(1, topk_idx_all[:, :eff_k], cand_prefix)

                                    rescued_prefix = cand_prefix & (itm_m_probs_all[:, :eff_k] > tau)
                                    rescued_mask_k.scatter_(1, topk_idx_all[:, :eff_k], rescued_prefix)

                                # 只保留 off-diagonal event
                                cand_mask_k = cand_mask_k & (~diag_mask)
                                rescued_mask_k = rescued_mask_k & (~diag_mask)

                                cand_masks_by_k[int(stat_k)] = cand_mask_k
                                rescued_masks_by_k[int(stat_k)] = rescued_mask_k
                        else:
                            for stat_k in self.fov_stat_ks:
                                cand_masks_by_k[int(stat_k)] = torch.zeros((bs, bs), dtype=torch.bool, device=image1.device)
                                rescued_masks_by_k[int(stat_k)] = torch.zeros((bs, bs), dtype=torch.bool, device=image1.device)

                        # ------------------------------
                        # Rescue stats: 只统计 batch-local, off-diagonal
                        # ------------------------------
                        eligible_fn_local_bool = nonpos_local_bool & gt_pos_local_bool

                        for stat_k in self.fov_stat_ks:
                            cand_mask_k = cand_masks_by_k[int(stat_k)]
                            rescued_mask_k = rescued_masks_by_k[int(stat_k)]

                            verified_true_fn_local_bool = cand_mask_k & gt_pos_local_bool
                            rescued_true_fn_local_bool = rescued_mask_k & gt_pos_local_bool
                            rescued_false_alarm_local_bool = rescued_mask_k & (~gt_pos_local_bool)

                            self._add_fov_stat(stat_k, 'coarse_pos_total', coarse_pos_local_bool.sum().item())
                            self._add_fov_stat(stat_k, 'coarse_fp_total', coarse_fp_local_bool.sum().item())
                            self._add_fov_stat(stat_k, 'pruned_total', effective_pruned_local_bool.sum().item())
                            self._add_fov_stat(stat_k, 'pruned_true_fp_total', pruned_true_fp_local_bool.sum().item())
                            self._add_fov_stat(stat_k, 'pruned_false_kill_total', pruned_false_kill_local_bool.sum().item())

                            self._add_fov_stat(stat_k, 'eligible_nonpos_total', nonpos_local_bool.sum().item())
                            self._add_fov_stat(stat_k, 'eligible_fn_total', eligible_fn_local_bool.sum().item())
                            self._add_fov_stat(stat_k, 'verified_total', cand_mask_k.sum().item())
                            self._add_fov_stat(stat_k, 'verified_true_fn_total', verified_true_fn_local_bool.sum().item())
                            self._add_fov_stat(stat_k, 'rescued_total', rescued_mask_k.sum().item())
                            self._add_fov_stat(stat_k, 'rescued_true_fn_total', rescued_true_fn_local_bool.sum().item())
                            self._add_fov_stat(stat_k, 'rescued_false_alarm_total', rescued_false_alarm_local_bool.sum().item())
                    # 重整后的 hard targets
                    sim_targets_rect = rectified_pos_idx / (rectified_pos_idx.sum(1, keepdim=True) + 1e-8)

                    # teacher soft + rectified hard
                    sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets_rect
                    sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets_rect

                else:
                    # 不用动量：keys 用 online (detach) (+queue)
                    if use_queue:
                        image_feat_all = torch.cat([image_feat.detach().t(), self.image_queue.clone().detach()], dim=1)
                        text_feat_all = torch.cat([text_feat.detach().t(), self.text_queue.clone().detach()], dim=1)
                    else:
                        image_feat_all = image_feat.detach().t()
                        text_feat_all = text_feat.detach().t()

                    sim_i2t_targets = sim_targets_hard
                    sim_t2i_targets = sim_targets_hard

            # ===== Student logits：必须使用上面构造的同一套 keys 集合 =====
            sim_i2t = image_feat @ text_feat_all / self.temp  # [B, B(+Q)]
            sim_t2i = text_feat @ image_feat_all / self.temp  # [B, B(+Q)]

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            loss_dict['loss_cl'] = (loss_i2t + loss_t2i) / 2

            # 队列更新：仍然使用“原始伪标签 ID”
            if use_queue:
                if use_momentum:
                    self._dequeue_and_enqueue(image_feat_m, text_feat_m, batch['pseudo_label'].view(-1, 1))
                else:
                    self._dequeue_and_enqueue(image_feat.detach(), text_feat.detach(), batch['pseudo_label'].view(-1, 1))

        # ========= Saliency compute (for curriculum mask) =========
        probability_matrix = None
        saliency = None
        saliency_compute_epoch = int(config.get('saliency_compute_epoch', 5))
        if epoch > saliency_compute_epoch:
            with torch.no_grad():
                saliency = self.compute_cross_modal_groundedness(
                    text_ids=text1['input_ids'],
                    attention_mask=text1['attention_mask'],
                    image_embeds=image_embeds,
                    image_atts=image_atts,
                    saliency_image=None,
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

        # ========= MLM =========
        enable_mlm_loss = bool(config.get('enable_mlm_loss', False))
        enable_soft_label = bool(config.get('mlm_soft_label', False))

        if enable_mlm_loss:
            input_ids = text1.input_ids.clone()
            labels = input_ids.clone()
            ids_before_debug = input_ids.clone()

            input_ids, labels = self.mask(
                input_ids,
                self.text_encoder.config.vocab_size,
                targets=labels,
                probability_matrix=probability_matrix
            )

            if enable_soft_label:
                with torch.no_grad():
                    # 软标签 teacher 必须来自动量分支；若上面没算到，则这里补一次（保证健壮）
                    if image_embeds_m is None or text_output_m is None:
                        # 若你希望 MLM soft label 必须依赖 use_momentum=True，可在这里 assert
                        self._momentum_update()
                        image_embeds_m = self.visual_encoder_m(image2)
                        text_output_m = self.text_encoder_m.bert(
                            text1['input_ids'],
                            attention_mask=text1['attention_mask'],
                            return_dict=True,
                            mode='text'
                        )

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
                # 防止 saliency 未计算就 debug 崩掉
                if saliency is not None:
                    self.debug_render_mask_with_norms(
                        epoch=int(epoch),
                        step=int(batch.get("global_step", 0)),
                        input_ids_before=ids_before_debug,
                        input_ids_after=input_ids,
                        targets=labels,
                        attention_mask=text1['attention_mask'],
                        probability_matrix=probability_matrix,
                        saliency_norm=saliency,
                        raw_texts=batch.get('caption1', None),
                        out_path=str(config.get("debug_mask_file", "./mask_output2.txt")),
                        limit_per_epoch=int(config.get("debug_mask_limit_per_epoch", 30)),
                        sample_per_step=int(config.get("debug_sample_per_step", 2)),
                        topk_tokens=int(config.get("debug_topk_tokens", 8)),
                        step_prob=float(config.get("debug_step_prob", 0.15)),
                    )

            loss_dict['loss_mlm'] = mlm_output.loss

        # ========= ITM =========
        enable_itm_loss = bool(config.get('enable_itm_loss', False))
        enable_itm_softmask = bool(config.get('enable_itm_softmask', False))

        if enable_itm_loss:
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
                itm_neg_sampling = str(config.get("itm_neg_sampling", "cl")).lower()

                rectify_epoch = int(config.get('rectify_epoch', 5))
                if rectified_pos_idx is not None and epoch >= rectify_epoch:
                    mask_same = rectified_pos_idx[:, :bs].bool()
                    if bool(config.get("rectify_sym_itm", True)):
                        mask_same = mask_same | mask_same.t()
                else:
                    idx_1d = batch['pseudo_label'].view(-1)
                    mask_same = torch.eq(idx_1d.view(bs, 1), idx_1d.view(1, bs))

                # 显式排除对角（自配对不应作为负样本）
                mask_same = mask_same.clone()
                mask_same.fill_diagonal_(True)

                if itm_neg_sampling == "cl":
                    # 统一确保 sim_i2t_batch / sim_t2i_batch 有定义
                    # ITC 永远开启时，优先沿用 ITC 的 batch 内相似度（对 queue 开启也只取前 bs 列）
                    if 'sim_i2t' in locals() and 'sim_t2i' in locals():
                        sim_i2t_batch = sim_i2t[:, :bs]
                        sim_t2i_batch = sim_t2i[:, :bs]
                    else:
                        sim_i2t_batch = image_feat @ text_feat.t()
                        sim_t2i_batch = text_feat @ image_feat.t()

                    weights_i2t = F.softmax(sim_i2t_batch, dim=1)
                    weights_t2i = F.softmax(sim_t2i_batch, dim=1)

                    # 允许作为负样本的位置
                    allow = (~mask_same).float()

                    # 先把不允许的位置置 0
                    weights_i2t = weights_i2t * allow
                    weights_t2i = weights_t2i * allow

                    row_sum_i2t = weights_i2t.sum(dim=1, keepdim=True)
                    row_sum_t2i = weights_t2i.sum(dim=1, keepdim=True)

                    # 若某行全 0（全部被 mask 掉），回退为均匀分布（仅在 allow 的位置上均匀）
                    fallback = allow / (allow.sum(dim=1, keepdim=True) + 1e-12)

                    weights_i2t = torch.where(row_sum_i2t > 0, weights_i2t / (row_sum_i2t + 1e-12), fallback)
                    weights_t2i = torch.where(row_sum_t2i > 0, weights_t2i / (row_sum_t2i + 1e-12), fallback)

                    image_neg_idx = torch.multinomial(weights_t2i, 1).squeeze(1)
                    text_neg_idx = torch.multinomial(weights_i2t, 1).squeeze(1)

                else:
                    valid = (~mask_same).float()
                    row_sum = valid.sum(dim=1, keepdim=True)
                    fallback = (~torch.eye(bs, device=image1.device, dtype=torch.bool)).float()
                    fallback = fallback / (fallback.sum(dim=1, keepdim=True) + 1e-12)

                    probs = valid / (row_sum + 1e-12)
                    probs = torch.where(row_sum > 0, probs, fallback)

                    image_neg_idx = torch.multinomial(probs, 1).squeeze(1)
                    text_neg_idx = torch.multinomial(probs, 1).squeeze(1)

            image_embeds_neg = image_embeds[image_neg_idx]
            text_embeds_neg = text_embeds[text_neg_idx]
            text_atts_neg = text_atts[text_neg_idx]

            text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
            text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
            image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
            image_atts_all = torch.cat([image_atts, image_atts], dim=0)

            output_neg_cross = self.text_encoder.bert(
                encoder_embeds=text_embeds_all,
                attention_mask=text_atts_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                return_dict=True,
                mode='fusion',
            )

            vl_embeddings = torch.cat(
                [output_pos.last_hidden_state[:, 0, :],
                output_neg_cross.last_hidden_state[:, 0, :]],
                dim=0
            )
            vl_output = self.itm_head(vl_embeddings)

            itm_labels = torch.cat(
                [torch.ones(bs, dtype=torch.long),
                torch.zeros(2 * bs, dtype=torch.long)],
                dim=0
            ).to(image1.device)

            loss_itm = F.cross_entropy(vl_output, itm_labels)
            loss_dict['loss_itm'] = loss_itm

        return loss_dict
    def _make_empty_fov_counter(self):
        return {
            'coarse_pos_total': 0,            # batch-local, off-diagonal coarse positives
            'coarse_fp_total': 0,             # 上面这些 coarse positives 里真实为负的数量
            'pruned_total': 0,                # 实际被 prune 掉的数量
            'pruned_true_fp_total': 0,        # 被 prune 且真实为负（成功剪掉的假正）
            'pruned_false_kill_total': 0,     # 被 prune 但真实为正（误杀）

            'eligible_nonpos_total': 0,       # batch-local, off-diagonal non-positives
            'eligible_fn_total': 0,           # 上面这些 non-positives 里真实为正（潜在假负）
            'verified_total': 0,              # 被送入 verifier 的候选数
            'verified_true_fn_total': 0,      # 候选池中真实为正的数量
            'rescued_total': 0,               # 最终被 rescue 为正的数量
            'rescued_true_fn_total': 0,       # 被 rescue 且真实为正（成功救回的假负）
            'rescued_false_alarm_total': 0,   # 被 rescue 但真实为负（误救）
        }


    def reset_fov_epoch_stats(self):
        self.fov_epoch_stats = {
            int(k): self._make_empty_fov_counter()
            for k in getattr(self, 'fov_stat_ks', [1, 2, 3])
        }


    def _add_fov_stat(self, topk, name, value):
        topk = int(topk)
        if topk not in self.fov_epoch_stats:
            self.fov_epoch_stats[topk] = self._make_empty_fov_counter()
        self.fov_epoch_stats[topk][name] += int(value)


    @staticmethod
    def _safe_div(num, den):
        return float(num) / float(den) if int(den) > 0 else 0.0


    def get_fov_epoch_stats(self):
        out = {}
        for k, raw in self.fov_epoch_stats.items():
            d = {kk: int(v) for kk, v in raw.items()}

            d['prune_ratio'] = self._safe_div(d['pruned_total'], d['coarse_pos_total'])
            d['prune_precision'] = self._safe_div(d['pruned_true_fp_total'], d['pruned_total'])
            d['prune_recall_over_fp'] = self._safe_div(d['pruned_true_fp_total'], d['coarse_fp_total'])

            d['candidate_ratio'] = self._safe_div(d['verified_total'], d['eligible_nonpos_total'])
            d['acceptance_rate'] = self._safe_div(d['rescued_total'], d['verified_total'])
            d['rescue_precision'] = self._safe_div(d['rescued_true_fn_total'], d['rescued_total'])
            d['rescue_recall_at_cand'] = self._safe_div(d['rescued_true_fn_total'], d['verified_true_fn_total'])
            d['rescue_recall_over_all_fn'] = self._safe_div(d['rescued_true_fn_total'], d['eligible_fn_total'])

            # 便于直接看绝对数
            d['successful_fp_suppressed'] = d['pruned_true_fp_total']
            d['successful_fn_rescued'] = d['rescued_true_fn_total']

            out[int(k)] = d
        return out


    def dump_fov_epoch_stats(self, epoch, save_dir="."):
        os.makedirs(save_dir, exist_ok=True)
        all_stats = self.get_fov_epoch_stats()
        train_topk = int(getattr(self, "_last_rectify_train_k", -1))

        for k, d in all_stats.items():
            path = os.path.join(save_dir, f"fov_stats_topk{k}.txt")
            with open(path, "a", encoding="utf-8") as f:
                f.write("=" * 100 + "\n")
                f.write(f"epoch={epoch} | train_rectify_topk={train_topk} | analysis_topk={k}\n")

                f.write("[raw counts]\n")
                f.write(f"coarse_pos_total={d['coarse_pos_total']}\n")
                f.write(f"coarse_fp_total={d['coarse_fp_total']}\n")
                f.write(f"pruned_total={d['pruned_total']}\n")
                f.write(f"pruned_true_fp_total={d['pruned_true_fp_total']}\n")
                f.write(f"pruned_false_kill_total={d['pruned_false_kill_total']}\n")

                f.write(f"eligible_nonpos_total={d['eligible_nonpos_total']}\n")
                f.write(f"eligible_fn_total={d['eligible_fn_total']}\n")
                f.write(f"verified_total={d['verified_total']}\n")
                f.write(f"verified_true_fn_total={d['verified_true_fn_total']}\n")
                f.write(f"rescued_total={d['rescued_total']}\n")
                f.write(f"rescued_true_fn_total={d['rescued_true_fn_total']}\n")
                f.write(f"rescued_false_alarm_total={d['rescued_false_alarm_total']}\n")

                f.write("[paper metrics]\n")
                f.write(f"prune_ratio={d['prune_ratio']:.6f}\n")
                f.write(f"prune_precision={d['prune_precision']:.6f}\n")
                f.write(f"candidate_ratio={d['candidate_ratio']:.6f}\n")
                f.write(f"acceptance_rate={d['acceptance_rate']:.6f}\n")
                f.write(f"rescue_precision={d['rescue_precision']:.6f}\n")
                f.write(f"rescue_recall_at_cand={d['rescue_recall_at_cand']:.6f}\n")

                f.write("[extra metrics]\n")
                f.write(f"prune_recall_over_fp={d['prune_recall_over_fp']:.6f}\n")
                f.write(f"rescue_recall_over_all_fn={d['rescue_recall_over_all_fn']:.6f}\n")

                f.write("[absolute answers]\n")
                f.write(f"successful_fp_suppressed={d['successful_fp_suppressed']}\n")
                f.write(f"successful_fn_rescued={d['successful_fn_rescued']}\n\n")