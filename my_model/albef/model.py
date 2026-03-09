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
        self.register_buffer('pos_mu', torch.tensor(0.0))
        self.register_buffer('pos_var', torch.tensor(0.0))
        self.register_buffer('rectify_initialized', torch.tensor(0))

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
                        # 1) 统计 batch 内“当前认为的正样本”的相似度分布（更稳定）
                        current_pos_sims = sim_i2t_m[:, :bs][pos_idx[:, :bs].bool()]
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

                        # 2) 假阳性抑制：伪正但分数显著低
                        fp_mask_i2t = rectified_pos_idx.bool() & (sim_i2t_m < (self.pos_mu - gamma * sigma))
                        fp_mask_t2i = rectified_pos_idx.bool() & (sim_t2i_m < (self.pos_mu - gamma * sigma))
                        fp_mask = fp_mask_i2t | fp_mask_t2i
                        rectified_pos_idx[fp_mask] = 0.0

                        # （可选但安全）强制对角为正，避免极端阈值误伤
                        if bool(config.get("rectify_keep_diag", True)):
                            # rectified_pos_idx 是 [B, B(+Q)]，fill_diagonal_ 只会写前 B 个对角元素
                            rectified_pos_idx.fill_diagonal_(1.0)

                        # 3) 假阴性打捞：Top-K + 动量 ITM 复核
                        k = int(config.get('rectify_topk', 1))
                        tau = float(config.get('rectify_tau', 0.85))

                        if k > 0:
                            neg_sim_i2t = sim_i2t_m[:, :bs].clone()
                            neg_sim_i2t[pos_idx[:, :bs].bool()] = -1e4  # 屏蔽原始正样本
                            _, topk_idx = neg_sim_i2t.topk(k, dim=1)    # [B,K]

                            # 准备 ITM 动量头输入
                            text_embeds_m_rep = text_output_m.last_hidden_state.unsqueeze(1).repeat(1, k, 1, 1).view(bs * k, -1, self.text_width)
                            text_atts_rep = text2['attention_mask'].unsqueeze(1).repeat(1, k, 1).view(bs * k, -1)

                            image_embeds_m_gat = image_embeds_m[topk_idx.flatten()]
                            image_atts_gat = image_atts[topk_idx.flatten()]

                            output_m_cross = self.text_encoder_m.bert(
                                encoder_embeds=text_embeds_m_rep,
                                attention_mask=text_atts_rep,
                                encoder_hidden_states=image_embeds_m_gat,
                                encoder_attention_mask=image_atts_gat,
                                return_dict=True,
                                mode='fusion'
                            )
                            itm_m_logits = self.itm_head_m(output_m_cross.last_hidden_state[:, 0, :])
                            itm_m_probs = F.softmax(itm_m_logits, dim=1)[:, 1]

                            fn_mask_flat = (itm_m_probs > tau).view(bs, k)
                            for i in range(bs):
                                for j in range(k):
                                    if fn_mask_flat[i, j]:
                                        rectified_pos_idx[i, topk_idx[i, j]] = 1.0

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