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
)
from .mixins.infmask import InfMaskMixin


class ALBEF(VisionBuilderMixin, MomentumMixin, QueueMixin, MLMMixin, SaliencyMixin, DebugMaskMixin, InfMaskMixin, nn.Module):
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
        # === [A] InfMask 专属 TINY 头 & 独立温度 ===
        d_inf = int(config.get('infmask_dim', 256))  # 目标维度（TINY）
        self.infmask_head = nn.Linear(self.text_width, d_inf, bias=False)
        self.infmask_ln   = nn.LayerNorm(d_inf)
        self.infmask_temp = nn.Parameter(torch.tensor(float(config.get('infmask_temp', 0.07))))

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
        image_embeds = self.visual_encoder(image1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image1.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        # extract text features
        text_output = self.text_encoder.bert(text2['input_ids'], attention_mask=text2['attention_mask'], return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        
        # ===== Contrastive loss =====
        enable_cl_loss = bool(config.get('enable_cl_loss', True))
        if enable_cl_loss:
            idx = idx.view(-1, 1)
            idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
            with torch.no_grad():
                self._momentum_update()
                image_embeds_m = self.visual_encoder_m(image2)
                image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
                image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

                text_output_m = self.text_encoder_m.bert(text2['input_ids'], attention_mask=text2['attention_mask'], return_dict=True, mode='text')
                text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
                sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp
                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ image_feat_all / self.temp

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            loss_dict['loss_cl'] = (loss_i2t + loss_t2i) / 2

            self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

        # ===== Masked Language Modeling =====
        enable_mlm_loss = bool(config.get('enable_mlm_loss', True))
        enable_soft_label = bool(config.get('mlm_soft_label', False))
        probability_matrix = None  # ensure defined for later use
        image_embeds_m = None      # ensure defined if used below
        if enable_mlm_loss:
            saliency_compute_epoch = config.get('saliency_compute_epoch', 99)
            if epoch > saliency_compute_epoch:
                with torch.no_grad():
                    saliency = self.compute_cross_modal_saliency(
                        text_ids=text1['input_ids'],
                        attention_mask=text1['attention_mask'],
                        image_embeds=image_embeds,
                        image_atts=image_atts,
                        layers=config.get('saliency_layers', 3),
                    )
                probability_matrix = self.build_curriculum_mask_probs(
                    saliency=saliency,
                    attention_mask=text1['attention_mask'],
                    input_ids=text1['input_ids'],
                    base_prob=float(config.get('mlm_probability', self.mlm_probability)),
                    focus_top_p=float(config.get('mlm_focus_top_p', 0.3)),
                    p_strong=float(config.get('mlm_p_strong', 0.95)),
                    p_min=float(config.get('mlm_prob_min', 0.0)),
                    p_max=float(config.get('mlm_prob_max', 0.95)),
                )
            else:
                probability_matrix = None

            input_ids = text1.input_ids.clone()
            labels = input_ids.clone()
            ids_before_debug = input_ids.clone()
            input_ids, labels = self.mask(
                input_ids,
                self.text_encoder.config.vocab_size,
                targets=labels,
                probability_matrix=None  # 与 InfMasking 解耦，若需可替换为 probability_matrix
            )
            debug_mask_epoch = config.get('debug_mask_epoch', 99)
            if epoch > debug_mask_epoch:
                self.debug_render_mask_diff(
                    epoch=epoch,
                    input_ids_before=ids_before_debug,
                    input_ids_after=input_ids,
                    targets=labels,
                    attention_mask=text1['attention_mask'],
                    raw_texts=batch.get('caption1', None),
                    limit_per_epoch=int(config.get('debug_mask_limit_per_epoch', 50)),
                    out_path=config.get('debug_mask_file', None),
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
            loss_dict['loss_mlm'] = mlm_output.loss

        # ===== ITM (matched/unmatched) =====
        enable_itm_loss = bool(config.get('enable_itm_loss', True))
        enable_soft_itm = bool(config.get('itm_soft_label', False))  # ✱ 新增开关
        if enable_itm_loss:
            # --- 学生：正样本 (text2, image1) ---
            output_pos = self.text_encoder.bert(
                encoder_embeds=text_embeds,
                attention_mask=text_atts,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                mode='fusion',
            )

            # --- 学生：用相似度采负样本（保持原逻辑） ---
            with torch.no_grad():
                bs = image1.size(0)
                # 如果 CL 关了，就用 in-batch cos sim
                if not enable_cl_loss:
                    sim_i2t = image_feat @ text_feat.t()
                    sim_t2i = text_feat @ image_feat.t()

                weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
                weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)
                mask = torch.eq(idx, idx.T)
                if idx.shape[0] <= 2:
                    raise ValueError("Batch size too small, idx.shape[0] = {}".format(idx.shape[0]))
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)

            # 按权重采样负图 / 负文
            image_neg_idx = torch.multinomial(weights_t2i, 1).flatten()
            image_embeds_neg = image_embeds[image_neg_idx]
            text_neg_idx = torch.multinomial(weights_i2t, 1).flatten()
            text_embeds_neg = text_embeds[text_neg_idx]
            text_atts_neg = text_atts[text_neg_idx]

            # --- 学生：负样本 cross-fusion ---
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

            vl_embeddings = torch.cat([
                output_pos.last_hidden_state[:, 0, :],
                output_neg_cross.last_hidden_state[:, 0, :]
            ], dim=0)
            vl_output = self.itm_head(vl_embeddings)

            itm_labels = torch.cat([
                torch.ones(bs, dtype=torch.long),
                torch.zeros(2 * bs, dtype=torch.long)
            ], dim=0).to(image1.device)

            # --- 软标签版本：用动量教师给 logit 变软 ---
            if enable_soft_itm:
                with torch.no_grad():
                    # 和 CL / InfMask 一样，先更新动量塔
                    self._momentum_update()

                    # 动量图像编码
                    image_embeds_m = self.visual_encoder_m(image1)
                    image_atts_m = torch.ones(
                        image_embeds_m.size()[:-1],
                        dtype=torch.long,
                        device=image1.device,
                    )
                    image_embeds_neg_m = image_embeds_m[image_neg_idx]

                    # 动量文本编码（text 模式）
                    text_output_m = self.text_encoder_m.bert(
                        text2['input_ids'],
                        attention_mask=text_atts,
                        return_dict=True,
                        mode='text',
                    )
                    text_embeds_m = text_output_m.last_hidden_state
                    text_embeds_neg_m = text_embeds_m[text_neg_idx]
                    text_atts_neg_m = text_atts_neg  # mask 同样索引即可

                    # 动量正样本 fusion (text2, image1)
                    output_pos_m = self.text_encoder_m.bert(
                        encoder_embeds=text_embeds_m,
                        attention_mask=text_atts,
                        encoder_hidden_states=image_embeds_m,
                        encoder_attention_mask=image_atts_m,
                        return_dict=True,
                        mode='fusion',
                    )

                    # 动量负样本 fusion，顺序与学生完全一致
                    text_embeds_all_m = torch.cat([text_embeds_m, text_embeds_neg_m], dim=0)
                    text_atts_all_m = torch.cat([text_atts, text_atts_neg_m], dim=0)
                    image_embeds_all_m = torch.cat([image_embeds_neg_m, image_embeds_m], dim=0)
                    image_atts_all_m = torch.cat([image_atts_m, image_atts_m], dim=0)

                    output_neg_cross_m = self.text_encoder_m.bert(
                        encoder_embeds=text_embeds_all_m,
                        attention_mask=text_atts_all_m,
                        encoder_hidden_states=image_embeds_all_m,
                        encoder_attention_mask=image_atts_all_m,
                        return_dict=True,
                        mode='fusion',
                    )

                    vl_embeddings_m = torch.cat([
                        output_pos_m.last_hidden_state[:, 0, :],
                        output_neg_cross_m.last_hidden_state[:, 0, :]
                    ], dim=0)
                    vl_output_m = self.itm_head_m(vl_embeddings_m)

                    # 老师的 soft label（2 类 softmax 概率）
                    soft_targets = F.softmax(vl_output_m, dim=-1)

                # 跟 CL 一样：teacher 分布 + one-hot 伪标签做 convex combination
                hard_targets = F.one_hot(
                    itm_labels, num_classes=vl_output.size(-1)
                ).float()
                mixed_targets = alpha * soft_targets + (1.0 - alpha) * hard_targets

                log_probs = F.log_softmax(vl_output, dim=-1)
                loss_itm = -(mixed_targets * log_probs).sum(dim=-1).mean()
            else:
                # 原来的硬标签 CE
                loss_itm = F.cross_entropy(vl_output, itm_labels)

            loss_dict['loss_itm'] = loss_itm

        else:
            # still compute output_pos for InfMasking alignment if needed
            output_pos = self.text_encoder.bert(
                encoder_embeds=text_embeds,
                attention_mask=text_atts,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                mode='fusion',
            )

                # ===== InfMasking (synergy) =====
        enable_infmask = bool(config.get('enable_infmask_loss', False))
        if enable_infmask:
            # === Teacher（动量塔）作为 full 视图对齐目标 ===
            with torch.no_grad():
                self._momentum_update()
                image_embeds_m_t = self.visual_encoder_m(image1)
                image_atts_m_t  = torch.ones(
                    image_embeds_m_t.size()[:-1],
                    dtype=torch.long,
                    device=image1.device,
                )
                output_pos_m = self.text_encoder_m.bert(
                    encoder_embeds=text_embeds,             # 与 student 相同的文本 token
                    attention_mask=text_atts,
                    encoder_hidden_states=image_embeds_m_t, # 老师的视觉编码
                    encoder_attention_mask=image_atts_m_t,
                    return_dict=True,
                    mode='fusion',
                )
                z_full = output_pos_m.last_hidden_state[:, 0, :]  # [B, D]

            B = z_full.size(0)
            device = z_full.device
            neg_filter = None
            if bool(config.get('infmask_filter_negatives', True)):
                same_id = torch.eq(idx.view(-1, 1), idx.view(1, -1))
                not_diag = ~torch.eye(B, dtype=torch.bool, device=device)
                neg_filter = (same_id & not_diag)

                if bool(config.get('infmask_use_knn_filter', False)):
                    k = int(config.get('infmask_knn_k', 3))
                    with torch.no_grad():
                        z = F.normalize(z_full.detach(), dim=-1)
                        sim = z @ z.t()
                        sim = sim - torch.eye(B, device=device) * 1e9
                        k = min(k, max(1, B - 1))
                        nbr = sim.topk(k=k, dim=1).indices
                        knn = torch.zeros(B, B, dtype=torch.bool, device=device)
                        for i in range(B):
                            knn[i, nbr[i]] = True
                        mutual = knn & knn.t()
                    neg_filter = neg_filter | (mutual & not_diag)

            sal_text = None
            if config.get('infmask_use_saliency', False) and 'saliency' in locals():
                sal_text = saliency

            loss_infmask = self.compute_infmask_loss(
                image=image1,                                   # ★ 新增：原始图像
                text_ids=text2['input_ids'],                    # ★ 新增：文本 token
                image_embeds=image_embeds.detach(),
                text_embeds=text_embeds.detach(),
                image_atts=image_atts,
                text_atts=text_atts,
                z_full=z_full,
                config=config,
                epoch=epoch,
                saliency_text=sal_text,
                saliency_image=None,
                neg_filter=neg_filter,
            )
            loss_dict['loss_infmask'] = loss_infmask



        # ===== Optional sim alignment =====
        enable_sim_loss = bool(config.get('enable_sim_loss', False))
        if enable_sim_loss:
            input_sim = text1.input_ids.clone()
            labels_sim = input_sim.clone()
            input_sim, labels_sim = self.mask(
                input_sim,
                self.text_encoder.config.vocab_size,
                targets=labels_sim,
                probability_matrix=(probability_matrix if probability_matrix is not None else None)
            )
            masked_text_out = self.text_encoder.bert(
                input_ids=input_sim,
                attention_mask=text1['attention_mask'],
                return_dict=True,
                mode='text',
            )
            masked_text_embeds = masked_text_out.last_hidden_state

            image2_embeds = self.visual_encoder(image2)
            image2_atts = torch.ones(image2_embeds.size()[:-1], dtype=torch.long, device=image2.device)

            fused_masked = self.text_encoder.bert(
                encoder_embeds=masked_text_embeds,
                attention_mask=text1['attention_mask'],
                encoder_hidden_states=image2_embeds,
                encoder_attention_mask=image2_atts,
                return_dict=True,
                mode='fusion',
            )

            cls_itm_pos = output_pos.last_hidden_state[:, 0, :]
            cls_masked  = fused_masked.last_hidden_state[:, 0, :]

            if bool(config.get('sim_anchor_stop_grad', True)):
                cls_itm_pos = cls_itm_pos.detach()

            z_a = F.normalize(cls_itm_pos, dim=-1)
            z_b = F.normalize(cls_masked,  dim=-1)
            loss_dict['loss_sim'] = (1.0 - (z_a * z_b).sum(dim=-1)).mean()

        return loss_dict