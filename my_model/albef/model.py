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
        idx = batch['image_id']#覆盖为图像 id，保持原逻辑：同一图的多条描述共享一个 id 
        # idx = batch['pseudo_label']  # 覆盖为伪标签，保持原逻辑
        

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
        
        # ===== Contrastive loss =====
        enable_cl_loss = bool(config.get('enable_cl_loss', True))
        use_momentum   = bool(config.get('use_momentum', True))   # 是否用动量分支
        use_queue      = bool(config.get('use_queue', True))      # <-- 新增：是否使用队列做对比

        if enable_cl_loss:
            idx = idx.view(-1, 1)

            # -------- 1) 构造对比的候选集合：batch-only 或 batch+queue --------
            if use_queue:
                idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)  # [1, B+Q]
            else:
                idx_all = idx.t()  # [1, B]

            pos_idx = torch.eq(idx, idx_all).float()  # [B, B(+Q)]
            sim_targets = pos_idx / (pos_idx.sum(1, keepdim=True) + 1e-8)

            # -------- 2) 计算 soft targets（可选动量） --------
            with torch.no_grad():
                if use_momentum:
                    self._momentum_update()

                    image_embeds_m = self.visual_encoder_m(image2)
                    image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)

                    text_output_m = self.text_encoder_m.bert(
                        text2['input_ids'],
                        attention_mask=text2['attention_mask'],
                        return_dict=True,
                        mode='text'
                    )
                    text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)

                    if use_queue:
                        image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)  # [D, B+Q]
                        text_feat_all  = torch.cat([text_feat_m.t(),  self.text_queue.clone().detach()],  dim=1)  # [D, B+Q]
                    else:
                        image_feat_all = image_feat_m.t()  # [D, B]
                        text_feat_all  = text_feat_m.t()   # [D, B]

                    sim_i2t_m = image_feat_m @ text_feat_all / self.temp  # [B, B(+Q)]
                    sim_t2i_m = text_feat_m @ image_feat_all / self.temp  # [B, B(+Q)]

                    # 这里保持你原始 ALBEF 的蒸馏式 target
                    sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                    sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

                else:
                    # 非动量：targets 直接用 sim_targets（你当前就是这么做的）
                    if use_queue:
                        image_feat_all = torch.cat([image_feat.detach().t(), self.image_queue.clone().detach()], dim=1)
                        text_feat_all  = torch.cat([text_feat.detach().t(),  self.text_queue.clone().detach()],  dim=1)
                    else:
                        image_feat_all = image_feat.detach().t()
                        text_feat_all  = text_feat.detach().t()

                    sim_i2t_targets = sim_targets
                    sim_t2i_targets = sim_targets

            # -------- 3) 用在线特征算 logits（注意：这里也要跟 use_queue 同步） --------
            if use_queue:
                # 注意：若 use_momentum=False，上面 image_feat_all/text_feat_all 已经是 detach 版本+queue
                # 若 use_momentum=True，上面 image_feat_all/text_feat_all 是 m 版本+queue（detached）
                # 这里对齐：logits 永远用 online feat 对 candidate 集合
                # candidate 集合应是：text_feat_all / image_feat_all（同维度）
                pass
            else:
                # batch-only: candidate 集合就是当前 batch 的 text_feat/image_feat
                # 为了避免混淆，直接重置为 online 的 batch-only（不依赖上面 no_grad 分支）
                image_feat_all = image_feat.t()
                text_feat_all  = text_feat.t()

            # logits
            sim_i2t = image_feat @ text_feat_all / self.temp  # [B, B(+Q)]
            sim_t2i = text_feat @ image_feat_all / self.temp  # [B, B(+Q)]

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            loss_dict['loss_cl'] = (loss_i2t + loss_t2i) / 2

            # -------- 4) 队列更新：只有 use_queue=True 才更新 --------
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
        if enable_itm_loss:
            # --- 学生：正样本 (text2, image1) ---
            output_pos = self.text_encoder.bert(
                encoder_embeds=text_embeds,
                attention_mask=text_atts,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                mode='fusion',
                output_attentions=enable_itm_softmask,  # 只有 softmask 才开
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
                    # 沿用原本：按 CL 相似度分布采样
                    # 如果 enable_cl_loss=False，你原代码这里会现算 sim；保持不变
                    if not enable_cl_loss:
                        sim_i2t = image_feat @ text_feat.t()  # [B,B]
                        sim_t2i = text_feat @ image_feat.t()  # [B,B]

                    weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)  # [B,B]
                    weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)  # [B,B]
                    weights_i2t.masked_fill_(mask_same, 0.0)
                    weights_t2i.masked_fill_(mask_same, 0.0)

                    # 兜底：若某行全 0（极端情况下同 id 太多），退化为随机（排除自身）
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
                    # random：均匀随机采样，排除同 id（含自身）
                    valid = (~mask_same).float()  # [B,B]
                    row_sum = valid.sum(dim=1, keepdim=True)

                    # 若某行无可选（整批同 id），退化为排除自身
                    if (row_sum == 0).any():
                        valid_fallback = (~torch.eye(bs, device=image1.device, dtype=torch.bool)).float()
                        valid = torch.where(row_sum > 0, valid, valid_fallback)
                        row_sum = valid.sum(dim=1, keepdim=True)

                    probs = valid / (row_sum + 1e-12)

                    image_neg_idx = torch.multinomial(probs, 1).squeeze(1)  # [B]
                    text_neg_idx  = torch.multinomial(probs, 1).squeeze(1)  # [B]


            
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

            vl_embeddings = torch.cat([
                output_pos.last_hidden_state[:, 0, :],              # [bs, D]
                output_neg_cross.last_hidden_state[:, 0, :],        # [2*bs, D]
            ], dim=0)                                              # [3*bs, D]
            vl_output = self.itm_head(vl_embeddings)               # [3*bs, 2]

            itm_labels = torch.cat([
                torch.ones(bs, dtype=torch.long),
                torch.zeros(2 * bs, dtype=torch.long)
            ], dim=0).to(image1.device)

            # 纯硬标签 CE
            loss_itm = F.cross_entropy(vl_output, itm_labels)
            loss_dict['loss_itm'] = loss_itm

            # 正样本 logits，用于后面的 ITM consistency
            vl_output_pos_full = vl_output[:bs].detach()


        return loss_dict