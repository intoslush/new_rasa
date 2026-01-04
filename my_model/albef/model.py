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
        use_momentum   = bool(config.get('use_momentum', True))   # <-- 新增：控制是否用动量模型

        if enable_cl_loss:
            idx = idx.view(-1, 1)
            idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

            with torch.no_grad():
                if use_momentum:
                    # ---- 动量分支：和你原来一致 ----
                    self._momentum_update()

                    image_embeds_m = self.visual_encoder_m(image2)
                    image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
                    image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

                    text_output_m = self.text_encoder_m.bert(
                        text2['input_ids'],
                        attention_mask=text2['attention_mask'],
                        return_dict=True,
                        mode='text'
                    )
                    text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
                    text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

                    sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                    sim_t2i_m = text_feat_m @ image_feat_all / self.temp
                    sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                    sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

                else:
                    # ---- 非动量分支：用当前特征( detach )做 soft target + 队列对比 ----
                    image_feat_all = torch.cat([image_feat.detach().t(), self.image_queue.clone().detach()], dim=1)
                    text_feat_all  = torch.cat([text_feat.detach().t(),  self.text_queue.clone().detach()], dim=1)

                    sim_i2t_c = image_feat.detach() @ text_feat_all / self.temp
                    sim_t2i_c = text_feat.detach()  @ image_feat_all / self.temp
                    sim_i2t_targets = sim_targets  # alpha * F.softmax(sim_i2t_c, dim=1) + (1 - alpha) * sim_targets
                    sim_t2i_targets = sim_targets  #alpha * F.softmax(sim_t2i_c, dim=1) + (1 - alpha) * sim_targets

            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ image_feat_all / self.temp

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            loss_dict['loss_cl'] = (loss_i2t + loss_t2i) / 2

            # 队列更新：动量用 m 特征；非动量用当前特征（detach）
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
                    saliency_image=saliency_image,  # 用已有的 patch 显著性
                    layers=int(config.get('saliency_layers', 3)),
                    use_entropy=bool(config.get("grounded_use_entropy", True)),
                    use_patch_saliency=bool(config.get("grounded_use_patch_saliency", True)),
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
            if epoch >= debug_epoch and bool(config.get("debug_log_saliency", True)):
                # 注意：这里用 text1（被 MLM mask 的那份）做 saliency 更直观
                with torch.no_grad():
                    sal_layers = int(config.get("debug_saliency_layers", 3))
                    #这是使用范数的方案
                    sal_norm, layer_deltas, layer_indices = self.compute_cross_modal_saliency(
                        text_ids=text1['input_ids'],
                        attention_mask=text1['attention_mask'],
                        image_embeds=image_embeds,
                        image_atts=image_atts,
                        layers=sal_layers,
                        return_layer_deltas=True,
                    )

                # 你要输出到当前目录 mask_output.txt
                #范数方案的debug
                self.debug_render_mask_with_norms(
                    epoch=int(epoch),
                    step=int(batch.get("global_step", 0)),   # 没有就传 n_iter/全局计数
                    input_ids_before=ids_before_debug,
                    input_ids_after=input_ids,
                    targets=labels,
                    attention_mask=text1['attention_mask'],
                    probability_matrix=(probability_matrix if probability_matrix is not None else None),
                    saliency_norm=sal_norm,
                    layer_deltas=layer_deltas,
                    layer_indices=layer_indices,
                    raw_texts=batch.get('caption1', None),
                    out_path=str(config.get("debug_mask_file", "./mask_output.txt")),
                    limit_per_epoch=int(config.get("debug_mask_limit_per_epoch", 30)),
                    sample_per_step=int(config.get("debug_sample_per_step", 2)),
                    topk_tokens=int(config.get("debug_topk_tokens", 8)),
                    step_prob=float(config.get("debug_step_prob", 0.15)),
                )
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
                    layer_deltas=layer_deltas,
                    layer_indices=layer_indices,
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

            # --- 学生：相似度采负样本（保持原逻辑） ---
            with torch.no_grad():
                bs = image1.size(0)
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

            image_neg_idx = torch.multinomial(weights_t2i, 1).flatten()
            image_embeds_neg = image_embeds[image_neg_idx]
            text_neg_idx = torch.multinomial(weights_i2t, 1).flatten()
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

            # CLS 拼接：前 bs 个是正样本
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
        else:
            # 仍然要给 output_pos 一个定义，供后续 InfMask / consistency 使用
            output_pos = self.text_encoder.bert(
                encoder_embeds=text_embeds,
                attention_mask=text_atts,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                mode='fusion',
            )
            vl_output_pos_full = None
            
        # ===== SoftMask ITM (positive-only extra branch) =====
        if enable_itm_softmask:
            # 一些超参
            sm_weight = float(config.get('itm_softmask_weight', 1.0))
            sm_beta   = float(config.get('itm_softmask_beta', 0.4))      # groundedness 引导强度
            sm_layers = int(config.get('itm_softmask_gcam_layers', 3))   # gcam聚合最后几层

            # 计算 softmask loss（只用正样本）
            loss_itm_sm = self.compute_itm_softmask_loss(
                text_embeds=text_embeds,                 # [B,Lt,D]
                text_atts=text_atts,
                text_ids=text2['input_ids'],
                image_embeds=image_embeds,               # [B,Lv,D]
                image_atts=image_atts,
                output_pos=output_pos,                   # 含 cross_attentions
                itm_head=self.itm_head,
                gcam_layers=sm_layers,
                beta=sm_beta,
                weight=sm_weight,
            )
            loss_dict['loss_itm_softmask'] = loss_itm_sm

        # ===== ITM consistency：masked 视图与 full 视图对齐 =====
        enable_itm_cons = bool(config.get('enable_itm_consistency', False))
        if enable_itm_loss and enable_itm_cons and (vl_output_pos_full is not None):
            bs = image1.size(0)
            device = image1.device

            # keep 比例可以单独配，尽量不要太狠
            keep_t_ratio = float(config.get('itm_cons_keep_t', 0.6))
            keep_v_ratio = float(config.get('itm_cons_keep_v', 0.6))
            min_keep_t = int(config.get('itm_cons_min_keep_t', 3))
            min_keep_v = int(config.get('itm_cons_min_keep_v', 3))

            use_saliency = bool(config.get('infmask_use_saliency', True))
            sal_text_for_itm = saliency if (use_saliency and 'saliency' in locals()) else None
            sal_img_for_itm = saliency_image if use_saliency else None
            saliency_phase = str(config.get('infmask_saliency_phase', 'none'))

            # --- 文本 keep mask（对 text2） ---
            B, L_t = text2['input_ids'].shape
            kv_keep_mask = self._infmask_build_keep_mask(
                B=B,
                L=L_t,
                keep_ratio=keep_t_ratio,
                min_keep=min_keep_t,
                device=device,
                must_keep_cls=True,
                saliency=sal_text_for_itm,
                saliency_phase=saliency_phase,
                valid_mask=text_atts.bool(),
            )

            # --- 图像 keep mask（对 image1 的 token） ---
            _, L_v, _ = image_embeds.shape
            q_keep_mask = self._infmask_build_keep_mask(
                B=B,
                L=L_v,
                keep_ratio=keep_v_ratio,
                min_keep=min_keep_v,
                device=device,
                must_keep_cls=True,
                saliency=sal_img_for_itm,
                saliency_phase=saliency_phase,
                valid_mask=image_atts.bool(),
            )

            # 输入级别 masking
            text_ids_m, text_atts_m = self._infmask_apply_text_input_mask(
                text_ids=text2['input_ids'],
                text_atts=text_atts,
                keep_mask=kv_keep_mask,
                mask_token_id=self.tokenizer.mask_token_id,
            )
            image_m, image_atts_m = self._infmask_apply_image_input_mask(
                image=image1,
                image_embeds=image_embeds,
                image_atts=image_atts,
                keep_mask=q_keep_mask,
            )

            # 重新编码 masked 文本 / 图像
            text_out_m = self.text_encoder.bert(
                text_ids_m,
                attention_mask=text_atts_m,
                return_dict=True,
                mode='text',
            )
            text_embeds_m = text_out_m.last_hidden_state
            image_embeds_m = self.visual_encoder(image_m)

            # 融合得到 masked 视图 CLS
            output_pos_mask = self.text_encoder.bert(
                encoder_embeds=text_embeds_m,
                attention_mask=text_atts_m,
                encoder_hidden_states=image_embeds_m,
                encoder_attention_mask=image_atts_m,
                return_dict=True,
                mode='fusion',
            )
            cls_mask = output_pos_mask.last_hidden_state[:, 0, :]
            logits_mask = self.itm_head(cls_mask)   # [bs, 2]

            # self-teacher：full 视图的正样本 logits，stop-grad
            T_cons = float(config.get('itm_cons_temp', 1.0))
            with torch.no_grad():
                p_t = F.softmax(vl_output_pos_full / T_cons, dim=-1)

            log_p_s = F.log_softmax(logits_mask / T_cons, dim=-1)
            loss_itm_cons = F.kl_div(log_p_s, p_t, reduction='batchmean') * (T_cons * T_cons)
            loss_dict['loss_itm_cons'] = loss_itm_cons
            
            
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

            # 文本显著性
            sal_text = saliency if ('saliency' in locals()) else None

            # 图像显著性：前面已经算好 saliency_image
            sal_img = saliency_image

            loss_infmask = self.compute_infmask_loss(
                image=image1,
                text_ids=text2['input_ids'],
                image_embeds=image_embeds.detach(),
                text_embeds=text_embeds.detach(),
                image_atts=image_atts,
                text_atts=text_atts,
                z_full=z_full,
                config=config,
                epoch=epoch,
                saliency_text=sal_text,
                saliency_image=sal_img,
                neg_filter=neg_filter,
            )
            loss_dict['loss_infmask'] = loss_infmask

        return loss_dict