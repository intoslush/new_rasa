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
        
        if bool(config.get('collect_stats', True)) and probability_matrix is not None:
             self.dump_global_mask_stats(
                input_ids=text1['input_ids'],
                probability_matrix=probability_matrix,
                step=batch.get("global_step", 0),
                sample_ratio=1 # 调节采样率
             )
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

        return loss_dict

    