from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
from my_model.vit import VisionTransformer
from my_model.xbert import BertConfig, BertForMaskedLM
import os
class ALBEF(nn.Module):
    def __init__(self, text_encoder=None, tokenizer=None, config=None):
        super().__init__()

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
        self.prd_head = nn.Linear(self.text_width, 2)
        self.mrtd_head = nn.Linear(self.text_width, 2)

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
        ]
        self.copy_params()

        # Queues
        self._init_queues(embed_dim)

    def forward(self, batch,alpha,config,epoch):#其中text2是概率同一个id的其他图片的描述,img1和img2是同一个图片的两个不同的增广
        # extract image features
        # image1, image2, text1, text2, alpha, idx, replace
        image1=batch['image1']
        image2=batch['image2']
        text1=self.tokenizer(batch['caption1'], padding='longest', max_length=config['max_words'], return_tensors="pt").to(image1.device)
        text2=self.tokenizer(batch['caption2'], padding='longest', max_length=config['max_words'], return_tensors="pt").to(image1.device)
        text_atts= text2['attention_mask']
        idx=batch['person_id']
        replace=batch['replace_flag']
        # pseudo_label=batch['pseudo_label']
        idx=batch['pseudo_label']
        image_embeds = self.visual_encoder(image1)#(13,577,768)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image1.device)#注意力掩码全一表示所有图像token都应该被关注
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)#用于取cls token的特征,shape(13,577)
        # extract text features
        text_output = self.text_encoder.bert(text2['input_ids'], attention_mask=text2['attention_mask'],
                                             return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        
        
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)#同样是取cls token的特征
        # Contrastive loss
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)#(13,65549)
        pos_idx = torch.eq(idx, idx_all).float()#(13,65549)
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)#归一化,做出样本标签队列,(13,65549)
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image2)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            text_output_m = self.text_encoder_m.bert(text2['input_ids'], attention_mask=text2['attention_mask'],
                                                     return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp#计算当前批次与队列的相似得分
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets#用来让标签匹配变软
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp


        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_cl = (loss_i2t + loss_t2i ) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

        # forward the positve image-text pairs
        # === Masked Language Modeling ===
        
        if epoch>4:
            # === 1) 计算跨模态显著性（用 image1 与 text1 对齐）===
            with torch.no_grad():
                saliency = self.compute_cross_modal_saliency(
                    text_ids=text1['input_ids'],
                    attention_mask=text1['attention_mask'],
                    image_embeds=image_embeds,    # 来自 image1 的编码
                    image_atts=image_atts,
                    layers=config.get('saliency_layers', 3),
                )
            # === 2) 基于显著性的（无阶段）强相关优先掩码概率 ===
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
            probability_matrix=None
        input_ids = text1.input_ids.clone()
        labels = input_ids.clone()
        ids_before_debug=input_ids.clone()
        input_ids, labels = self.mask(
            input_ids,
            self.text_encoder.config.vocab_size,
            targets=labels,
            probability_matrix=None#probability_matrix
        )
        if epoch > 35:
            # config['debug_mask_file'] = "./mask_debug/all_epochs.txt"
            self.debug_render_mask_diff(
                epoch=epoch,
                input_ids_before=ids_before_debug,
                input_ids_after=input_ids,
                targets=labels,
                attention_mask=text1['attention_mask'],
                raw_texts=batch.get('caption1', None),
                limit_per_epoch=int(config.get('debug_mask_limit_per_epoch', 50)),
                out_path=config.get('debug_mask_file', None),  # <== 单文件路径
            )
        
        # 前向传播：不使用 soft label，只使用 hard label
        mlm_output = self.text_encoder(
            input_ids,
            attention_mask=text1.attention_mask,
            encoder_hidden_states=image_embeds,            # 图文融合
            encoder_attention_mask=image_atts,
            return_dict=True,
            labels=labels,                                   # 监督目标
        )

        # 获取标准的交叉熵 loss
        loss_mlm = mlm_output.loss

        #两个模态融合的部分
        output_pos = self.text_encoder.bert(encoder_embeds=text_embeds,
                                            attention_mask=text_atts,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts,
                                            return_dict=True,
                                            mode='fusion',
                                            )
        with torch.no_grad():
            bs = image1.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)
            
            mask = torch.eq(idx, idx.T)
            if idx.shape[0] <= 2:
                raise ValueError("Batch size too small, idx.shape[0] = {}".format(idx.shape[0]))
            weights_i2t.masked_fill_(mask, 0)#通过掩码保证不会选正样本作为难样本
            weights_t2i.masked_fill_(mask, 0)
            # weights_t2i = weights_t2i + 1e-8
            # weights_i2t = weights_i2t + 1e-8
        # select a negative image for each text
        
        image_neg_idx = torch.multinomial(weights_t2i, 1).flatten()
        image_embeds_neg = image_embeds[image_neg_idx]#难的负样本
        # select a negative text for each image
        text_neg_idx = torch.multinomial(weights_i2t, 1).flatten()
        text_embeds_neg = text_embeds[text_neg_idx]
        text_atts_neg = text_atts[text_neg_idx]
        # forward the negative image-text pairs
        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)
        output_neg_cross = self.text_encoder.bert(encoder_embeds=text_embeds_all,
                                                  attention_mask=text_atts_all,
                                                  encoder_hidden_states=image_embeds_all,
                                                  encoder_attention_mask=image_atts_all,
                                                  return_dict=True,
                                                  mode='fusion',
                                                  )
        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg_cross.last_hidden_state[:, 0, :]],
                                  dim=0)
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image1.device)
        #与融合后的标签直接做交叉熵
        loss_pitm = F.cross_entropy(vl_output, itm_labels)
        ############新增的loss#############################

        input_sim = text1.input_ids.clone()
        labels_sim = input_sim.clone()
        input_sim, labels_sim = self.mask(
            input_sim,
            self.text_encoder.config.vocab_size,
            targets=labels_sim,
            probability_matrix=probability_matrix
        )
        # ===== 新增：融合特征对齐（image2 + masked text1） vs (ITM 正样本融合) =====
        # 1) 取 mask 后的 text1 表示（上面你已经得到 input_ids/labels，这里直接复用 input_ids）
        masked_text_out = self.text_encoder.bert(
            input_ids=input_sim,                               # 已被 mask 的 text1
            attention_mask=text1['attention_mask'],
            return_dict=True,
            mode='text',
        )
        masked_text_embeds = masked_text_out.last_hidden_state  # [B, L, H_text]

        # 2) 编码 image2（主干，不用 momentum，方便反传）
        image2_embeds = self.visual_encoder(image2)             # [B, S_img, H_vision]
        image2_atts = torch.ones(image2_embeds.size()[:-1], dtype=torch.long, device=image2.device)

        # 3) 融合：(image2 + masked text1)
        fused_masked = self.text_encoder.bert(
            encoder_embeds=masked_text_embeds,
            attention_mask=text1['attention_mask'],
            encoder_hidden_states=image2_embeds,
            encoder_attention_mask=image2_atts,
            return_dict=True,
            mode='fusion',
        )

        # 4) 取两侧的 [CLS] 融合特征
        cls_itm_pos = output_pos.last_hidden_state[:, 0, :]     # ITM 正样本融合 (image1 + text2)
        cls_masked  = fused_masked.last_hidden_state[:, 0, :]   # (image2 + masked text1)

        # 5) 余弦相似对齐损失（1 - cos），默认不反传到 ITM 分支，避免相互挤压
        if bool(config.get('sim_anchor_stop_grad', True)):
            cls_itm_pos = cls_itm_pos.detach()

        z_a = F.normalize(cls_itm_pos, dim=-1)
        z_b = F.normalize(cls_masked,  dim=-1)
        loss_sim = (1.0 - (z_a * z_b).sum(dim=-1)).mean()


        return {"loss_cl":loss_cl, "loss_pitm":loss_pitm, "loss_mlm":loss_mlm, "loss_prd":torch.tensor(0), "loss_mrtd":torch.tensor(0),"loss_sim": loss_sim,}

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch._dynamo.disable
    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
    
    @torch._dynamo.disable
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        # gather keys before updating queue
        if torch.distributed.is_initialized():
            image_feats = concat_all_gather(image_feat)
            text_feats = concat_all_gather(text_feat)
            idxs = concat_all_gather(idx)
        else:
            image_feats = image_feat
            text_feats = text_feat
            idxs = idx
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        empty = self.image_queue.size(1) - ptr#队列长和指针
        if batch_size <= empty:
            self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
            self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        else:
            self.image_queue[:, ptr:] = image_feats[:empty].T
            self.text_queue[:, ptr:] = text_feats[:empty].T
            self.idx_queue[:, ptr:] = idxs[:empty].T
            self.image_queue[:, :batch_size - empty] = image_feats[empty:].T
            self.text_queue[:, :batch_size - empty] = text_feats[empty:].T
            self.idx_queue[:, :batch_size - empty] = idxs[empty:].T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    def mask(self, input_ids, vocab_size, targets=None, masked_indices=None, probability_matrix=None):
        device = input_ids.device

        # 保证概率矩阵与 input_ids 在同一设备
        if probability_matrix is None:
            prob = torch.full(input_ids.shape, self.mlm_probability, device=device, dtype=torch.float32)
        else:
            prob = probability_matrix.to(device=device, dtype=torch.float32)

        if masked_indices is None:
            masked_indices = torch.bernoulli(prob).to(dtype=torch.bool, device=device)

        # 不 mask PAD/CLS
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        masked_indices[input_ids == self.tokenizer.sep_token_id] = False

        if targets is not None:
            targets = targets.to(device)
            targets[~masked_indices] = -100  # 只对被 mask 的位置算 loss

        # 80%：换成 [MASK]
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, 0.8, device=device, dtype=torch.float32)
        ).to(torch.bool) & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10%：换成随机词
        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device, dtype=torch.float32))
            .to(torch.bool) & masked_indices & ~indices_replaced
        )
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long, device=device)
        input_ids[indices_random] = random_words[indices_random]

        # 剩下 10%：保留原词
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


    
    def _build_vit(self, img_size):
        return VisionTransformer(
            img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

    def _init_queues(self, embed_dim):
        self.register_buffer("image_queue", nn.functional.normalize(torch.randn(embed_dim, self.queue_size), dim=0))
        self.register_buffer("text_queue", nn.functional.normalize(torch.randn(embed_dim, self.queue_size), dim=0))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def compute_cross_modal_saliency(
        self,
        text_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_embeds: torch.Tensor,
        image_atts: torch.Tensor,
        layers: int = 3,
    ) -> torch.Tensor:
        """
        返回 shape [B, L] 的文本 token 显著性，按样本缩放到 [0,1]。
        """
        out = self.text_encoder.bert(
            input_ids=text_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            output_attentions=True,
            return_dict=True,
            mode='multi_modal',
        )
        # cross_attentions: list[Tensor[B, H, L, S]]，取最后 layers 层
        attn_list = out.cross_attentions[-layers:]
        attn = torch.stack(attn_list, dim=0).mean(0)  # [B, H, L, S]
        sal = attn.mean(1).sum(-1)                    # [B, L] 先均头，再对图像源求和

        # 只保留有效 token，并按样本缩放到 [0,1]
        sal = sal * attention_mask  # PAD 位置为 0
        sal_min = sal.masked_fill(attention_mask == 0, 1e9).amin(dim=1, keepdim=True)
        sal_min = torch.where(torch.isinf(sal_min), torch.zeros_like(sal_min), sal_min)
        sal_max = sal.amax(dim=1, keepdim=True)
        denom = (sal_max - sal_min).clamp(min=1e-6)
        sal_norm = ((sal - sal_min) / denom) * attention_mask  # 无效位置依旧为 0
        return sal_norm
    # @torch.no_grad()
    # def build_curriculum_mask_probs(
    #     self,
    #     saliency: torch.Tensor,               # [B, L], 已在 [0,1]
    #     attention_mask: torch.Tensor,         # [B, L]
    #     epoch: int = None,                    # 保留参数以兼容旧调用，但本函数不使用
    #     total_epochs: int = None,             # 同上
    #     base_prob: float = 0.15,              # 目标整体掩码率
    #     gamma: float = 1.5,                   # >1 强化高显著性
    #     p_max: float = 0.95,                  # 上限避免极端
    #     saliency_threshold: float = 0.0,      # 低于该阈值的 token 不参与采样
    # ) -> torch.Tensor:
    #     """
    #     返回 [B, L] 概率矩阵。仅依据显著性分配，无“易/难”阶段逻辑。
    #     注意：不在此函数内强制去掉 [CLS]/[SEP]/[PAD]；这些在外部或 mask() 里处理。
    #     """
    #     device = saliency.device
    #     attn = attention_mask.to(saliency.dtype)  # {0,1}

    #     # 1) 强调高显著性；可选阈值过滤
    #     s = saliency.clamp(0.0, 1.0) * attn
    #     if saliency_threshold > 0.0:
    #         keep_high = (s >= saliency_threshold).to(s.dtype)
    #         s = s * keep_high

    #     w = (s.clamp(0, 1) ** gamma) * attn  # [B, L]

    #     # 若某样本全被过滤（和为0），退化成均匀分布在有效位上
    #     sum_w = w.sum(dim=1, keepdim=True)
    #     fallback = (sum_w <= 1e-12).to(w.dtype)
    #     w = torch.where(fallback.bool(), attn, w)
    #     sum_w = w.sum(dim=1, keepdim=True).clamp_min(1e-6)

    #     # 2) 把期望掩码数 (base_prob * #valid) 按 w 比例分配到各 token
    #     L_valid = attn.sum(dim=1, keepdim=True).clamp_min(1.0)
    #     exp_tokens = base_prob * L_valid
    #     prob = exp_tokens * (w / sum_w)

    #     # 3) 截断与掩码
    #     prob = prob.clamp(0.0, p_max) * attn

    #     return prob
    
    @torch.no_grad()
    def build_curriculum_mask_probs(
        self,
        saliency: torch.Tensor,          # [B, L]，compute_cross_modal_saliency 的输出（已[0,1] & masked by attn）
        attention_mask: torch.Tensor,    # [B, L]，1=有效
        input_ids: torch.Tensor,         # [B, L]，用于避开特殊符号
        base_prob: float = None,         # 期望整体 MLM 掩码比例，默认用 self.mlm_probability
        focus_top_p: float = 0.3,        # 仅在每条样本的显著性 top p 范围内作为“强相关候选”
        p_strong: float = 0.95,          # 强相关候选上的掩码概率（上限）
        p_min: float = 0.0,              # 概率下限
        p_max: float = 0.95,             # 概率上限（避免 1.0）
    ) -> torch.Tensor:
        """
        返回 [B, L] 的概率矩阵，偏向显著性高的 token；无阶段，仅用于最后若干 epoch。
        会自动求解非候选位置的 p_weak，使期望掩码比例接近 base_prob。
        """
        device = saliency.device
        B, L = saliency.shape
        if base_prob is None:
            base_prob = float(self.mlm_probability)

        # 可被 mask 的位置：有效 & 非特殊符号
        maskable = attention_mask.bool().clone()
        for sp_id in [getattr(self.tokenizer, "pad_token_id", None),
                    getattr(self.tokenizer, "cls_token_id", None),
                    getattr(self.tokenizer, "sep_token_id", None)]:
            if sp_id is not None:
                maskable &= (input_ids != sp_id)

        probs = torch.zeros((B, L), device=device, dtype=torch.float32)

        # 逐样本构造概率
        for b in range(B):
            valid_pos = maskable[b]                            # 允许参与 MLM 的位置
            n_valid = int(valid_pos.sum().item())
            if n_valid == 0:
                continue

            # 目标期望掩码数
            target_E = base_prob * n_valid

            # 在 valid 里按显著性降序排序，取 top_p 作为“强相关候选”
            sal_b = saliency[b].clone()
            sal_b[~valid_pos] = -1e9
            k_candidate = max(1, int(round(n_valid * focus_top_p)))
            k_candidate = min(k_candidate, n_valid)
            topk_vals, topk_idx = torch.topk(sal_b, k_candidate, dim=-1, largest=True, sorted=False)
            strong_mask = torch.zeros(L, dtype=torch.bool, device=device)
            strong_mask[topk_idx] = True
            strong_mask &= valid_pos

            n_strong = int(strong_mask.sum().item())
            if n_strong == 0:
                # 如果显著性全是 0（或都被过滤），退化为均匀概率
                p = max(p_min, min(p_max, base_prob))
                probs[b, valid_pos] = p
                continue

            # 先假设强相关候选统一给 p_strong
            # 再解非候选的 p_weak 以匹配整体期望 target_E：
            #   target_E = p_strong * n_strong + p_weak * (n_valid - n_strong)
            remain = max(0, n_valid - n_strong)
            if remain == 0:
                # 所有可 mask 的位置都在强相关集合里
                p_strong_adj = min(p_strong, target_E / max(1, n_strong))
                p_strong_adj = float(max(p_min, min(p_max, p_strong_adj)))
                probs[b, strong_mask] = p_strong_adj
                continue

            # 有非候选位置，先尝试用固定 p_strong
            p_weak = (target_E - p_strong * n_strong) / remain
            if p_weak < p_min - 1e-9:
                # 强相关已经过量，降低 p_strong 以满足期望
                p_strong_adj = target_E / n_strong
                p_strong_adj = float(max(p_min, min(p_max, p_strong_adj)))
                probs[b, strong_mask] = p_strong_adj
                # 非候选直接设为 p_min（通常为 0）
                probs[b, valid_pos & (~strong_mask)] = float(p_min)
            else:
                # 正常情形：强相关用 p_strong，非候选用 p_weak（再做夹取）
                p_strong_adj = float(max(p_min, min(p_max, p_strong)))
                p_weak_adj = float(max(p_min, min(p_max, p_weak)))
                probs[b, strong_mask] = p_strong_adj
                probs[b, valid_pos & (~strong_mask)] = p_weak_adj

        # 最终再次把不可 mask 的位置置零
        probs[~maskable] = 0.0
        return probs
    

    @torch.no_grad()
    def debug_render_mask_diff(
        self,
        epoch: int,
        input_ids_before: torch.Tensor,   # [B, L]，mask() 前
        input_ids_after: torch.Tensor,    # [B, L]，mask() 后
        targets: torch.Tensor,            # [B, L]，-100 表示未 mask
        attention_mask: torch.Tensor,     # [B, L]
        raw_texts=None,                   # 可传 batch['caption1']
        limit_per_epoch: int = 50,
        out_path: str = None,             # 改为单文件路径
    ) -> None:
        """
        将原句、掩码后句子、被 mask 的词及替换情况**追加**写入同一个文件。
        分布式时仅 rank 0 写；每个 epoch 最多写 limit_per_epoch 条。
        """
        # 仅主进程写
        if torch.distributed.is_initialized():
            try:
                if torch.distributed.get_rank() != 0:
                    return
            except Exception:
                pass

        # 选择输出文件：优先参数 -> 成员属性 -> config -> 默认路径
        if out_path is None:
            out_path = getattr(self, "debug_mask_file", None)
            if out_path is None:
                out_path = getattr(self, "config_debug_mask_file", None) or "./mask_debug/mask_debug_all.txt"

        # 确保目录存在
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # 统计：每个 epoch 的已写条数
        if not hasattr(self, "_dbg_written_per_epoch"):
            self._dbg_written_per_epoch = {}  # {epoch: count}

        written = int(self._dbg_written_per_epoch.get(int(epoch), 0))
        if written >= limit_per_epoch:
            return

        B, L = input_ids_before.shape
        to_write_blocks = []

        # 若本 epoch 首次写入，则加一个 epoch 分隔头
        if written == 0:
            sep = "=" * 100
            to_write_blocks.append(f"\n{sep}\n[Epoch {int(epoch)}]  Mask Debug\n{sep}\n")

        for b in range(B):
            if written + len(to_write_blocks) - (1 if written == 0 else 0) >= limit_per_epoch:
                break

            valid = attention_mask[b].bool()
            masked_pos = (targets[b] != -100) & valid
            if masked_pos.sum().item() == 0:
                continue

            ids_before = input_ids_before[b].tolist()
            ids_after  = input_ids_after[b].tolist()

            # 句子级
            try:
                orig_sent  = self.tokenizer.decode([t for i,t in enumerate(ids_before) if valid[i]], skip_special_tokens=True)
                after_sent = self.tokenizer.decode([t for i,t in enumerate(ids_after)  if valid[i]], skip_special_tokens=True)
            except Exception:
                orig_sent  = self.tokenizer.decode(ids_before, skip_special_tokens=False)
                after_sent = self.tokenizer.decode(ids_after,  skip_special_tokens=False)

            # token 级差异
            pos_idx = torch.nonzero(masked_pos, as_tuple=False).squeeze(-1).tolist()
            if isinstance(pos_idx, int):
                pos_idx = [pos_idx]

            diff_lines = []
            for j in pos_idx:
                t0 = ids_before[j]
                t1 = ids_after[j]
                tok0 = self.tokenizer.convert_ids_to_tokens(int(t0))
                tok1 = self.tokenizer.convert_ids_to_tokens(int(t1))
                diff_lines.append(f"(pos={j}) {tok0}  ->  {tok1}")

            raw = None
            if raw_texts is not None:
                try:
                    raw = str(raw_texts[b])
                except Exception:
                    raw = None

            block = []
            block.append("-" * 80)
            block.append(f"Sample #{b}")
            if raw:
                block.append(f"RAW : {raw}")
            block.append(f"ORIG: {orig_sent}")
            block.append(f"MASK: {after_sent}")
            block.append("MASKED TOKENS:")
            for dl in diff_lines:
                block.append("  - " + dl)
            block.append("")  # 空行
            to_write_blocks.append("\n".join(block))

        if to_write_blocks:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write("\n".join(to_write_blocks) + "\n")
            # 更新本 epoch 已写条数
            # 注意：如果本次包含了 epoch 头部，那不计入条数
            added = len(to_write_blocks)
            if written == 0:
                added -= 1
            self._dbg_written_per_epoch[int(epoch)] = written + max(0, added)

        # 方便在 __init__ 里用 config 传路径（可选）
        if not hasattr(self, "config_debug_mask_file"):
            # 若从 config 里有该键，记一份，后续就能默认使用
            try:
                if hasattr(self, "tokenizer") and hasattr(self, "__dict__"):
                    pass  # 保留占位，避免静态检查告警
            except Exception:
                pass


    
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
