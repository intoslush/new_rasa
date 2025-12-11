import torch

class SaliencyMixin:
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
        显著性 = 若干个带 cross-attn 的层里，token 表征在该层“前后变化量”的平均值。
        """

        # 1. 跑一遍多模态 BERT，拿到所有层的 hidden_states
        out = self.text_encoder.bert(
            input_ids=text_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            output_hidden_states=True,   # 关键
            output_attentions=False,     # 不再需要 cross_attentions 了
            return_dict=True,
            mode='multi_modal',
        )

        hidden_states = out.hidden_states  # tuple，长度 = num_layers + 1，每个 [B, L, D]

        # 2. 找出哪些层是带 cross-attention 的层
        #    你的 BertLayer 里用 config.fusion_layer 来决定是否有 cross-attention
        if hasattr(self.text_encoder, "bert"):
            cfg = self.text_encoder.bert.config
        else:
            cfg = self.text_encoder.config

        fusion_layer = getattr(cfg, "fusion_layer", 0)
        num_layers = cfg.num_hidden_layers

        # cross-attn 层的下标：fusion_layer, fusion_layer+1, ..., num_layers-1
        cross_layer_indices = list(range(fusion_layer, num_layers))
        if len(cross_layer_indices) == 0:
            # 极端情况：没有显式 cross-attn，就退化成用最后几层的 block 差
            cross_layer_indices = list(range(max(0, num_layers - layers), num_layers))

        # 只取最后 `layers` 个 cross-attn 层
        if layers is not None and layers > 0 and layers < len(cross_layer_indices):
            cross_layer_indices = cross_layer_indices[-layers:]

        # 3. 对每个选中的层 l，算：
        #    delta_l = || h_after(l) - h_before(l) ||_2
        #    这里 h_before(l) = hidden_states[l]   （上一层输出）
        #         h_after(l)  = hidden_states[l+1] （当前层输出）
        #    注意：这包含了该层的 self-attn + cross-attn + FFN 整个 block 的效果，
        #          但我们只在“带 cross-attn 的层”上算，所以可以视为“这一层 cross-modal 交互造成的变化”。
        deltas = []
        for layer_idx in cross_layer_indices:
            h_before = hidden_states[layer_idx]     # [B, L, D]
            h_after = hidden_states[layer_idx + 1]  # [B, L, D]

            # 为了数值稳定，转成 float 做 L2 范数
            diff = (h_after - h_before).float()
            delta = diff.pow(2).sum(-1).sqrt()      # [B, L]
            deltas.append(delta)

        if len(deltas) == 0:
            # 理论上不会走到这里，保险兜底：全 0 显著性
            sal = torch.zeros_like(attention_mask, dtype=torch.float32)
        else:
            # 在所选层上做平均，得到最终 per-token saliency
            sal = torch.stack(deltas, dim=0).mean(0)   # [B, L]

        # 4. 只保留有效 token，并按样本内做 min-max 归一到 [0,1]
        sal = sal * attention_mask  # PAD 位置为 0

        # 有效 token 的最小 / 最大值（忽略 PAD）
        sal_min = sal.masked_fill(attention_mask == 0, 1e9).amin(dim=1, keepdim=True)
        sal_min = torch.where(torch.isinf(sal_min), torch.zeros_like(sal_min), sal_min)
        sal_max = sal.amax(dim=1, keepdim=True)

        denom = (sal_max - sal_min).clamp(min=1e-6)
        sal_norm = ((sal - sal_min) / denom) * attention_mask  # 再把 PAD 清 0

        return sal_norm


    @torch.no_grad()
    def build_curriculum_mask_probs(
        self,
        saliency: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        base_prob: float = None,
        focus_top_p: float = 0.3,
        p_strong: float = 0.95,
        p_min: float = 0.0,
        p_max: float = 0.95,
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

        for b in range(B):
            valid_pos = maskable[b]
            n_valid = int(valid_pos.sum().item())
            if n_valid == 0:
                continue

            target_E = base_prob * n_valid

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
                p = max(p_min, min(p_max, base_prob))
                probs[b, valid_pos] = p
                continue

            remain = max(0, n_valid - n_strong)
            if remain == 0:
                p_strong_adj = min(p_strong, target_E / max(1, n_strong))
                p_strong_adj = float(max(p_min, min(p_max, p_strong_adj)))
                probs[b, strong_mask] = p_strong_adj
                continue

            p_weak = (target_E - p_strong * n_strong) / remain
            if p_weak < p_min - 1e-9:
                p_strong_adj = target_E / n_strong
                p_strong_adj = float(max(p_min, min(p_max, p_strong_adj)))
                probs[b, strong_mask] = p_strong_adj
                probs[b, valid_pos & (~strong_mask)] = float(p_min)
            else:
                p_strong_adj = float(max(p_min, min(p_max, p_strong)))
                p_weak_adj = float(max(p_min, min(p_max, p_weak)))
                probs[b, strong_mask] = p_strong_adj
                probs[b, valid_pos & (~strong_mask)] = p_weak_adj

        probs[~maskable] = 0.0
        return probs
