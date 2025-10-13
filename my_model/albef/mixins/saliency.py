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
        attn_list = out.cross_attentions[-layers:]
        attn = torch.stack(attn_list, dim=0).mean(0)  # [B, H, L, S]
        sal = attn.mean(1).sum(-1)                    # [B, L]

        # 只保留有效 token，并按样本缩放到 [0,1]
        sal = sal * attention_mask  # PAD 位置为 0
        sal_min = sal.masked_fill(attention_mask == 0, 1e9).amin(dim=1, keepdim=True)
        sal_min = torch.where(torch.isinf(sal_min), torch.zeros_like(sal_min), sal_min)
        sal_max = sal.amax(dim=1, keepdim=True)
        denom = (sal_max - sal_min).clamp(min=1e-6)
        sal_norm = ((sal - sal_min) / denom) * attention_mask
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
