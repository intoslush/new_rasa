import torch
import math

class SaliencyMixin:
    @torch.no_grad()
    def compute_cross_modal_groundedness(
        self,
        text_ids: torch.Tensor,           # [B, Lt]
        attention_mask: torch.Tensor,     # [B, Lt]
        image_embeds: torch.Tensor,       # [B, Lv, D]
        image_atts: torch.Tensor,         # [B, Lv]
        saliency_image: torch.Tensor = None,  # [B, Lv] (CLS+patch), 你的 saliency_image
        layers: int = 3,
        use_entropy: bool = True,
        use_patch_saliency: bool = True,
        eps: float = 1e-6,
        return_debug: bool = False,
    ):
        """
        返回 [B, Lt] groundedness saliency in [0,1]，越大越“视觉可接地”。
        g_fg(t) = mean_{l,h} sum_p A(t,p)*s(p)   (p 不含图像 CLS)
        可选乘以 peakiness: (1 - entropy/logP)
        """

        # 1) 跑一遍 multi-modal，并强制输出 attentions（否则拿不到 cross_attentions）
        out = self.text_encoder.bert(
            input_ids=text_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            output_attentions=True,          # ✅ 必须
            output_hidden_states=False,
            return_dict=True,
            mode='multi_modal',
        )

        cross_atts = out.cross_attentions  # tuple: each [B,H,Lt,Lv]
        if (cross_atts is None) or (len(cross_atts) == 0):
            z = torch.zeros_like(attention_mask, dtype=torch.float32)
            return (z, None) if return_debug else z

        # 2) 选最后几层 cross-attn
        if layers is not None and layers > 0 and layers < len(cross_atts):
            cross_atts_sel = cross_atts[-layers:]
        else:
            cross_atts_sel = cross_atts

        # [nL, B, H, Lt, Lv]
        A = torch.stack(cross_atts_sel, dim=0).float()

        # 3) 去掉图像 CLS，只保留 patch：Lv = 1+P
        A_patch = A[..., 1:]  # [nL,B,H,Lt,P]
        P = A_patch.size(-1)

        # 4) foreground-aware：用你的 saliency_image 给 patch 加权
        if (saliency_image is not None) and use_patch_saliency:
            s = saliency_image.float()
            # 对齐长度
            Lv = A.size(-1)
            if s.size(1) != Lv:
                s = s[:, :Lv]
            s_patch = s[:, 1:]  # [B,P]
            # 归一化（防止不同图像尺度差异）
            s_patch = s_patch.clamp(min=0.0)
            s_patch = s_patch / (s_patch.sum(dim=-1, keepdim=True) + eps)  # [B,P]
            # g_fg: sum_p A(t,p)*s(p)
            g_fg = (A_patch * s_patch.unsqueeze(0).unsqueeze(2).unsqueeze(3)).sum(dim=-1)  # [nL,B,H,Lt]
        else:
            # fallback：不用 saliency_image 时，用 max-p 作为 groundedness
            g_fg = A_patch.max(dim=-1).values  # [nL,B,H,Lt]

        # reduce over heads and layers -> [B,Lt]
        g_fg_h = g_fg.mean(dim=2)      # [nL,B,Lt]
        g_fg_l = g_fg_h.mean(dim=0)    # [B,Lt]

        peaked = None
        if use_entropy:
            # 对 entropy 用 renorm（因为去掉 CLS 后不再和为 1）
            A_norm = A_patch / (A_patch.sum(dim=-1, keepdim=True) + eps)  # [nL,B,H,Lt,P]
            ent = -(A_norm * (A_norm + eps).log()).sum(dim=-1)            # [nL,B,H,Lt]
            ent = ent / (math.log(P + eps))                                # normalize to [0,1] roughly
            peaked = (1.0 - ent).clamp(min=0.0, max=1.0)                  # [nL,B,H,Lt]
            peaked = peaked.mean(dim=2).mean(dim=0)                       # [B,Lt]
            g = g_fg_l * peaked
        else:
            g = g_fg_l

        # 5) 过滤无效 token + special token
        g = g * attention_mask.float()

        for sp_id in [getattr(self.tokenizer, "pad_token_id", None),
                      getattr(self.tokenizer, "cls_token_id", None),
                      getattr(self.tokenizer, "sep_token_id", None)]:
            if sp_id is not None:
                g = g.masked_fill(text_ids == sp_id, 0.0)

        # 6) per-sample min-max 归一化到 [0,1]
        valid = attention_mask.bool()
        g_min = g.masked_fill(~valid, 1e9).amin(dim=1, keepdim=True)
        g_min = torch.where(torch.isinf(g_min), torch.zeros_like(g_min), g_min)
        g_max = g.masked_fill(~valid, -1e9).amax(dim=1, keepdim=True)
        g_max = torch.where(torch.isinf(g_max), torch.zeros_like(g_max), g_max)
        denom = (g_max - g_min).clamp(min=eps)
        g_norm = ((g - g_min) / denom).clamp(0.0, 1.0) * valid.float()

        if not return_debug:
            return g_norm

        debug = {
            "g_fg": g_fg_l.detach(),
            "peaked": (peaked.detach() if peaked is not None else None),
            "num_layers": len(cross_atts_sel),
        }
        return g_norm, debug
    

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
