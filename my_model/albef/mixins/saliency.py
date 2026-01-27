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
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        返回 [B, L] 的概率矩阵，偏向显著性高的 token（strong group）。
        目标：每个样本的期望 mask 数接近 base_prob * n_valid，并在 [p_min, p_max] 约束下尽量守恒。

        旧版本 bug：
          - 当 p_weak < p_min 时，直接令 strong=E/ns, weak=p_min，会导致期望 E' = E + p_min*nw，系统性过量 mask。
          - clamp 后不重算另一侧，也会破坏期望守恒。

        新版本策略：
          1) 选 top focus_top_p 为 strong 组
          2) 先固定 p_strong 解 p_weak；若越界则固定 p_weak 到边界，反解 p_strong
          3) 若边界约束导致无解或误差过大，兜底为全体均匀概率 p=E/n_valid（再 clamp）
        """
        device = saliency.device
        B, L = saliency.shape
        if base_prob is None:
            base_prob = float(getattr(self, "mlm_probability", 0.15))

        # 1) maskable：有效 token & 非 special
        maskable = attention_mask.bool().clone()
        for sp_id in [
            getattr(self.tokenizer, "pad_token_id", None),
            getattr(self.tokenizer, "cls_token_id", None),
            getattr(self.tokenizer, "sep_token_id", None),
        ]:
            if sp_id is not None:
                maskable &= (input_ids != sp_id)

        probs = torch.zeros((B, L), device=device, dtype=torch.float32)

        # 工具：clamp float
        def _clamp(x: float) -> float:
            return float(max(p_min, min(p_max, x)))

        for b in range(B):
            valid_pos = maskable[b]
            n_valid = int(valid_pos.sum().item())
            if n_valid <= 0:
                continue

            # 期望 mask 数
            E = float(base_prob) * float(n_valid)

            # 2) 选 strong 候选（topk in valid positions）
            sal_b = saliency[b].float().clone()
            sal_b[~valid_pos] = -1e9
            k = int(round(n_valid * float(focus_top_p)))
            k = max(1, min(k, n_valid))

            _, topk_idx = torch.topk(sal_b, k, dim=-1, largest=True, sorted=False)
            strong_mask = torch.zeros(L, dtype=torch.bool, device=device)
            strong_mask[topk_idx] = True
            strong_mask &= valid_pos

            ns = int(strong_mask.sum().item())
            nw = int(n_valid - ns)

            # 兜底：如果 ns==0 或 nw==0，直接均匀/只在一侧求解
            if ns <= 0:
                pU = _clamp(E / max(1.0, float(n_valid)))
                probs[b, valid_pos] = pU
                continue

            if nw <= 0:
                # 全是 strong
                pS = _clamp(E / max(1.0, float(ns)))
                probs[b, strong_mask] = pS
                continue

            # 3) 先固定 p_strong 解 p_weak
            pS = float(p_strong)
            pW = (E - pS * ns) / max(1.0, float(nw))

            # 若 pW 越界：固定 pW 到边界，反解 pS（保证守恒）
            if pW < p_min:
                pW = float(p_min)
                pS = (E - pW * nw) / max(1.0, float(ns))
            elif pW > p_max:
                pW = float(p_max)
                pS = (E - pW * nw) / max(1.0, float(ns))

            # clamp 后检查是否还能接近期望
            pS_c = _clamp(pS)
            pW_c = _clamp(pW)
            E_hat = pS_c * ns + pW_c * nw

            # 如果误差过大（通常是边界约束导致无解），兜底均匀概率
            # 误差阈值：相对 E 或绝对很小情况下用绝对阈值
            tol = max(1e-4 * max(1.0, E), 1e-3)
            if abs(E_hat - E) > tol:
                pU = _clamp(E / max(1.0, float(n_valid)))
                probs[b, valid_pos] = pU
            else:
                probs[b, strong_mask] = pS_c
                probs[b, valid_pos & (~strong_mask)] = pW_c

        probs[~maskable] = 0.0
        return probs