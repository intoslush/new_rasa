# my_model/albef/mixins/softmask_itm.py
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

class SoftMaskITMMixin:
    @staticmethod
    def _softmask_build_valid_word_mask(
        input_ids: torch.Tensor,          # [B, Lt]
        attention_mask: torch.Tensor,     # [B, Lt]
        pad_id: Optional[int],
        cls_id: Optional[int],
        sep_id: Optional[int],
    ) -> torch.Tensor:
        """返回 [B,Lt] bool，表示可采样的 token 位置（有效且非 special）。"""
        valid = attention_mask.bool().clone()
        if pad_id is not None:
            valid &= (input_ids != pad_id)
        if cls_id is not None:
            valid &= (input_ids != cls_id)
        if sep_id is not None:
            valid &= (input_ids != sep_id)
        return valid

    @torch.no_grad()
    def _softmask_sample_word_index(
        self,
        groundedness: torch.Tensor,       # [B, Lt], in [0,1], special 已经尽量为0也没关系
        input_ids: torch.Tensor,          # [B, Lt]
        attention_mask: torch.Tensor,     # [B, Lt]
        beta: float = 0.7,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        融合点 1：基于 groundedness 的引导随机采样。
        p = (1-beta)*Uniform(valid) + beta*Normalize(groundedness+eps on valid)
        返回 i_w: [B]，每个样本一个词 index。
        """
        B, Lt = input_ids.shape
        device = input_ids.device

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        cls_id = getattr(self.tokenizer, "cls_token_id", None)
        sep_id = getattr(self.tokenizer, "sep_token_id", None)
        valid = self._softmask_build_valid_word_mask(input_ids, attention_mask, pad_id, cls_id, sep_id)  # [B,Lt]

        # fallback：如果某些样本 valid 全空，就退化为在 attention_mask 上采样（至少不会崩）
        att_valid = attention_mask.bool()

        iw = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            v = valid[b]
            if int(v.sum().item()) == 0:
                v = att_valid[b]
                if int(v.sum().item()) == 0:
                    iw[b] = 0
                    continue

            idxs = torch.nonzero(v, as_tuple=False).flatten()  # [n]
            n = idxs.numel()
            uni = torch.full((n,), 1.0 / max(1, n), device=device)

            g = groundedness[b, idxs].float().clamp(min=0.0)
            g = g + eps
            g = g / (g.sum() + eps)

            p = (1.0 - beta) * uni + beta * g
            p = p / (p.sum() + eps)

            iw[b] = idxs[torch.multinomial(p, 1).item()]

        return iw

    def _softmask_compute_word_gcam(
        self,
        q_pos: torch.Tensor,                     # [B], positive logit
        cross_attentions: Tuple[torch.Tensor],   # tuple of [B,H,Lt,Lv]
        gcam_layers: int = 3,
    ) -> torch.Tensor:
        """
        Grad-CAM over cross-attn:
          A_gcam = mean_k mean_h relu( (d q_pos / d A_k) * A_k )
        返回 [B, Lt, Lv]
        """
        assert isinstance(cross_attentions, (tuple, list)) and len(cross_attentions) > 0
        # 取最后几层更稳定且省一些开销
        if gcam_layers is not None and gcam_layers > 0 and gcam_layers < len(cross_attentions):
            atts = cross_attentions[-gcam_layers:]
        else:
            atts = cross_attentions

        gcam_per_layer = []
        # 用 sum() 让 grad 一次性算完
        out_sum = q_pos.sum()

        for A_k in atts:
            # A_k: [B,H,Lt,Lv], requires_grad=True
            grad = torch.autograd.grad(
                outputs=out_sum,
                inputs=A_k,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]  # [B,H,Lt,Lv]

            gcam = F.relu(grad * A_k)        # [B,H,Lt,Lv]
            gcam = gcam.mean(dim=1)          # mean over heads -> [B,Lt,Lv]
            gcam_per_layer.append(gcam)

        A_gcam = torch.stack(gcam_per_layer, dim=0).mean(dim=0)  # mean over layers -> [B,Lt,Lv]
        return A_gcam

    def _softmask_build_visual_mask(
        self,
        A_gcam: torch.Tensor,          # [B,Lt,Lv]
        i_w: torch.Tensor,             # [B]
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        取 word 行 -> 对 patch min-max -> M_soft = 1 - norm
        返回 [B,Lv]，CLS 位置=1，patch in [0,1]，最后 detach。
        """
        B, Lt, Lv = A_gcam.shape
        device = A_gcam.device

        b_idx = torch.arange(B, device=device)
        a_word = A_gcam[b_idx, i_w, :]   # [B,Lv]

        # 只对 patch(1:)做 min-max
        patch = a_word[:, 1:]
        mn = patch.min(dim=1, keepdim=True).values
        mx = patch.max(dim=1, keepdim=True).values
        patch_norm = (patch - mn) / (mx - mn + eps)   # [B, Lv-1]

        M = torch.ones_like(a_word)
        M[:, 1:] = 1.0 - patch_norm
        return M.detach()

    def compute_itm_softmask_loss(
        self,
        *,
        text_embeds: torch.Tensor,         # [B,Lt,D]  (来自 text2 的 text_output)
        text_atts: torch.Tensor,           # [B,Lt]
        text_ids: torch.Tensor,            # [B,Lt]    (text2['input_ids'])
        image_embeds: torch.Tensor,        # [B,Lv,D]  (image1 vision tokens)
        image_atts: torch.Tensor,          # [B,Lv]
        output_pos,                        # fusion 输出，必须带 cross_attentions
        itm_head,                          # self.itm_head
        gcam_layers: int = 3,
        beta: float = 0.7,
        weight: float = 1.0,
    ) -> torch.Tensor:
        """
        完整 SoftMask-ITM 分支：
        1) 用 output_pos 的 cross_attentions + q_pos 算 A_gcam
        2) 用 groundedness 引导采样 i_w
        3) 构造 M_soft，mask image_embeds
        4) rerun fusion，算正样本 CE
        """
        # 1) 正样本 logits（用于 q_pos）
        cls_pos = output_pos.last_hidden_state[:, 0, :]      # [B,D]
        logits_pos = itm_head(cls_pos)                       # [B,2]
        q_pos = logits_pos[:, 1]                             # [B]

        # 2) Grad-CAM over cross-attn
        cross_atts = output_pos.cross_attentions
        A_gcam = self._softmask_compute_word_gcam(q_pos, cross_atts, gcam_layers=gcam_layers)  # [B,Lt,Lv]

        # 3) groundedness 引导采样 i_w（用你现有相关性计算，no_grad）
        with torch.no_grad():
            g = self.compute_cross_modal_groundedness(
                text_ids=text_ids,
                attention_mask=text_atts,
                image_embeds=image_embeds,
                image_atts=image_atts,
                saliency_image=None,     # 这里只用于采样词，不需要 patch saliency
                layers=3,
                use_entropy=True,
                use_patch_saliency=False,
            )  # [B,Lt] in [0,1]
            i_w = self._softmask_sample_word_index(
                groundedness=g,
                input_ids=text_ids,
                attention_mask=text_atts,
                beta=beta,
            )  # [B]

        # 4) build M_soft and mask visual tokens
        M_soft = self._softmask_build_visual_mask(A_gcam, i_w)         # [B,Lv]
        image_embeds_sm = image_embeds * M_soft.unsqueeze(-1)          # [B,Lv,D]

        # 5) rerun fusion (positive pairs only)
        out_sm = self.text_encoder.bert(
            encoder_embeds=text_embeds,
            attention_mask=text_atts,
            encoder_hidden_states=image_embeds_sm,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode='fusion',
        )
        logits_sm = itm_head(out_sm.last_hidden_state[:, 0, :])        # [B,2]
        labels_pos = torch.ones(logits_sm.size(0), dtype=torch.long, device=logits_sm.device)

        loss_sm = F.cross_entropy(logits_sm, labels_pos)
        return loss_sm * float(weight)
