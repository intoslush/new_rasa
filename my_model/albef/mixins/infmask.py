from typing import Optional, Tuple, Dict
import torch
import torch.nn.functional as F
from torch import nn


class InfMaskMixin(nn.Module):
    """
    Plug-in mixin to add InfMasking-style loss:
      - Before fusion, create K masked multi-modal views with high masking ratios
      - Compute fused CLS for each masked view, align to the full fused CLS (same sample)
      - In-batch negatives only (simple/stable); optionally extend later with queues

    Expected attributes in the host model:
      - self.text_encoder.bert(... mode='fusion')
      - self.temp : nn.Parameter or float temperature
      - concat_all_gather (optional, not used here by default)
    """

    # ----------------------
    # Public entry point
    # ----------------------
    def compute_infmask_loss(
        self,
        *,
        image_embeds: torch.Tensor,          # [B, L_v, D]
        text_embeds: torch.Tensor,           # [B, L_t, D]
        image_atts: torch.Tensor,            # [B, L_v]
        text_atts: torch.Tensor,             # [B, L_t]
        z_full: torch.Tensor,                # [B, D] full (unmasked) fused CLS
        config: Dict,
        epoch: int,
        saliency_text: Optional[torch.Tensor] = None,  # [B, L_t] optional token saliency
        saliency_image: Optional[torch.Tensor] = None, # [B, L_v] optional patch saliency
    ) -> torch.Tensor:
        device = image_embeds.device
        B, L_v, D = image_embeds.shape
        _, L_t, _ = text_embeds.shape

        # ---- hyperparams with safe defaults
        K = int(config.get('infmask_K', 6))
        keep_v_min, keep_v_max = config.get('infmask_keep_v_range', (0.1, 0.2))
        keep_t_min, keep_t_max = config.get('infmask_keep_t_range', (0.1, 0.2))
        min_keep_v = int(config.get('infmask_min_keep_v', 3))
        min_keep_t = int(config.get('infmask_min_keep_t', 3))
        anchor_stop_grad = bool(config.get('infmask_anchor_stop_grad', True))
        use_saliency = bool(config.get('infmask_use_saliency', False))
        # phase: 'none' | 'keep_top' | 'mask_top'
        saliency_phase = str(config.get('infmask_saliency_phase', 'none'))
        phase_switch_epoch = int(config.get('infmask_saliency_switch_epoch', 999999))
        if epoch >= phase_switch_epoch:
            # flip keep/mask strategy after switch epoch
            saliency_phase = 'mask_top' if saliency_phase == 'keep_top' else saliency_phase

        # modes probability
        modes_probs = config.get('infmask_modes_probs', {
            'kv_only': 0.4,    # mask TEXT only (K/V side)
            'q_only':  0.2,    # mask IMAGE only (Q side)
            'both':    0.4,    # mask BOTH sides
        })
        # normalize
        total_p = sum(max(0.0, float(v)) for v in modes_probs.values()) or 1.0
        for k in modes_probs:
            modes_probs[k] = float(modes_probs[k]) / total_p

        # temperature
        temp = self.temp if isinstance(self.temp, torch.Tensor) else torch.tensor(self.temp, device=device)

        # normalize z_full for cosine-contrast
        z_full_n = F.normalize(z_full.detach() if anchor_stop_grad else z_full, dim=-1)

        # prepare labels for InfoNCE (positives are the diagonal)
        labels = torch.arange(B, device=device)

        losses = []
        for k in range(K):
            mode = self._infmask_sample_mode(modes_probs)
            # sample keep ratios
            keep_v = float(torch.empty(1).uniform_(keep_v_min, keep_v_max).item())
            keep_t = float(torch.empty(1).uniform_(keep_t_min, keep_t_max).item())

            # build per-view masks (True=keep)
            kv_keep_mask = None
            q_keep_mask  = None
            if mode in ('kv_only', 'both'):
                kv_keep_mask = self._infmask_build_keep_mask(
                    B=B, L=L_t, keep_ratio=keep_t, min_keep=min_keep_t,
                    device=device, must_keep_cls=True,
                    saliency=(saliency_text if (use_saliency and saliency_text is not None) else None),
                    saliency_phase=saliency_phase
                )  # [B, L_t]
            if mode in ('q_only', 'both'):
                q_keep_mask = self._infmask_build_keep_mask(
                    B=B, L=L_v, keep_ratio=keep_v, min_keep=min_keep_v,
                    device=device, must_keep_cls=True,
                    saliency=(saliency_image if (use_saliency and saliency_image is not None) else None),
                    saliency_phase=saliency_phase
                )  # [B, L_v]

            # apply masking to embeds & atts
            img_m, img_att_m = self._infmask_apply_image_mask(image_embeds, image_atts, q_keep_mask)
            txt_m, txt_att_m = self._infmask_apply_text_mask(text_embeds, text_atts, kv_keep_mask)

            # fuse and get CLS for masked view
            out_mask = self.text_encoder.bert(
                encoder_embeds=txt_m,
                attention_mask=txt_att_m,
                encoder_hidden_states=img_m,
                encoder_attention_mask=img_att_m,
                return_dict=True,
                mode='fusion',
            )
            z_mask = out_mask.last_hidden_state[:, 0, :]                 # [B, D_t]
            z_mask_n = F.normalize(z_mask, dim=-1)                        # [B, D]

            # InfoNCE over in-batch full CLS
            # logits[i, j] = <z_mask_i, z_full_j> / temp
            logits = (z_mask_n @ z_full_n.t()) / temp
            loss_k = F.cross_entropy(logits, labels)
            losses.append(loss_k)

        return torch.stack(losses, dim=0).mean() if len(losses) > 0 else torch.tensor(0.0, device=device)

    # ----------------------
    # Helpers
    # ----------------------
    @staticmethod
    def _infmask_sample_mode(modes_probs: Dict[str, float]) -> str:
        # multinomial over provided modes
        modes = list(modes_probs.keys())
        probs = torch.tensor([modes_probs[m] for m in modes])
        idx = torch.multinomial(probs, 1).item()
        return modes[idx]

    @staticmethod
    def _infmask_build_keep_mask(
        *, B: int, L: int, keep_ratio: float, min_keep: int, device: torch.device,
        must_keep_cls: bool = True,
        saliency: Optional[torch.Tensor] = None,      # [B, L]
        saliency_phase: str = 'none',                 # 'none' | 'keep_top' | 'mask_top'
    ) -> torch.Tensor:
        """
        Return boolean mask [B, L] where True=keep, False=mask.
        If saliency is provided:
          - keep_top : prefer keeping high-saliency tokens
          - mask_top : prefer masking high-saliency tokens
        """
        keep = torch.zeros(B, L, dtype=torch.bool, device=device)
        k_target = max(1, int(round(keep_ratio * L)))
        k_target = max(k_target, min_keep)

        if saliency is None or saliency_phase == 'none':
            # random keep
            randv = torch.rand(B, L, device=device)
            # threshold so that approx keep_ratio are kept; then ensure min_keep with top-k
            thresh = torch.quantile(randv, q=1.0 - keep_ratio, dim=1, keepdim=True)
            keep |= randv >= thresh
        else:
            # sort by saliency per sample
            # higher saliency first
            if saliency_phase == 'keep_top':
                # keep top-k_target salient tokens
                idx = torch.argsort(saliency, dim=1, descending=True)
                topk = idx[:, :k_target]
                keep.scatter_(1, topk, True)
            elif saliency_phase == 'mask_top':
                # mask top salient tokens; keep the rest down to k_target tokens
                idx = torch.argsort(saliency, dim=1, descending=True)
                # propose to drop the first m, keep remaining tokens randomly until k_target
                drop = idx[:, :max(1, int(0.5 * L))]  # drop at most half by saliency; rest random
                keep[:] = True
                keep.scatter_(1, drop, False)
                # enforce at least k_target keeps
                count = keep.sum(1)
                need = torch.clamp(k_target - count, min=0)
                for b in range(B):
                    if need[b] > 0:
                        # randomly add back some
                        cand = (~keep[b]).nonzero(as_tuple=False).flatten()
                        if cand.numel() > 0:
                            add_idx = cand[torch.randperm(cand.numel(), device=device)[:need[b]]]
                            keep[b, add_idx] = True
            else:
                # fallback random
                randv = torch.rand(B, L, device=device)
                thresh = torch.quantile(randv, q=1.0 - keep_ratio, dim=1, keepdim=True)
                keep |= randv >= thresh

        # ensure min_keep and CLS kept
        if must_keep_cls and L > 0:
            keep[:, 0] = True
        for b in range(B):
            if keep[b].sum().item() < min_keep:
                # randomly turn on to reach min_keep
                off = (~keep[b]).nonzero(as_tuple=False).flatten()
                if off.numel() > 0:
                    add = off[torch.randperm(off.numel(), device=keep.device)[:(min_keep - int(keep[b].sum().item()))]]
                    keep[b, add] = True
        return keep

    @staticmethod
    def _infmask_apply_text_mask(
        text_embeds: torch.Tensor,   # [B, L_t, D]
        text_atts: torch.Tensor,     # [B, L_t]
        keep_mask: Optional[torch.Tensor],    # [B, L_t] True=keep
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if keep_mask is None:
            return text_embeds, text_atts
        masked_embeds = text_embeds.clone()
        masked_atts   = text_atts.clone()
        # set masked positions to zero + attention off
        mask = ~keep_mask
        masked_embeds[mask] = 0.0
        masked_atts[mask] = 0
        return masked_embeds, masked_atts

    @staticmethod
    def _infmask_apply_image_mask(
        image_embeds: torch.Tensor,  # [B, L_v, D]
        image_atts: torch.Tensor,    # [B, L_v]
        keep_mask: Optional[torch.Tensor],    # [B, L_v] True=keep
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if keep_mask is None:
            return image_embeds, image_atts
        masked_embeds = image_embeds.clone()
        masked_atts   = image_atts.clone()
        mask = ~keep_mask
        masked_embeds[mask] = 0.0
        masked_atts[mask] = 0
        return masked_embeds, masked_atts