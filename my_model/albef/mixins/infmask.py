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
        neg_filter: Optional[torch.Tensor] = None, # NEW: [B,B]，True=不要把它当负样本
    ) -> torch.Tensor:
        device = image_embeds.device
        B, L_v, D = image_embeds.shape
        _, L_t, _ = text_embeds.shape
        # === [C] 课程式调度：起始/坡度/K/keep 比例 ===
        start_ep = int(config.get('infmask_start_epoch', 5))     # 第 5 个 epoch 开始启用
        ramp_ep  = int(config.get('infmask_ramp_epochs', 10))    # 10 个 epoch 线性增强
        K_min    = int(config.get('infmask_K_min', 2))
        K_max    = int(config.get('infmask_K_max', 6))
        # 文本/图像保留比例：从“高保留(弱遮挡)”→“低保留(强遮挡)”
        keep_t_high, keep_t_low = config.get('infmask_keep_t_schedule', (0.9, 0.5))
        keep_v_high, keep_v_low = config.get('infmask_keep_v_schedule', (0.9, 0.5))

        if epoch < start_ep:
            # 课程未开始：返回 0，不参与总 loss
            return torch.zeros([], device=device, dtype=image_embeds.dtype)
        # 线性进度 t ∈ [0,1]
        t = min(1.0, max(0.0, (epoch - start_ep) / max(1, ramp_ep)))
        K = int(round(K_min + t * (K_max - K_min)))

        # 当前 epoch 的 keep 区间（上下界可设相同，保留原接口）
        keep_t_min = keep_t_high + t * (keep_t_low - keep_t_high)
        keep_t_max = keep_t_min
        keep_v_min = keep_v_high + t * (keep_v_low - keep_v_high)
        keep_v_max = keep_v_min
        # # ---- hyperparams with safe defaults
        # K = int(config.get('infmask_K', 6))
        # keep_v_min, keep_v_max = config.get('infmask_keep_v_range', (0.1, 0.2))
        # keep_t_min, keep_t_max = config.get('infmask_keep_t_range', (0.1, 0.2))
        
        
        min_keep_v = int(config.get('infmask_min_keep_v', 3))
        min_keep_t = int(config.get('infmask_min_keep_t', 3))
        anchor_stop_grad = bool(config.get('infmask_anchor_stop_grad', False))
        use_saliency = bool(config.get('infmask_use_saliency', False))
        # phase: 'none' | 'keep_top' | 'mask_top'
        saliency_phase = str(config.get('infmask_saliency_phase', 'none'))
        phase_switch_epoch = int(config.get('infmask_saliency_switch_epoch', 999999))
        if epoch >= phase_switch_epoch:
            # flip keep/mask strategy after switch epoch
            saliency_phase = 'mask_top' if saliency_phase == 'keep_top' else saliency_phase

        # modes probability
        modes_probs = config.get('infmask_modes_probs', {
            'kv_only': 0.5,    # mask IMAGE only (K/V side)
            'q_only':  0.1,    # mask TEXT only (Q side)
            'both':    0.4,    # mask BOTH sides
        })
        # normalize
        total_p = sum(max(0.0, float(v)) for v in modes_probs.values()) or 1.0
        for k in modes_probs:
            modes_probs[k] = float(modes_probs[k]) / total_p

        # temperature
        temp = self.infmask_temp
        if bool(config.get('infmask_freeze_temp', True)) and temp.requires_grad:
            temp = temp.detach()
        temp = temp.clamp(0.02, 0.2)

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
            if mode in ('q_only', 'both'):
                kv_keep_mask = self._infmask_build_keep_mask(
                    B=B, L=L_t, keep_ratio=keep_t, min_keep=min_keep_t,
                    device=device, must_keep_cls=True,
                    saliency=(saliency_text if (use_saliency and saliency_text is not None) else None),
                    saliency_phase=saliency_phase,
                    valid_mask=text_atts.bool(),                     # NEW
                )
            if mode in ('kv_only', 'both'):
                q_keep_mask = self._infmask_build_keep_mask(
                    B=B, L=L_v, keep_ratio=keep_v, min_keep=min_keep_v,
                    device=device, must_keep_cls=True,
                    saliency=(saliency_image if (use_saliency and saliency_image is not None) else None),
                    saliency_phase=saliency_phase,
                    valid_mask=image_atts.bool(),                    # NEW
                )

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
            z_full_p = self.infmask_ln(self.infmask_head(z_full))
            z_mask_p = self.infmask_ln(self.infmask_head(z_mask))
            z_full_p = F.normalize(z_full_p, dim=-1)
            z_mask_p = F.normalize(z_mask_p, dim=-1)

            logits = (z_mask_p @ z_full_p.t()) / temp

            if neg_filter is not None:
                # 确保不影响对角正样本
                diag = torch.eye(B, dtype=torch.bool, device=device)
                mask = neg_filter.clone()
                mask[diag] = False
                logits = logits.masked_fill(mask, float('-inf'))
            loss_k = F.cross_entropy(logits, labels)
            losses.append(loss_k)

        return (torch.stack(losses, dim=0).mean() if len(losses) > 0  else torch.zeros([], device=device, dtype=image_embeds.dtype))

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
        valid_mask: Optional[torch.Tensor] = None,    # [B, L] True=有效位（例如 attention_mask==1）
    ) -> torch.Tensor:
        """
        返回布尔 keep 掩码 [B, L]（True=保留，False=遮挡），
        仅在 valid_mask==True 的位置上进行采样与计数；CLS 位若 must_keep_cls=True 则强制保留。
        - 随机模式：在有效位中等概率抽样到目标保留数（含 min_keep 约束）
        - 显著性模式：
            keep_top : 直接在有效位中选前 k 个显著 token
            mask_top : 在有效位中遮前 (1-keep_ratio) 部分，剩余即为保留（严格比例）
        """
        if valid_mask is None:
            valid_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        keep = torch.zeros(B, L, dtype=torch.bool, device=device)

        for b in range(B):
            valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                # 无有效位：全 False，但如果需要保 CLS 且存在位置 0，则保留 0
                if must_keep_cls and L > 0:
                    keep[b, 0] = True
                continue

            # 以“有效长度”为准计算 k_target，并裁剪到 [min_keep, 有效长度]
            eff_len = int(valid_idx.numel())
            k_target = max(1, int(round(keep_ratio * eff_len)))
            k_target = max(min_keep, k_target)
            k_target = min(k_target, eff_len)

            # 先处理必须保留的 CLS
            picked = set()
            if must_keep_cls and L > 0 and valid_mask[b, 0]:
                keep[b, 0] = True
                picked.add(0)

            # 三种模式
            if saliency is None or saliency_phase == 'none':
                # 等概率随机：从有效位中（去掉已选 CLS）再采够 (k_target - 已选)
                rest = valid_idx[~torch.isin(valid_idx, torch.tensor(list(picked), device=device))] if picked else valid_idx
                need = k_target - len(picked)
                if need > 0 and rest.numel() > 0:
                    choose = rest[torch.randperm(rest.numel(), device=device)[:need]]
                    keep[b, choose] = True

            elif saliency_phase == 'keep_top':
                s = saliency[b, valid_idx]
                order = torch.argsort(s, dim=0, descending=True)    # 高显著优先
                choose = valid_idx[order[:k_target]]
                keep[b, choose] = True
                if must_keep_cls and L > 0:
                    keep[b, 0] = True  # 再次确保 CLS

            elif saliency_phase == 'mask_top':
                # 严格比例：在有效位中按显著性遮掉前 (1 - keep_ratio) 部分，其余为保留
                s = saliency[b, valid_idx]
                order = torch.argsort(s, dim=0, descending=True)    # 高显著在前
                drop_cnt = max(0, eff_len - k_target)
                drop_idx = valid_idx[order[:drop_cnt]]              # 这些被遮
                keep[b, valid_idx] = True                           # 先全保留有效位
                keep[b, drop_idx] = False                           # 再遮掉 top
                if must_keep_cls and L > 0 and valid_mask[b, 0]:
                    keep[b, 0] = True

            else:
                # fallback 随机
                rest = valid_idx[~torch.isin(valid_idx, torch.tensor(list(picked), device=device))] if picked else valid_idx
                need = k_target - len(picked)
                if need > 0 and rest.numel() > 0:
                    choose = rest[torch.randperm(rest.numel(), device=device)[:need]]
                    keep[b, choose] = True

            # 兜底：若最终保留不足 min_keep（理论上不会，因为上面按有效长度算过），再补到 min_keep
            cur = int(keep[b].logical_and(valid_mask[b]).sum().item())
            if cur < min_keep:
                rest = torch.nonzero(valid_mask[b] & (~keep[b]), as_tuple=False).flatten()
                need = min_keep - cur
                if need > 0 and rest.numel() > 0:
                    add = rest[torch.randperm(rest.numel(), device=device)[:need]]
                    keep[b, add] = True

        # 保证 CLS（若存在）最终为 True
        if must_keep_cls and L > 0:
            keep[:, 0] = True

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