from typing import Optional, Tuple, Dict
import math
import torch
import torch.nn.functional as F
from torch import nn

class InfMaskMixin(nn.Module):
    """
    InfMasking（修改版）：
      - 每个样本构造 K 个“重遮挡”的多模态视图；
      - 现在不再在 encoder 输出上置零，而是：
          * 在原始文本 token 上按 keep_mask 把一部分 token 换成 [MASK]；
          * 在原始图像上按 patch keep_mask 做块级置零；
        然后重新跑 text encoder / ViT，再做跨模态融合；
      - 用 masked 视图的 CLS 去对齐 full 视图 CLS（InfoNCE），负样本仍是 in-batch。
    """

    def compute_infmask_loss(
        self,
        *,
        image: torch.Tensor,                # [B, 3, H, W] 原始图像
        text_ids: torch.Tensor,             # [B, L_t] 文本 token id
        image_embeds: torch.Tensor,         # [B, L_v, D] full 视觉 token（用来推断 patch 数）
        text_embeds: torch.Tensor,          # [B, L_t, D] full 文本 token（只用 shape）
        image_atts: torch.Tensor,           # [B, L_v]
        text_atts: torch.Tensor,            # [B, L_t]
        z_full: torch.Tensor,               # [B, D] full fused CLS (teacher)
        config: Dict,
        epoch: int,
        saliency_text: Optional[torch.Tensor] = None,  # [B, L_t] 可选文本显著性
        saliency_image: Optional[torch.Tensor] = None, # [B, L_v] 可选图像显著性（patch 粒度）
        neg_filter: Optional[torch.Tensor] = None,     # [B,B] True=不要当负样本
    ) -> torch.Tensor:
        device = image.device
        B, C, H, W = image.shape
        _, L_v, _ = image_embeds.shape
        _, L_t, _ = text_embeds.shape

        # === 课程式调度：起始/坡度/K/keep 比例 ===
        start_ep = int(config.get('infmask_start_epoch', 5))
        ramp_ep  = int(config.get('infmask_ramp_epochs', 10))
        K_min    = int(config.get('infmask_K_min', 2))
        K_max    = int(config.get('infmask_K_max', 6))

        keep_t_high, keep_t_low = config.get('infmask_keep_t_schedule', (0.9, 0.6))
        keep_v_high, keep_v_low = config.get('infmask_keep_v_schedule', (0.9, 0.6))

        if epoch < start_ep:
            # 课程未开始：不启用 InfMask，返回 0
            return torch.zeros([], device=device, dtype=image_embeds.dtype)

        # 线性进度 t ∈ [0,1]
        t = min(1.0, max(0.0, (epoch - start_ep) / max(1, ramp_ep)))
        K = 1  #int(round(K_min + t * (K_max - K_min)))

        keep_t_min = keep_t_high + t * (keep_t_low - keep_t_high)
        keep_t_max = keep_t_min
        keep_v_min = keep_v_high + t * (keep_v_low - keep_v_high)
        keep_v_max = keep_v_min

        min_keep_v = int(config.get('infmask_min_keep_v', 3))
        min_keep_t = int(config.get('infmask_min_keep_t', 3))
        anchor_stop_grad = bool(config.get('infmask_anchor_stop_grad', False))
        use_saliency = bool(config.get('infmask_use_saliency', False))

        # phase: 'none' | 'keep_top' | 'mask_top'
        saliency_phase = str(config.get('infmask_saliency_phase', 'none'))
        phase_switch_epoch = int(config.get('infmask_saliency_switch_epoch', 999999))
        if epoch >= phase_switch_epoch:
            # 到一定 epoch 后把 keep_top / mask_top 翻转
            saliency_phase = 'mask_top' if saliency_phase == 'keep_top' else saliency_phase

        # 三种模式的概率（kv_only: 只遮图像; q_only: 只遮文本; both: 都遮）
        modes_probs = config.get('infmask_modes_probs', {
            'kv_only': 0.5,
            'q_only':  0.1,
            'both':    0.4,
        })
        total_p = sum(max(0.0, float(v)) for v in modes_probs.values()) or 1.0
        for k in modes_probs:
            modes_probs[k] = float(modes_probs[k]) / total_p

        # 温度
        temp = self.infmask_temp
        if bool(config.get('infmask_freeze_temp', True)) and temp.requires_grad:
            temp = temp.detach()
        temp = temp.clamp(0.02, 0.2)

        # anchor stop_grad（可选）
        if anchor_stop_grad:
            z_full = z_full.detach()

        # InfoNCE label（对角线为正样本）
        labels = torch.arange(B, device=device)

        # 找 mask_token_id：优先 tokenizer，其次 text_encoder.config
        mask_token_id = None
        if hasattr(self, "tokenizer") and getattr(self.tokenizer, "mask_token_id", None) is not None:
            mask_token_id = int(self.tokenizer.mask_token_id)
        elif hasattr(self, "text_encoder") and hasattr(self.text_encoder, "config") \
             and getattr(self.text_encoder.config, "mask_token_id", None) is not None:
            mask_token_id = int(self.text_encoder.config.mask_token_id)
        else:
            raise ValueError(
                "InfMaskMixin: cannot find mask_token_id from tokenizer or text_encoder.config"
            )

        losses = []
        for _ in range(K):
            mode = self._infmask_sample_mode(modes_probs)
            # 采样当前视图的 keep 比例
            keep_v = float(torch.empty(1, device=device).uniform_(keep_v_min, keep_v_max).item())
            keep_t = float(torch.empty(1, device=device).uniform_(keep_t_min, keep_t_max).item())

            # 构造文本/图像的 keep mask（True = 保留）
            kv_keep_mask = None  # 文本侧
            q_keep_mask  = None  # 图像侧
            if mode in ('q_only', 'both'):
                kv_keep_mask = self._infmask_build_keep_mask(
                    B=B, L=L_t, keep_ratio=keep_t, min_keep=min_keep_t,
                    device=device, must_keep_cls=True,
                    saliency=(saliency_text if (use_saliency and saliency_text is not None) else None),
                    saliency_phase=saliency_phase,
                    valid_mask=text_atts.bool(),
                )
            if mode in ('kv_only', 'both'):
                q_keep_mask = self._infmask_build_keep_mask(
                    B=B, L=L_v, keep_ratio=keep_v, min_keep=min_keep_v,
                    device=device, must_keep_cls=True,
                    saliency=(saliency_image if (use_saliency and saliency_image is not None) else None),
                    saliency_phase=saliency_phase,
                    valid_mask=image_atts.bool(),
                )

            # === 1) 在“输入级别”构造 masked 文本 / 图像 ===
            # 文本：不保留的 token 改成 [MASK]，attention 仍然为 1（保留位置信息）
            text_ids_m, text_atts_m = self._infmask_apply_text_input_mask(
                text_ids=text_ids,
                text_atts=text_atts,
                keep_mask=kv_keep_mask,
                mask_token_id=mask_token_id,
            )
            # 图像：基于 patch keep_mask 在 H×W 上做块级置零，并配套 encoder_attention_mask
            image_m, image_atts_m = self._infmask_apply_image_input_mask(
                image=image,
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
            text_embeds_m = text_out_m.last_hidden_state          # [B, L_t, D_t]

            image_embeds_m = self.visual_encoder(image_m)         # [B, L_v, D_v]

            # === 2) 融合，拿 masked 视图 CLS ===
            out_mask = self.text_encoder.bert(
                encoder_embeds=text_embeds_m,
                attention_mask=text_atts_m,
                encoder_hidden_states=image_embeds_m,
                encoder_attention_mask=image_atts_m,
                return_dict=True,
                mode='fusion',
            )
            z_mask = out_mask.last_hidden_state[:, 0, :]          # [B, D_t]

            # === 3) Tiny 头 + 归一化 + InfoNCE ===
            z_full_p = self.infmask_ln(self.infmask_head(z_full))
            z_mask_p = self.infmask_ln(self.infmask_head(z_mask))
            z_full_p = F.normalize(z_full_p, dim=-1)
            z_mask_p = F.normalize(z_mask_p, dim=-1)

            logits = (z_mask_p @ z_full_p.t()) / temp

            if neg_filter is not None:
                # 不动对角线正样本，只屏蔽“不想当负样本”的位置
                diag = torch.eye(B, dtype=torch.bool, device=device)
                mask = neg_filter.clone()
                mask[diag] = False
                logits = logits.masked_fill(mask, float('-inf'))

            loss_k = F.cross_entropy(logits, labels)
            losses.append(loss_k)

        if len(losses) == 0:
            return torch.zeros([], device=device, dtype=image_embeds.dtype)
        return torch.stack(losses, dim=0).mean()

    # ----------------------
    # Helpers
    # ----------------------
    @staticmethod
    def _infmask_sample_mode(modes_probs: Dict[str, float]) -> str:
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
        """
        if valid_mask is None:
            valid_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        keep = torch.zeros(B, L, dtype=torch.bool, device=device)

        for b in range(B):
            valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                if must_keep_cls and L > 0:
                    keep[b, 0] = True
                continue

            eff_len = int(valid_idx.numel())
            k_target = max(1, int(round(keep_ratio * eff_len)))
            k_target = max(min_keep, k_target)
            k_target = min(k_target, eff_len)

            picked = set()
            if must_keep_cls and L > 0 and valid_mask[b, 0]:
                keep[b, 0] = True
                picked.add(0)

            if saliency is None or saliency_phase == 'none':
                rest = valid_idx[
                    ~torch.isin(valid_idx, torch.tensor(list(picked), device=device))
                ] if picked else valid_idx
                need = k_target - len(picked)
                if need > 0 and rest.numel() > 0:
                    choose = rest[torch.randperm(rest.numel(), device=device)[:need]]
                    keep[b, choose] = True

            elif saliency_phase == 'keep_top':
                s = saliency[b, valid_idx]
                order = torch.argsort(s, dim=0, descending=True)
                choose = valid_idx[order[:k_target]]
                keep[b, choose] = True
                if must_keep_cls and L > 0:
                    keep[b, 0] = True

            elif saliency_phase == 'mask_top':
                s = saliency[b, valid_idx]
                order = torch.argsort(s, dim=0, descending=True)
                drop_cnt = max(0, eff_len - k_target)
                drop_idx = valid_idx[order[:drop_cnt]]
                keep[b, valid_idx] = True
                keep[b, drop_idx] = False
                if must_keep_cls and L > 0 and valid_mask[b, 0]:
                    keep[b, 0] = True
            else:
                rest = valid_idx[
                    ~torch.isin(valid_idx, torch.tensor(list(picked), device=device))
                ] if picked else valid_idx
                need = k_target - len(picked)
                if need > 0 and rest.numel() > 0:
                    choose = rest[torch.randperm(rest.numel(), device=device)[:need]]
                    keep[b, choose] = True

            cur = int(keep[b].logical_and(valid_mask[b]).sum().item())
            if cur < min_keep:
                rest = torch.nonzero(valid_mask[b] & (~keep[b]), as_tuple=False).flatten()
                need = min_keep - cur
                if need > 0 and rest.numel() > 0:
                    add = rest[torch.randperm(rest.numel(), device=device)[:need]]
                    keep[b, add] = True

        if must_keep_cls and L > 0:
            keep[:, 0] = True
        return keep

    @staticmethod
    def _infmask_apply_text_input_mask(
        *,
        text_ids: torch.Tensor,              # [B, L_t]
        text_atts: torch.Tensor,             # [B, L_t]
        keep_mask: Optional[torch.Tensor],   # [B, L_t] True=keep
        mask_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在 token 级别做 mask：
          - keep_mask=False 的位置替换成 [MASK] id；
          - attention_mask 不变（仍为 1），保留位置信息，只把内容抹掉。
        """
        if keep_mask is None:
            return text_ids, text_atts
        masked_ids = text_ids.clone()
        masked_atts = text_atts.clone()
        mask = ~keep_mask
        masked_ids[mask] = mask_token_id
        return masked_ids, masked_atts

    @staticmethod
    def _infmask_apply_image_input_mask(
        *,
        image: torch.Tensor,                 # [B, 3, H, W]
        image_embeds: torch.Tensor,          # [B, L_v, D]
        image_atts: torch.Tensor,            # [B, L_v]
        keep_mask: Optional[torch.Tensor],   # [B, L_v] True=keep
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据 keep_mask 在输入图像上做 patch 级别的 mask，再配套一个 encoder_attention_mask:
          - CLS 位（index 0）通常恒为 True；
          - patch 部分 [1:] reshape 成 [B, gh, gw]，上采样到 H×W 作为 0/1 mask，
            对整块像素置零；
          - encoder_attention_mask 的 CLS=1，patch 按 keep_mask 设为 0/1。
        """
        if keep_mask is None:
            return image, image_atts

        B, C, H, W = image.shape
        _, L_v, _ = image_embeds.shape
        device = image.device

        n_patch = L_v - 1
        g = int(round(math.sqrt(n_patch)))
        if g * g != n_patch:
            raise ValueError(
                f"InfMaskMixin: cannot infer patch grid from L_v={L_v}, n_patch={n_patch}"
            )

        patch_keep = keep_mask[:, 1:]                # [B, n_patch]
        patch_keep_2d = patch_keep.view(B, 1, g, g).float()
        patch_mask_img = F.interpolate(patch_keep_2d, size=(H, W), mode='nearest')
        patch_mask_img = (patch_mask_img >= 0.5).float()  # [B,1,H,W]

        masked_image = image * patch_mask_img         # 块级置零

        masked_atts = image_atts.clone()
        masked_atts[:, 1:] = patch_keep.long()        # CLS 保留，patch 0/1
        return masked_image, masked_atts
