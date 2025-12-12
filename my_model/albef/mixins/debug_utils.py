import os
import torch

import os
import torch

class DebugMaskMixin:
    @torch.no_grad()
    def debug_render_mask_with_norms(
        self,
        *,
        epoch: int,
        step: int,
        input_ids_before: torch.Tensor,   # [B,L]
        input_ids_after: torch.Tensor,    # [B,L]
        targets: torch.Tensor,            # [B,L] -100 means not masked
        attention_mask: torch.Tensor,     # [B,L]
        probability_matrix: torch.Tensor = None,  # [B,L] 可选
        saliency_norm: torch.Tensor = None,       # [B,L] 可选
        layer_deltas: torch.Tensor = None,        # [nL,B,L] 可选
        layer_indices=None,                       # list[int]
        raw_texts=None,
        out_path: str = "./mask_output.txt",
        limit_per_epoch: int = 30,
        sample_per_step: int = 2,
        topk_tokens: int = 8,
        step_prob: float = 0.15,
    ):
        # DDP: only rank0
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        # 触发概率（降低IO）
        if step_prob < 1.0:
            if torch.rand(1).item() > float(step_prob):
                return

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        if not hasattr(self, "_dbg_written_per_epoch2"):
            self._dbg_written_per_epoch2 = {}
        written = int(self._dbg_written_per_epoch2.get(int(epoch), 0))
        if written >= limit_per_epoch:
            return

        B, L = input_ids_before.shape

        # 随机挑样本（更“随机”）
        perm = torch.randperm(B, device=input_ids_before.device)
        pick = perm[: min(int(sample_per_step), B)].tolist()

        blocks = []
        if written == 0:
            sep = "=" * 110
            blocks.append(f"\n{sep}\n[Epoch {int(epoch)} | step {int(step)}] Mask+Norm Debug\n{sep}\n")

        for b in pick:
            if written >= limit_per_epoch:
                break

            valid = attention_mask[b].bool()
            masked_pos = (targets[b] != -100) & valid
            if masked_pos.sum().item() == 0:
                continue

            ids0 = input_ids_before[b].tolist()
            ids1 = input_ids_after[b].tolist()
            toks0 = [self.tokenizer.convert_ids_to_tokens(int(t)) for t in ids0]
            toks1 = [self.tokenizer.convert_ids_to_tokens(int(t)) for t in ids1]

            try:
                orig_sent = self.tokenizer.decode([ids0[i] for i in range(L) if valid[i]], skip_special_tokens=True)
                after_sent = self.tokenizer.decode([ids1[i] for i in range(L) if valid[i]], skip_special_tokens=True)
            except Exception:
                orig_sent = self.tokenizer.decode(ids0, skip_special_tokens=False)
                after_sent = self.tokenizer.decode(ids1, skip_special_tokens=False)

            # --- 逐层范数统计 ---
            layer_lines = []
            total_mean = None
            total_sum = None
            if layer_deltas is not None and layer_indices is not None:
                # layer_deltas: [nL,B,L]
                means = []
                for li, layer_id in enumerate(layer_indices):
                    d = layer_deltas[li, b]  # [L]
                    m = float(d[valid].mean().item())
                    means.append(m)
                    layer_lines.append(f"  - layer {int(layer_id):02d}: mean_delta={m:.6f}")
                total_mean = sum(means) / max(1, len(means))
                total_sum = sum(means)

            # --- 指标：masked token 的显著性更偏向高/低？ ---
            metric_lines = []
            if saliency_norm is not None:
                s = saliency_norm[b]
                s_all = float(s[valid].mean().item())
                s_m = float(s[masked_pos].mean().item())
                ratio = s_m / (s_all + 1e-6)
                metric_lines.append(f"  - sal_mean_all={s_all:.6f} | sal_mean_masked={s_m:.6f} | ratio={ratio:.4f}")

                # top-k hit
                k = min(int(topk_tokens), int(valid.sum().item()))
                s2 = s.clone()
                s2[~valid] = -1e9
                topk = torch.topk(s2, k=k, largest=True).indices
                hit = float(masked_pos[topk].float().mean().item())
                metric_lines.append(f"  - top{k}_masked_hit_rate={hit:.4f}")

            # --- 列出 masked 的 token、以及 mask 概率/显著性 ---
            masked_idx = torch.nonzero(masked_pos, as_tuple=False).squeeze(-1).tolist()
            if isinstance(masked_idx, int):
                masked_idx = [masked_idx]
            token_lines = []
            for j in masked_idx[:50]:
                p = float(probability_matrix[b, j].item()) if probability_matrix is not None else -1.0
                s = float(saliency_norm[b, j].item()) if saliency_norm is not None else -1.0
                token_lines.append(f"  - pos={j:02d} {toks0[j]} -> {toks1[j]} | p={p:.4f} | sal={s:.4f}")

            raw = None
            if raw_texts is not None:
                try:
                    raw = str(raw_texts[b])
                except Exception:
                    raw = None

            block = []
            block.append("-" * 90)
            block.append(f"Sample #{b}")
            if raw:
                block.append(f"RAW : {raw}")
            block.append(f"ORIG: {orig_sent}")
            block.append(f"MASK: {after_sent}")

            if total_mean is not None:
                block.append(f"NORMS: mean_over_layers={total_mean:.6f} | sum_over_layers={total_sum:.6f}")
                block.append("LAYER_MEAN_DELTAS:")
                block.extend(layer_lines)

            if metric_lines:
                block.append("METRICS:")
                block.extend(metric_lines)

            block.append("MASKED TOKENS (with prob/saliency):")
            block.extend(token_lines)
            block.append("")

            blocks.append("\n".join(block))
            written += 1

        if len(blocks) > 0:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write("\n".join(blocks) + "\n")
            self._dbg_written_per_epoch2[int(epoch)] = written
