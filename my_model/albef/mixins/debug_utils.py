import os
import torch

class DebugMaskMixin:
    """
    Pure-text debug for MLM masking:
      - per-token table: idx, tok_before, tok_after, masked?, p_mask, saliency, ranks
      - summary metrics + top-k lists
    """

    # -------------------------
    # helpers
    # -------------------------
    def _is_rank0(self) -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _ensure_dir(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _tokenize_ids(self, ids):
        return [self.tokenizer.convert_ids_to_tokens(int(t)) for t in ids]

    def _safe_decode(self, ids, valid_mask=None):
        try:
            if valid_mask is not None:
                ids2 = [ids[i] for i in range(len(ids)) if bool(valid_mask[i])]
                return self.tokenizer.decode(ids2, skip_special_tokens=True)
            return self.tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            try:
                return " ".join([self.tokenizer.convert_ids_to_tokens(int(t)) for t in ids])
            except Exception:
                return str(ids)

    def _ascii_bar(self, x: float, width: int = 16) -> str:
        x = float(max(0.0, min(1.0, x)))
        n = int(round(x * width))
        return "[" + ("#" * n) + ("." * (width - n)) + "]"

    def _clip(self, s: str, n: int = 18) -> str:
        s = str(s)
        return s if len(s) <= n else (s[: n - 1] + "…")

    def _compute_rank_map(self, scores: torch.Tensor, valid: torch.Tensor, descending: bool = True):
        """
        scores: [L] float
        valid : [L] bool
        returns: rank_map [L] int (1..n_valid), invalid -> 0
        """
        L = scores.numel()
        rank_map = torch.zeros(L, dtype=torch.long, device=scores.device)
        if int(valid.sum().item()) == 0:
            return rank_map

        s = scores.clone()
        s[~valid] = -1e9 if descending else 1e9
        order = torch.argsort(s, descending=descending)
        r = 1
        for idx in order.tolist():
            if not bool(valid[idx]):
                continue
            rank_map[idx] = r
            r += 1
        return rank_map

    # -------------------------
    # main entry
    # -------------------------
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
        probability_matrix: torch.Tensor = None,  # [B,L]
        saliency_norm: torch.Tensor = None,       # [B,L]
        layer_deltas: torch.Tensor = None,
        layer_indices=None,
        raw_texts=None,
        out_path: str = "./mask_output.txt",
        limit_per_epoch: int = 30,
        sample_per_step: int = 2,
        topk_tokens: int = 8,
        step_prob: float = 0.15,
        max_tokens_per_sample: int = 160,
        show_all_tokens: bool = True,
        only_show_maskable: bool = False,
        max_table_rows: int = 260,
        # NEW (optional): if you want to check "top focus group"
        focus_top_p: float = None,        # e.g. 0.3, keep None if you don't want
    ):
        if not self._is_rank0():
            return

        if step_prob < 1.0 and torch.rand(1).item() > float(step_prob):
            return

        if not hasattr(self, "_dbg_written_per_epoch_text"):
            self._dbg_written_per_epoch_text = {}
        written = int(self._dbg_written_per_epoch_text.get(int(epoch), 0))
        if written >= int(limit_per_epoch):
            return

        self._ensure_dir(out_path)

        B, L = input_ids_before.shape
        perm = torch.randperm(B, device=input_ids_before.device)
        pick = perm[: min(int(sample_per_step), B)].tolist()

        blocks = []
        if written == 0:
            sep = "=" * 120
            blocks.append(f"\n{sep}\n[Epoch {int(epoch)}] Mask Debug (TEXT-ONLY, per-token p_mask + saliency)\n{sep}\n")

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        cls_id = getattr(self.tokenizer, "cls_token_id", None)
        sep_id = getattr(self.tokenizer, "sep_token_id", None)
        special_ids = set([x for x in [pad_id, cls_id, sep_id] if x is not None])

        for b in pick:
            if written >= int(limit_per_epoch):
                break

            valid = attention_mask[b].bool()

            # maskable 与 build_curriculum_mask_probs 对齐：valid & 非 special
            maskable = valid.clone()
            if special_ids:
                for sid in special_ids:
                    maskable &= (input_ids_before[b] != int(sid))

            # ✅ 关键修复：masked_pos 应该对齐 maskable，而不是 valid
            masked_pos = (targets[b] != -100) & maskable

            n_valid = int(valid.sum().item())
            n_maskable = int(maskable.sum().item())
            n_masked = int(masked_pos.sum().item())
            if n_valid == 0:
                continue
            if n_masked == 0 and not show_all_tokens:
                continue

            ids0 = input_ids_before[b].tolist()
            ids1 = input_ids_after[b].tolist()
            toks0 = self._tokenize_ids(ids0)
            toks1 = self._tokenize_ids(ids1)

            # truncate but keep last masked
            show_L = min(int(max_tokens_per_sample), L)
            if show_L < L and n_masked > 0:
                last_mask = int(torch.nonzero(masked_pos, as_tuple=False).max().item())
                show_L = min(L, max(show_L, last_mask + 1))

            toks0 = toks0[:show_L]
            toks1 = toks1[:show_L]
            valid_show = valid[:show_L]
            maskable_show = maskable[:show_L]
            masked_show = masked_pos[:show_L]

            orig_sent = self._safe_decode(ids0[:show_L], valid_mask=valid_show)
            after_sent = self._safe_decode(ids1[:show_L], valid_mask=valid_show)

            raw = None
            if raw_texts is not None:
                try:
                    raw = str(raw_texts[b])
                except Exception:
                    raw = None

            # tensors
            dev = input_ids_before.device
            sal_b = (saliency_norm[b, :show_L].float().to(dev) if saliency_norm is not None
                     else torch.zeros(show_L, device=dev, dtype=torch.float32))
            prob_b = (probability_matrix[b, :show_L].float().to(dev) if probability_matrix is not None
                      else torch.zeros(show_L, device=dev, dtype=torch.float32))

            # ✅ 关键修复：rank_sal 也应基于 maskable_show（与 p_mask 对齐）
            rank_sal = self._compute_rank_map(sal_b, maskable_show, descending=True)
            rank_prob = self._compute_rank_map(prob_b, maskable_show, descending=True)

            # summary metrics
            metric_lines = []

            if saliency_norm is not None:
                if n_maskable > 0:
                    s_all = float(sal_b[maskable_show].mean().item())
                else:
                    s_all = 0.0

                if int(masked_show.sum().item()) > 0:
                    s_m = float(sal_b[masked_show].mean().item())
                else:
                    s_m = 0.0

                ratio = s_m / (s_all + 1e-6)
                metric_lines.append(f"  - sal_mean_maskable={s_all:.6f} | sal_mean_masked={s_m:.6f} | ratio={ratio:.4f}")

                # ✅ top-k saliency hit rate：也只在 maskable 集合里选 topk
                k = min(int(topk_tokens), max(1, int(maskable_show.sum().item())))
                s2 = sal_b.clone()
                s2[~maskable_show] = -1e9
                topk_idx = torch.topk(s2, k=k, largest=True).indices
                hit = float(masked_show[topk_idx].float().mean().item())
                metric_lines.append(f"  - top{k}_sal_masked_hit_rate={hit:.4f}")

            if probability_matrix is not None:
                if int(maskable_show.sum().item()) > 0:
                    p_all = float(prob_b[maskable_show].mean().item())
                    E_hat = float(prob_b[maskable_show].sum().item())  # 期望 masked 数
                else:
                    p_all, E_hat = 0.0, 0.0

                if int((masked_show & maskable_show).sum().item()) > 0:
                    p_m = float(prob_b[masked_show & maskable_show].mean().item())
                else:
                    p_m = 0.0

                metric_lines.append(f"  - p_mask_mean_maskable={p_all:.6f} | p_mask_mean_masked={p_m:.6f} | E_hat=sum(p)={E_hat:.3f}")

                # 可选：看 saliency 最高的一组 token 的平均 p 是否更高
                if (focus_top_p is not None) and (n_maskable > 0):
                    k_focus = int(round(float(focus_top_p) * float(n_maskable)))
                    k_focus = max(1, min(k_focus, n_maskable))
                    s3 = sal_b.clone()
                    s3[~maskable_show] = -1e9
                    idx_focus = torch.topk(s3, k=k_focus, largest=True).indices
                    focus_mean_p = float(prob_b[idx_focus].mean().item())
                    rest_mask = maskable_show.clone()
                    rest_mask[idx_focus] = False
                    rest_mean_p = float(prob_b[rest_mask].mean().item()) if int(rest_mask.sum().item()) > 0 else 0.0
                    metric_lines.append(f"  - focus_top_p={float(focus_top_p):.3f}: mean_p_focus={focus_mean_p:.6f} | mean_p_rest={rest_mean_p:.6f}")

            # legacy layer delta (optional)
            layer_lines = []
            if (layer_deltas is not None) and (layer_indices is not None):
                means = []
                for li, layer_id in enumerate(layer_indices):
                    d = layer_deltas[li, b, :show_L].float()
                    m = float(d[valid_show].mean().item()) if int(valid_show.sum().item()) > 0 else 0.0
                    means.append(m)
                    layer_lines.append(f"  - layer {int(layer_id):02d}: mean_delta={m:.6f}")
                if means:
                    layer_lines.insert(0, f"  - mean_over_layers={sum(means)/len(means):.6f}")

            # per-token table
            header = (
                "IDX  M  MASKABLE  TOK_BEFORE           -> TOK_AFTER            | p_mask    | sal      | rank_sal | rank_p"
            )
            table = [header, "-" * len(header)]

            rows = 0
            for j in range(show_L):
                if not bool(valid_show[j]):
                    continue
                if only_show_maskable and not bool(maskable_show[j]):
                    continue
                if (not show_all_tokens) and (not bool(masked_show[j])):
                    continue

                mflag = "*" if bool(masked_show[j]) else " "
                mkable = "Y" if bool(maskable_show[j]) else "N"

                p = float(prob_b[j].item()) if probability_matrix is not None else -1.0
                s = float(sal_b[j].item()) if saliency_norm is not None else -1.0
                bar = self._ascii_bar(s if s >= 0 else 0.0, width=12)

                rs = int(rank_sal[j].item())
                rp = int(rank_prob[j].item())

                table.append(
                    f"{j:03d}  {mflag}    {mkable}     "
                    f"{self._clip(toks0[j], 18):<18} -> {self._clip(toks1[j], 18):<18} | "
                    f"{p:8.4f} | {s:7.4f} {bar} | "
                    f"{rs:7d} | {rp:6d}"
                )
                rows += 1
                if rows >= int(max_table_rows):
                    table.append(f"... (table truncated at {max_table_rows} rows)")
                    break

            # quick scan top-k
            topk_txt = []
            if saliency_norm is not None and int(maskable_show.sum().item()) > 0:
                k = min(int(topk_tokens), max(1, int(maskable_show.sum().item())))
                s2 = sal_b.clone()
                s2[~maskable_show] = -1e9
                idx = torch.topk(s2, k=k, largest=True).indices.tolist()
                topk_txt.append(
                    "TOP-SAL: " + ", ".join(
                        [f"{i}:{toks0[i]}(sal={float(sal_b[i]):.3f},p={float(prob_b[i]):.3f},mk={int(maskable_show[i].item())})" for i in idx]
                    )
                )

            if probability_matrix is not None and int(maskable_show.sum().item()) > 0:
                k = min(int(topk_tokens), max(1, int(maskable_show.sum().item())))
                p2 = prob_b.clone()
                p2[~maskable_show] = -1e9
                idx = torch.topk(p2, k=k, largest=True).indices.tolist()
                topk_txt.append(
                    "TOP-P  : " + ", ".join(
                        [f"{i}:{toks0[i]}(p={float(prob_b[i]):.3f},sal={float(sal_b[i]):.3f},mk={int(maskable_show[i].item())})" for i in idx]
                    )
                )

            # block write
            bt = []
            bt.append("-" * 120)
            bt.append(
                f"Sample #{int(b)} | step={int(step)} | valid={n_valid} | maskable={n_maskable} | "
                f"masked={n_masked} (masked/maskable={n_masked/max(1,n_maskable):.3f})"
            )
            if raw:
                bt.append(f"RAW : {raw}")
            bt.append(f"ORIG: {orig_sent}")
            bt.append(f"MASK: {after_sent}")

            if layer_lines:
                bt.append("LAYER-DELTA (legacy):")
                bt.extend(layer_lines)

            if metric_lines:
                bt.append("METRICS:")
                bt.extend(metric_lines)

            if topk_txt:
                bt.append("QUICKSCAN:")
                bt.extend(["  - " + x for x in topk_txt])

            bt.append("TOKENS (per-token p_mask + saliency; '*' means actually masked this step):")
            bt.extend(table)
            bt.append("")
            blocks.append("\n".join(bt))

            written += 1

        if blocks:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write("\n".join(blocks) + "\n")
            self._dbg_written_per_epoch_text[int(epoch)] = int(written)
