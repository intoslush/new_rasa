import os
import json
import html
import torch


class DebugMaskMixin:
    """
    Debug utility for MLM masking:
      - TXT: readable report with metrics + per-token lines
      - HTML: highlighted token visualization (mask + saliency/prob)
      - JSONL: structured records for later analysis

    Works with:
      - saliency_norm from groundedness (recommended)
      - optional layer_deltas (legacy hidden-delta ablation)
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

    def _safe_decode(self, ids, valid_mask=None):
        # ids: list[int]
        try:
            if valid_mask is not None:
                ids2 = [ids[i] for i in range(len(ids)) if bool(valid_mask[i])]
                return self.tokenizer.decode(ids2, skip_special_tokens=True)
            return self.tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            # fallback
            try:
                return " ".join([self.tokenizer.convert_ids_to_tokens(int(t)) for t in ids])
            except Exception:
                return str(ids)

    def _ascii_bar(self, x: float, width: int = 18) -> str:
        # x in [0,1]
        x = float(max(0.0, min(1.0, x)))
        n = int(round(x * width))
        return "[" + ("#" * n) + ("." * (width - n)) + "]"

    def _tokenize_ids(self, ids):
        return [self.tokenizer.convert_ids_to_tokens(int(t)) for t in ids]

    def _maybe_get_scalar(self, mat, b, j, default=-1.0):
        if mat is None:
            return float(default)
        try:
            return float(mat[b, j].item())
        except Exception:
            return float(default)

    def _get_epoch_paths(self, out_path: str, epoch: int):
        # keep your original out_path as TXT (append)
        txt_path = out_path
        base, ext = os.path.splitext(out_path)
        if ext.lower() != ".txt":
            base = out_path
        html_path = f"{base}_e{int(epoch):03d}.html"
        jsonl_path = f"{base}_e{int(epoch):03d}.jsonl"
        return txt_path, html_path, jsonl_path

    def _html_header(self):
        return """<!doctype html>
<html><head><meta charset="utf-8">
<style>
body { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; padding: 14px; }
.block { border: 1px solid #ddd; border-radius: 10px; padding: 12px; margin: 12px 0; }
.meta { color: #333; font-size: 13px; margin-bottom: 8px; white-space: pre-wrap; }
.sent { font-size: 14px; line-height: 1.7; white-space: pre-wrap; }
.tok { padding: 2px 4px; border-radius: 6px; margin: 1px 1px; display: inline-block; }
.masked { outline: 2px solid rgba(220, 0, 0, 0.55); }
.small { color: #666; font-size: 12px; }
hr { border: none; border-top: 1px solid #eee; margin: 10px 0; }
</style></head><body>
"""

    def _html_token_span(self, tok: str, sal: float, prob: float, masked: bool):
        # sal/prob in [0,1] (or -1)
        sal = 0.0 if sal < 0 else max(0.0, min(1.0, float(sal)))
        # background alpha from saliency; keep color neutral-ish
        alpha = 0.10 + 0.70 * sal
        bg = f"rgba(30, 144, 255, {alpha:.3f})"  # blue-ish
        cls = "tok masked" if masked else "tok"
        title = f"sal={sal:.4f} | p={float(prob):.4f}"
        safe_tok = html.escape(tok)
        return f'<span class="{cls}" style="background:{bg}" title="{html.escape(title)}">{safe_tok}</span>'

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
        layer_deltas: torch.Tensor = None,        # [nL,B,L] optional legacy
        layer_indices=None,                       # list[int] optional legacy
        raw_texts=None,
        out_path: str = "./mask_output.txt",
        limit_per_epoch: int = 30,
        sample_per_step: int = 2,
        topk_tokens: int = 8,
        step_prob: float = 0.15,
        # NEW options (all safe defaults)
        write_html: bool = True,
        write_jsonl: bool = True,
        max_tokens_per_sample: int = 120,
        max_masked_list: int = 60,
    ):
        # DDP: only rank0
        if not self._is_rank0():
            return

        # sample trigger (reduce IO)
        if step_prob < 1.0:
            if torch.rand(1).item() > float(step_prob):
                return

        # per-epoch quota
        if not hasattr(self, "_dbg_written_per_epoch_v2"):
            self._dbg_written_per_epoch_v2 = {}
        written = int(self._dbg_written_per_epoch_v2.get(int(epoch), 0))
        if written >= int(limit_per_epoch):
            return

        txt_path, html_path, jsonl_path = self._get_epoch_paths(out_path, epoch)
        self._ensure_dir(txt_path)
        self._ensure_dir(html_path)
        self._ensure_dir(jsonl_path)

        # init html file if needed
        if write_html and (not os.path.exists(html_path)):
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self._html_header())
                f.write(f"<div class='small'>Epoch {int(epoch)} debug file (append during training). You can refresh the page.</div>\n<hr/>\n")

        B, L = input_ids_before.shape

        # choose samples
        perm = torch.randperm(B, device=input_ids_before.device)
        pick = perm[: min(int(sample_per_step), B)].tolist()

        # TXT header per first write in epoch
        blocks_txt = []
        if written == 0:
            sep = "=" * 110
            blocks_txt.append(f"\n{sep}\n[Epoch {int(epoch)}] Mask Debug (groundedness-ready)\n{sep}\n")

        for b in pick:
            if written >= int(limit_per_epoch):
                break

            valid = attention_mask[b].bool()
            masked_pos = (targets[b] != -100) & valid
            n_valid = int(valid.sum().item())
            n_masked = int(masked_pos.sum().item())
            if n_masked == 0:
                continue

            ids0 = input_ids_before[b].tolist()
            ids1 = input_ids_after[b].tolist()
            toks0 = self._tokenize_ids(ids0)
            toks1 = self._tokenize_ids(ids1)

            # truncate for readability
            show_L = min(int(max_tokens_per_sample), L)
            # make sure we keep masked positions if truncating
            # if masked exists beyond show_L, bump show_L
            if show_L < L:
                last_mask = int(torch.nonzero(masked_pos, as_tuple=False).max().item())
                show_L = min(L, max(show_L, last_mask + 1))

            toks0_show = toks0[:show_L]
            toks1_show = toks1[:show_L]
            valid_show = valid[:show_L]
            masked_show = masked_pos[:show_L]

            orig_sent = self._safe_decode(ids0[:show_L], valid_mask=valid_show)
            after_sent = self._safe_decode(ids1[:show_L], valid_mask=valid_show)

            raw = None
            if raw_texts is not None:
                try:
                    raw = str(raw_texts[b])
                except Exception:
                    raw = None

            # ---- metrics
            metric_lines = []
            if saliency_norm is not None:
                s = saliency_norm[b]
                s_all = float(s[valid].mean().item()) if n_valid > 0 else 0.0
                s_m = float(s[masked_pos].mean().item()) if n_masked > 0 else 0.0
                ratio = s_m / (s_all + 1e-6)
                metric_lines.append(f"  - sal_mean_all={s_all:.6f} | sal_mean_masked={s_m:.6f} | ratio={ratio:.4f}")

                k = min(int(topk_tokens), max(1, n_valid))
                s2 = s.clone()
                s2[~valid] = -1e9
                topk = torch.topk(s2, k=k, largest=True).indices
                hit = float(masked_pos[topk].float().mean().item())
                metric_lines.append(f"  - top{k}_masked_hit_rate={hit:.4f}")

                # extra: correlation-ish proxy (masked above median?)
                med = float(s[valid].median().item()) if n_valid > 0 else 0.0
                high_mask_rate = float((masked_pos & (s >= med)).float().sum().item() / max(1, n_masked))
                metric_lines.append(f"  - masked_in_sal>=median_rate={high_mask_rate:.4f}")

            # stopword hit rate (if you added mlm_stopword_ids)
            stop_ids = getattr(self, "mlm_stopword_ids", None)
            if stop_ids:
                is_stop = torch.zeros_like(input_ids_before[b], dtype=torch.bool)
                for sid in stop_ids:
                    is_stop |= (input_ids_before[b] == int(sid))
                sw_hit = float((masked_pos & is_stop).float().sum().item() / max(1, n_masked))
                metric_lines.append(f"  - masked_stopword_rate={sw_hit:.4f}")

            # legacy layer delta stats (optional)
            layer_lines = []
            if (layer_deltas is not None) and (layer_indices is not None):
                means = []
                for li, layer_id in enumerate(layer_indices):
                    d = layer_deltas[li, b]  # [L]
                    m = float(d[valid].mean().item()) if n_valid > 0 else 0.0
                    means.append(m)
                    layer_lines.append(f"  - layer {int(layer_id):02d}: mean_delta={m:.6f}")
                total_mean = sum(means) / max(1, len(means))
                total_sum = sum(means)
                layer_lines.insert(0, f"  - mean_over_layers={total_mean:.6f} | sum_over_layers={total_sum:.6f}")

            # ---- masked token list
            masked_idx = torch.nonzero(masked_pos, as_tuple=False).squeeze(-1).tolist()
            if isinstance(masked_idx, int):
                masked_idx = [masked_idx]
            masked_idx = masked_idx[: int(max_masked_list)]

            token_lines = []
            for j in masked_idx:
                p = self._maybe_get_scalar(probability_matrix, b, j, default=-1.0)
                s = self._maybe_get_scalar(saliency_norm, b, j, default=-1.0)
                bar = self._ascii_bar(s if s >= 0 else 0.0)
                token_lines.append(
                    f"  - pos={int(j):03d} {toks0[j]} -> {toks1[j]} | p={p:.4f} | sal={s:.4f} {bar}"
                )

            # ---- TXT block
            bt = []
            bt.append("-" * 100)
            bt.append(f"Sample #{int(b)} | valid={n_valid} | masked={n_masked} ({n_masked/max(1,n_valid):.3f}) | step={int(step)}")
            if raw:
                bt.append(f"RAW : {raw}")
            bt.append(f"ORIG: {orig_sent}")
            bt.append(f"MASK: {after_sent}")

            if layer_lines:
                bt.append("NORMS (legacy hidden-delta):")
                bt.extend(layer_lines)

            if metric_lines:
                bt.append("METRICS:")
                bt.extend(metric_lines)

            bt.append("MASKED TOKENS (pos / before->after / prob / saliency):")
            bt.extend(token_lines)
            bt.append("")
            blocks_txt.append("\n".join(bt))

            # ---- JSONL record
            if write_jsonl:
                rec = {
                    "epoch": int(epoch),
                    "step": int(step),
                    "sample": int(b),
                    "n_valid": int(n_valid),
                    "n_masked": int(n_masked),
                    "orig": orig_sent,
                    "masked": after_sent,
                    "raw": raw,
                    "masked_positions": [int(j) for j in masked_idx],
                    "masked_tokens_before": [toks0[int(j)] for j in masked_idx],
                    "masked_tokens_after": [toks1[int(j)] for j in masked_idx],
                }
                if saliency_norm is not None:
                    rec["saliency_masked"] = [self._maybe_get_scalar(saliency_norm, b, int(j)) for j in masked_idx]
                if probability_matrix is not None:
                    rec["prob_masked"] = [self._maybe_get_scalar(probability_matrix, b, int(j)) for j in masked_idx]
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            written += 1

        # write TXT (append)
        if len(blocks_txt) > 0:
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write("\n".join(blocks_txt) + "\n")
            self._dbg_written_per_epoch_v2[int(epoch)] = int(written)
