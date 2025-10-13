import os
import torch

class DebugMaskMixin:
    @torch.no_grad()
    def debug_render_mask_diff(
        self,
        epoch: int,
        input_ids_before: torch.Tensor,   # [B, L]
        input_ids_after: torch.Tensor,    # [B, L]
        targets: torch.Tensor,            # [B, L]
        attention_mask: torch.Tensor,     # [B, L]
        raw_texts=None,
        limit_per_epoch: int = 50,
        out_path: str = None,
    ) -> None:
        """
        将原句、掩码后句子、被 mask 的词及替换情况**追加**写入同一个文件。
        分布式时仅 rank 0 写；每个 epoch 最多写 limit_per_epoch 条。
        """
        if torch.distributed.is_initialized():
            try:
                if torch.distributed.get_rank() != 0:
                    return
            except Exception:
                pass

        if out_path is None:
            out_path = getattr(self, "debug_mask_file", None)
            if out_path is None:
                out_path = getattr(self, "config_debug_mask_file", None) or "./mask_debug/mask_debug_all.txt"

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if not hasattr(self, "_dbg_written_per_epoch"):
            self._dbg_written_per_epoch = {}

        written = int(self._dbg_written_per_epoch.get(int(epoch), 0))
        if written >= limit_per_epoch:
            return

        B, L = input_ids_before.shape
        to_write_blocks = []

        if written == 0:
            sep = "=" * 100
            to_write_blocks.append(f"\n{sep}\n[Epoch {int(epoch)}]  Mask Debug\n{sep}\n")

        for b in range(B):
            if written + len(to_write_blocks) - (1 if written == 0 else 0) >= limit_per_epoch:
                break

            valid = attention_mask[b].bool()
            masked_pos = (targets[b] != -100) & valid
            if masked_pos.sum().item() == 0:
                continue

            ids_before = input_ids_before[b].tolist()
            ids_after  = input_ids_after[b].tolist()

            try:
                orig_sent  = self.tokenizer.decode([t for i,t in enumerate(ids_before) if valid[i]], skip_special_tokens=True)
                after_sent = self.tokenizer.decode([t for i,t in enumerate(ids_after)  if valid[i]], skip_special_tokens=True)
            except Exception:
                orig_sent  = self.tokenizer.decode(ids_before, skip_special_tokens=False)
                after_sent = self.tokenizer.decode(ids_after,  skip_special_tokens=False)

            pos_idx = torch.nonzero(masked_pos, as_tuple=False).squeeze(-1).tolist()
            if isinstance(pos_idx, int):
                pos_idx = [pos_idx]

            diff_lines = []
            for j in pos_idx:
                t0 = ids_before[j]
                t1 = ids_after[j]
                tok0 = self.tokenizer.convert_ids_to_tokens(int(t0))
                tok1 = self.tokenizer.convert_ids_to_tokens(int(t1))
                diff_lines.append(f"(pos={j}) {tok0}  ->  {tok1}")

            raw = None
            if raw_texts is not None:
                try:
                    raw = str(raw_texts[b])
                except Exception:
                    raw = None

            block = []
            block.append("-" * 80)
            block.append(f"Sample #{b}")
            if raw:
                block.append(f"RAW : {raw}")
            block.append(f"ORIG: {orig_sent}")
            block.append(f"MASK: {after_sent}")
            block.append("MASKED TOKENS:")
            for dl in diff_lines:
                block.append("  - " + dl)
            block.append("")
            to_write_blocks.append("\n".join(block))

        if to_write_blocks:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write("\n".join(to_write_blocks) + "\n")
            added = len(to_write_blocks)
            if written == 0:
                added -= 1
            self._dbg_written_per_epoch[int(epoch)] = written + max(0, added)

        if not hasattr(self, "config_debug_mask_file"):
            try:
                if hasattr(self, "tokenizer") and hasattr(self, "__dict__"):
                    pass
            except Exception:
                pass