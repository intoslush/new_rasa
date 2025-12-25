# my_model/albef/attr_supervision.py
import re
import torch
import torch.nn.functional as F

_DEFAULT_STOPWORDS = {
    "the","a","an","of","in","on","with","and","is","are","was","were","to","for","from","by",
    "as","at","it","this","that","these","those","be","been","being","or","if","then","than",
}
_DEFAULT_WP_SUFFIX = {"##s","##es","##ed","##ing","##ly","##er","##est"}

def build_keep_token_ids_from_df(
    tokenizer,
    df: torch.Tensor,
    min_df: int = 1,
    extra_stopwords=None,
    remove_wordpiece_suffix: bool = False,
    wordpiece_suffix_list=None,
):
    """
    df: [vocab_size] long
    keep: token ids that appear in dataset (df>=min_df) AND not special AND not stopwords tokens
    optional: remove WordPiece suffix tokens like ##s/##ing/...
    """
    stopwords = set(_DEFAULT_STOPWORDS)
    if extra_stopwords:
        stopwords |= set(extra_stopwords)

    # special ids
    special_ids = set()
    for name in ["pad_token_id", "cls_token_id", "sep_token_id", "mask_token_id", "unk_token_id"]:
        tid = getattr(tokenizer, name, None)
        if tid is not None:
            special_ids.add(int(tid))

    # stopword token ids (WordPiece)
    stop_token_ids = set()
    for w in stopwords:
        ids = tokenizer(w, add_special_tokens=False)["input_ids"]
        for i in ids:
            stop_token_ids.add(int(i))

    # optional: suffix token ids
    suffix_token_ids = set()
    if remove_wordpiece_suffix:
        suffix_set = set(wordpiece_suffix_list) if wordpiece_suffix_list else set(_DEFAULT_WP_SUFFIX)
        # 注意：这些字符串必须是 tokenizer vocab 里真实存在的 token
        for tok in suffix_set:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is None:
                continue
            if int(tid) != tokenizer.unk_token_id:
                suffix_token_ids.add(int(tid))

    # df-based keep
    keep = torch.nonzero(df >= int(min_df), as_tuple=False).view(-1).tolist()

    # filter
    keep2 = []
    for tid in keep:
        tid = int(tid)
        if tid in special_ids:
            continue
        if tid in stop_token_ids:
            continue
        if remove_wordpiece_suffix and (tid in suffix_token_ids):
            continue
        keep2.append(tid)

    return sorted(set(keep2))



def build_keep_token_ids(tokenizer, attr_vocab_path: str, extra_stopwords=None):
    stopwords = set(_DEFAULT_STOPWORDS)
    if extra_stopwords:
        stopwords |= set(extra_stopwords)

    # special ids
    special_ids = set()
    for name in ["pad_token_id", "cls_token_id", "sep_token_id", "mask_token_id", "unk_token_id"]:
        tid = getattr(tokenizer, name, None)
        if tid is not None:
            special_ids.add(int(tid))

    keep = set()

    # 把 stopwords 映射到 token ids（用于剔除）
    stop_token_ids = set()
    for w in stopwords:
        ids = tokenizer(w, add_special_tokens=False)["input_ids"]
        for i in ids:
            stop_token_ids.add(int(i))

    with open(attr_vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.lower()

            # 简单规范化：把连字符当空格（你数据模板化时通常更稳）
            s = re.sub(r"[-_/]+", " ", s).strip()
            if not s:
                continue

            ids = tokenizer(s, add_special_tokens=False)["input_ids"]
            for i in ids:
                ii = int(i)
                if ii in special_ids:
                    continue
                if ii in stop_token_ids:
                    continue
                keep.add(ii)

    keep_ids = sorted(list(keep))
    return keep_ids


def make_id2col(vocab_size: int, keep_ids: list[int]) -> torch.Tensor:
    """
    id2col[token_id] = column index in [0..K-1], else -1
    """
    id2col = torch.full((vocab_size,), -1, dtype=torch.long)
    for j, tid in enumerate(keep_ids):
        id2col[tid] = j
    return id2col




def build_attr_targets(input_ids: torch.Tensor,
                       attention_mask: torch.Tensor,
                       id2col: torch.Tensor,
                       idf_keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    input_ids: [B,L]
    attention_mask: [B,L]  (pad=0)
    id2col: [vocab_size] with -1 for ignore
    idf_keep: [K]

    return:
      targets_norm: [B,K] (L1 normalized, empty rows are 0)
      non_empty: [B] bool
    """
    B, L = input_ids.shape
    device = input_ids.device

    # map token_id -> col index (-1 ignore)
    col = id2col[input_ids]                 # [B,L]
    valid = (col >= 0) & (attention_mask > 0)

    # scatter_add to count occurrences, then binarize to multi-hot
    K = int(idf_keep.numel())
    targets = torch.zeros((B, K), device=device, dtype=torch.float32)

    # 把 invalid 的 col 临时置 0，但 add 的值是 0，不会影响
    col_safe = col.clamp(min=0)
    targets.scatter_add_(dim=1, index=col_safe, src=valid.to(torch.float32))
    targets = (targets > 0).to(torch.float32)

    # idf reweight
    targets = targets * idf_keep.view(1, -1)

    # L1 normalize per sample
    s = targets.sum(dim=1, keepdim=True)            # [B,1]
    non_empty = (s.squeeze(1) > 0)
    targets_norm = torch.zeros_like(targets)
    targets_norm[non_empty] = targets[non_empty] / (s[non_empty] + 1e-6)

    return targets_norm, non_empty

def build_token_targets(input_ids, attention_mask, id2col, idf_keep):
    B, L = input_ids.shape
    col = id2col[input_ids]                       # [B,L]
    valid = (col >= 0) & (attention_mask > 0)

    K = int(idf_keep.numel())
    targets = torch.zeros((B, K), device=input_ids.device, dtype=torch.float32)

    col_safe = col.clamp(min=0)
    targets.scatter_add_(dim=1, index=col_safe, src=valid.to(torch.float32))
    targets = (targets > 0).to(torch.float32)     # multi-hot

    targets = targets * idf_keep.view(1, -1)      # idf reweight

    s = targets.sum(dim=1, keepdim=True)
    non_empty = (s.squeeze(1) > 0)

    out = torch.zeros_like(targets)
    out[non_empty] = targets[non_empty] / (s[non_empty] + 1e-6)  # L1 norm
    return out, non_empty

def soft_ce_loss(logits: torch.Tensor, targets_norm: torch.Tensor, non_empty: torch.Tensor) -> torch.Tensor:
    """
    logits: [B,K]
    targets_norm: [B,K], L1-normalized
    non_empty: [B] bool

    对 empty 的样本不计入 loss（等价 skip）
    """
    per = -(F.log_softmax(logits, dim=1) * targets_norm).sum(dim=1)  # [B]
    w = non_empty.to(per.dtype)
    loss = (per * w).sum() / (w.sum() + 1e-6)
    return loss
