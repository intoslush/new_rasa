# build_offline_idf.py
import re
import torch
from tqdm import tqdm
import os
from dataset import get_dataloder
from utils.options import get_args 



# 如果你 tokenizer 是 huggingface 的 AutoTokenizer / BertTokenizer，一般是这样构建：

from transformers import BertModel, BertTokenizer 


_DEFAULT_STOPWORDS = {
    # 够用版本（你也可以扩展）
    "the","a","an","of","in","on","with","and","is","are","was","were","to","for","from","by",
    "as","at","it","this","that","these","those","be","been","being","or","if","then","than",
    "into","over","under","between","during","while","about","against","up","down","out","off",
    "can","could","may","might","will","would","do","does","did","have","has","had",
}


def _word_level_filter(text: str, stopwords: set) -> str:
    """
    英文 word-level stopwords 过滤，然后再送 tokenizer。
    - 小写
    - 用正则抽取 words / numbers（把标点丢掉，避免 'the,' 这种漏过滤）
    """
    text = (text or "").lower()
    # 抽取单词/数字（按你数据模板化风格，这样足够稳定）
    words = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text)
    words = [w for w in words if w not in stopwords]
    return " ".join(words)


@torch.no_grad()
def build_df_idf_from_loader(train_loader, tokenizer, stopwords: set,
                             out_path: str,
                             idf_clamp_max: float | None = None):
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        # 有些 tokenizer vocab_size 是 len(tokenizer)
        vocab_size = len(tokenizer)

    df = torch.zeros(vocab_size, dtype=torch.long)
    N = 0

    # 特殊 token id 过滤
    special_ids = set()
    for name in ["pad_token_id", "cls_token_id", "sep_token_id", "mask_token_id", "unk_token_id"]:
        tid = getattr(tokenizer, name, None)
        if tid is not None:
            special_ids.add(int(tid))

    pbar = tqdm(train_loader, desc="Building DF/IDF (offline)", ncols=120)
    for batch in pbar:
        # batch['caption1'] 通常是 list[str]（collate 后）
        caps = batch["caption1"]
        if isinstance(caps, str):
            caps = [caps]

        for cap in caps:
            N += 1
            filtered = _word_level_filter(cap, stopwords)
            if not filtered:
                continue

            enc = tokenizer(
                filtered,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            ids = enc["input_ids"]

            # caption 内 unique
            uniq = set(ids)
            # 过滤特殊 token
            uniq = [i for i in uniq if int(i) not in special_ids]
            if uniq:
                df[torch.tensor(uniq, dtype=torch.long)] += 1

    # idf = log((N+1)/(df+1))
    idf = torch.log((df.to(torch.float32) + 0.0))  # placeholder
    idf = torch.log((torch.tensor(float(N + 1.0)) / (df.to(torch.float32) + 1.0)))

    if idf_clamp_max is not None:
        idf = idf.clamp(min=0.0, max=float(idf_clamp_max))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "N": int(N),
        "df": df,
        "idf": idf,
        "vocab_size": int(vocab_size),
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", "unknown"),
        "stopwords_size": int(len(stopwords)),
    }
    torch.save(payload, out_path)
    return payload


def main():
    args = get_args()

    # ⚠️ 离线统计建议强制单进程/非分布式，避免 DistributedSampler 只覆盖子集
    # 你如果必须在 DDP 环境跑，请单独起一个非 ddp 的 job。
    args.distributed = False

    # 复用你原始 dataloader 构造
    train_loader, _, _, _ = get_dataloder(args)

    # tokenizer 构造：尽量和训练一致
    # 你如果训练时 tokenizer 是从 build_model(args) 里拿的，建议这里也复用同样的名字
    text_encoder_name = getattr(args, "text_encoder", "bert-base-uncased")
    local_model_path = "./bert-base-uncased" 
    # 检查是否已存在模型文件 
    required_files = ["config.json", "model.safetensors", "vocab.txt"] 
    is_model_ready = all(os.path.exists(os.path.join(local_model_path, f)) for f in required_files) 
    # 如果没有模型文件就下载并保存 
    if not is_model_ready: 
        model = BertModel.from_pretrained('bert-base-uncased') 
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
        os.makedirs(local_model_path, exist_ok=True) 
        model.save_pretrained(local_model_path) 
        tokenizer.save_pretrained(local_model_path) 
    # 加载模型和 tokenizer（不论是下载的还是本地已有的） 
    model = BertModel.from_pretrained(local_model_path) 
    tokenizer = BertTokenizer.from_pretrained(local_model_path) 

    out_path = getattr(args, "idf_out", f"./idf_cache/{args.dataset_name}_train_idf.pt")
    idf_max = getattr(args, "idf_clamp_max", None)

    if os.path.exists(out_path):
        print(f"[build_offline_idf] Found existing: {out_path} (skip)")
        return

    payload = build_df_idf_from_loader(
        train_loader=train_loader,
        tokenizer=tokenizer,
        stopwords=set(_DEFAULT_STOPWORDS),
        out_path=out_path,
        idf_clamp_max=idf_max,
    )

    print(f"[build_offline_idf] Saved -> {out_path}")
    print(f"  N={payload['N']}, vocab_size={payload['vocab_size']}, stopwords={payload['stopwords_size']}")


if __name__ == "__main__":
    main()
