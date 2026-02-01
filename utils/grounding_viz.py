# grounding_viz.py
import os
import math
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt

# ALBEF normalize
ALBEF_MEAN = (0.48145466, 0.4578275, 0.40821073)
ALBEF_STD  = (0.26862954, 0.26130258, 0.27577711)
import re

def _sanitize_filename(s: str, max_len: int = 80) -> str:
    """
    Make token string safe for filenames across OS.
    """
    if s is None:
        return "None"
    s = str(s).strip()
    s = s.replace(" ", "_")
    # remove path separators and weird chars
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    # keep only reasonable length
    if len(s) > max_len:
        s = s[:max_len]
    if s == "":
        s = "empty"
    return s


@torch.no_grad()
def render_token_grounding_folder_one(
    model,
    tokenizer,
    image_path: str,
    caption: str,
    image_res=384,
    num_tokens=10,
    layers=6,
    skip_wordpiece=True,
    seed=0,
    out_dir="./viz_one_sentence",
    overview_name="overview.png",
    write_caption_txt=True,
):
    """
    Output structure (per sentence):
      out_dir/
        caption.txt              (optional)
        overview.png             (grid: image + N token overlays)
        token_00_<tok>.png
        token_01_<tok>.png
        ...
    """
    device = next(model.parameters()).device
    os.makedirs(out_dir, exist_ok=True)

    # 0) write caption
    if write_caption_txt:
        cap_path = os.path.join(out_dir, "caption.txt")
        with open(cap_path, "w", encoding="utf-8") as f:
            f.write(str(caption).strip() + "\n")

    # 1) load original image
    pil = Image.open(image_path).convert("RGB")
    orig_path = os.path.join(out_dir, "image.png")
    pil.save(orig_path)

    # 2) letterbox to image_res
    rk = ResizeKeepRatioPad(size=image_res)
    pil_384, meta = rk(pil)

    # 3) preprocess tensor
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(ALBEF_MEAN, ALBEF_STD),
    ])
    img = tfm(pil_384).unsqueeze(0).to(device)

    # 4) tokenize
    tok = tokenizer([caption], padding="longest", return_tensors="pt").to(device)
    text_ids = tok["input_ids"]
    attn_mask = tok["attention_mask"]

    # 5) visual encode
    image_embeds = model.visual_encoder(img)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

    # 6) token->patch attention [Lt, P]
    attn_tok_patch = extract_token2patch_attn(
        model.text_encoder.bert, text_ids, attn_mask, image_embeds, image_atts, layers=layers
    )

    # 7) choose tokens
    choose_idx, choose_tok = choose_random_tokens(
        tokenizer, text_ids[0], attn_mask[0],
        k=num_tokens, skip_wordpiece=skip_wordpiece, seed=seed
    )

    # 8) base image in original resolution (no stretching)
    base = np.asarray(pil).astype(np.float32) / 255.0  # [H,W,3]

    # 9) patch grid meta
    gh, gw, ps = get_timm_vit_patch_grid(model.visual_encoder, image_res)
    P = gh * gw
    if attn_tok_patch.size(-1) != P:
        raise RuntimeError(
            f"Patch count mismatch: attn P={attn_tok_patch.size(-1)} vs grid {gh}x{gw}={P}. patch_size={ps}"
        )

    # ---------- A) save overview grid (keep your original behavior) ----------
    overview_path = os.path.join(out_dir, overview_name)

    ncol = 1 + len(choose_idx)
    fig, axes = plt.subplots(1, ncol, figsize=(2.4*ncol, 3.6))
    if ncol == 1:
        axes = [axes]

    axes[0].imshow(base)
    axes[0].set_title("image", fontsize=10)
    axes[0].axis("off")

    for j, tidx in enumerate(choose_idx, start=1):
        hm_384 = token_patch_to_heatmap(attn_tok_patch[tidx], (gh, gw), out_hw=(image_res, image_res))
        hm_orig = unpad_and_resize_heatmap(hm_384, meta)
        hm = hm_orig.detach().cpu().numpy()

        axes[j].imshow(base)
        axes[j].imshow(hm, cmap="jet", alpha=0.45)
        axes[j].set_title(choose_tok[j-1], fontsize=10)
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(overview_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- B) save per-token images ----------
    for j, tidx in enumerate(choose_idx):
        tok_str = choose_tok[j] if j < len(choose_tok) else f"idx{tidx}"
        safe_tok = _sanitize_filename(tok_str)

        hm_384 = token_patch_to_heatmap(attn_tok_patch[tidx], (gh, gw), out_hw=(image_res, image_res))
        hm_orig = unpad_and_resize_heatmap(hm_384, meta)
        hm = hm_orig.detach().cpu().numpy()

        per_path = os.path.join(out_dir, f"token_{j:02d}_{safe_tok}.png")

        fig = plt.figure(figsize=(4.0, 4.0))
        plt.imshow(base)
        plt.imshow(hm, cmap="jet", alpha=0.45)
        plt.title(tok_str, fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(per_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return {
        "out_dir": out_dir,
        "overview_path": overview_path,
        "tokens": choose_tok,
        "token_indices": choose_idx,
        "caption_txt": os.path.join(out_dir, "caption.txt") if write_caption_txt else None,
    }

class ResizeKeepRatioPad:
    """letterbox: keep ratio -> resize -> pad to SxS; return meta for inverse mapping."""
    def __init__(self, size=384, interpolation=Image.Resampling.BICUBIC, fill=(0,0,0)):
        self.size = size
        self.interpolation = interpolation
        self.fill = fill

    def __call__(self, img: Image.Image):
        w, h = img.size
        S = self.size
        scale = min(S / w, S / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        img_r = img.resize((nw, nh), self.interpolation)
        canvas = Image.new("RGB", (S, S), self.fill)
        left = (S - nw) // 2
        top = (S - nh) // 2
        canvas.paste(img_r, (left, top))
        meta = {
            "orig_size": (w, h),
            "resize_size": (nw, nh),
            "pad": (left, top),
            "size": S,
        }
        return canvas, meta


def get_timm_vit_patch_grid(visual_encoder, image_res: int):
    """
    timm ViT: visual_encoder.patch_embed.patch_size typically exists.
    """
    ps = getattr(getattr(visual_encoder, "patch_embed", None), "patch_size", None)
    if ps is None:
        # fallback: vit-b/16 typical
        ps = 16
    if isinstance(ps, (tuple, list)):
        ps = ps[0]
    gh = image_res // int(ps)
    gw = image_res // int(ps)
    return gh, gw, int(ps)

@torch.no_grad()
def extract_token2patch_attn(
    text_encoder_bert,   # model.text_encoder.bert
    text_ids,            # [1, Lt]
    attention_mask,      # [1, Lt]
    image_embeds,        # [1, 1+P, D]
    image_atts,          # [1, 1+P]
    layers=6,
    eps=1e-6,
):
    """
    return attn_tok_patch: [Lt, P]
    """
    out = text_encoder_bert(
        input_ids=text_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
        mode="multi_modal",
    )
    cross_atts = out.cross_attentions  # tuple of [B,H,Lt,Lv]
    if cross_atts is None or len(cross_atts) == 0:
        raise RuntimeError("No cross_attentions returned. Check output_attentions=True and mode='multi_modal'.")

    sel = cross_atts[-layers:] if (layers is not None and 0 < layers < len(cross_atts)) else cross_atts
    A = torch.stack(sel, dim=0).float()      # [nL,1,H,Lt,Lv]
    A = A[..., 1:]                           # drop image CLS -> [nL,1,H,Lt,P]
    A = A / (A.sum(dim=-1, keepdim=True) + eps)  # renorm after drop CLS
    attn = A.mean(dim=0).mean(dim=1)[0]      # mean over layers & heads -> [Lt,P]
    attn = attn * attention_mask[0].unsqueeze(-1).float()
    return attn  # [Lt,P]

def token_patch_to_heatmap(attn_vec, grid_hw, out_hw=(384,384), eps=1e-6):
    """
    attn_vec: [P]
    return: [out_h,out_w] in [0,1]
    """
    gh, gw = grid_hw
    m = attn_vec.view(1,1,gh,gw)
    m = F.interpolate(m, size=out_hw, mode="bilinear", align_corners=False)[0,0]
    m = (m - m.min()) / (m.max() - m.min() + eps)
    return m

def unpad_and_resize_heatmap(hm_384, meta, eps=1e-6):
    """
    hm_384: torch [384,384]
    return: torch [H_orig, W_orig]
    """
    left, top = meta["pad"]
    nw, nh = meta["resize_size"]
    w, h = meta["orig_size"]
    hm_crop = hm_384[top:top+nh, left:left+nw]                # remove padding
    hm_crop = hm_crop[None,None]                             # [1,1,nh,nw]
    hm_orig = F.interpolate(hm_crop, size=(h,w), mode="bilinear", align_corners=False)[0,0]
    hm_orig = (hm_orig - hm_orig.min()) / (hm_orig.max() - hm_orig.min() + eps)
    return hm_orig

def choose_random_tokens(tokenizer, input_ids_1d, attn_mask_1d, k=10, skip_wordpiece=True, seed=0):
    ids = input_ids_1d.tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)

    valid = []
    for i, tk in enumerate(toks):
        if attn_mask_1d[i].item() == 0:
            continue
        if tk in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
        if skip_wordpiece and tk.startswith("##"):
            continue
        valid.append(i)

    random.seed(seed)
    if len(valid) == 0:
        return [], []
    choose = random.sample(valid, k=min(k, len(valid)))
    return choose, [toks[i] for i in choose]

@torch.no_grad()
def render_token_grounding_grid_one(
    model,              # your ALBEF
    tokenizer,
    image_path: str,
    caption: str,
    image_res=384,
    num_tokens=10,
    layers=6,
    skip_wordpiece=True,
    seed=0,
    out_path="figB.png",
):
    """
    Save a grid: [image] + [token overlays...]
    """
    device = next(model.parameters()).device

    # 1) load original image
    pil = Image.open(image_path).convert("RGB")

    # 2) letterbox to 384
    rk = ResizeKeepRatioPad(size=image_res)
    pil_384, meta = rk(pil)

    # 3) preprocess tensor (normalize)
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(ALBEF_MEAN, ALBEF_STD),
    ])
    img = tfm(pil_384).unsqueeze(0).to(device)  # [1,3,384,384]

    # 4) tokenize
    tok = tokenizer([caption], padding="longest", return_tensors="pt").to(device)
    text_ids = tok["input_ids"]
    attn_mask = tok["attention_mask"]

    # 5) visual encode -> [1, 1+P, D]
    image_embeds = model.visual_encoder(img)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

    # 6) token->patch attention [Lt,P]
    attn_tok_patch = extract_token2patch_attn(
        model.text_encoder.bert, text_ids, attn_mask, image_embeds, image_atts, layers=layers
    )

    # 7) choose tokens
    choose_idx, choose_tok = choose_random_tokens(
        tokenizer, text_ids[0], attn_mask[0], k=num_tokens, skip_wordpiece=skip_wordpiece, seed=seed
    )

    # 8) base image for plotting: use ORIGINAL (no stretching)
    base = np.asarray(pil).astype(np.float32) / 255.0  # [H,W,3]

    # 9) patch grid
    gh, gw, ps = get_timm_vit_patch_grid(model.visual_encoder, image_res)
    P = gh * gw
    if attn_tok_patch.size(-1) != P:
        raise RuntimeError(f"Patch count mismatch: attn P={attn_tok_patch.size(-1)} vs grid {gh}x{gw}={P}. patch_size={ps}")

    # 10) plot grid
    ncol = 1 + len(choose_idx)
    fig, axes = plt.subplots(1, ncol, figsize=(2.4*ncol, 3.6))
    if ncol == 1:
        axes = [axes]

    axes[0].imshow(base)
    axes[0].set_title("image", fontsize=10)
    axes[0].axis("off")

    for j, tidx in enumerate(choose_idx, start=1):
        hm_384 = token_patch_to_heatmap(attn_tok_patch[tidx], (gh, gw), out_hw=(image_res, image_res))
        hm_orig = unpad_and_resize_heatmap(hm_384, meta)  # [H_orig, W_orig]
        hm = hm_orig.detach().cpu().numpy()

        axes[j].imshow(base)
        axes[j].imshow(hm, cmap="jet", alpha=0.45)
        axes[j].set_title(choose_tok[j-1], fontsize=10)
        axes[j].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"out_path": out_path, "tokens": choose_tok, "token_indices": choose_idx}
