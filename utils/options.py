import argparse
import numpy as np
import random
import torch
import os
def get_args():
    parser = argparse.ArgumentParser(description="ICPG Args")
    parser.add_argument('--config', default='./configs/PS_cuhk_pedes.yaml')
    # parser.add_argument('--output_dir', default='output/cuhk-pedes')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval_mAP', action='store_true', help='whether to evaluate mAP')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--batch_size', default=13, type=int)
    parser.add_argument('--embed_dim', default=577, type=int)
    parser.add_argument('--tokenizer_path', default="./bert-base-uncased", type=str)
    parser.add_argument('--dataset_name', default="CUHK-PEDES", type=str, help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument('--root_dir',default="./re_id", type=str)
    parser.add_argument('--num_epoch', default=30   , type=int)

    ######################## general settings ########################
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    # parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    # ######################## model general settings ########################
    # parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model  
    # parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    # parser.add_argument("--img_aug", default=False, action='store_true')
    # parser.add_argument("--embed_dim", type=int, default=512)
    # parser.add_argument("--e_l", type=int, default=20)
    # parser.add_argument("--margin", type=float, default=0.3)

    # ## cross modal transfomer setting
    # parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    # parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    # parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    # parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    # parser.add_argument("--MLM", default=False, action='store_true', help="whether to use Mask Language Modeling dataset")

    # ######################## loss settings ########################
    # parser.add_argument("--loss_names", default='itc+cdm+chm', help="which loss to use ['cdm', 'chm, 'itc']")
    # parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    
    # ######################## vison trainsformer settings ########################
    # parser.add_argument("--img_size", type=tuple, default=(384, 128))
    # parser.add_argument("--stride_size", type=int, default=16) #########  

    # ######################## text transformer settings ########################
    # parser.add_argument("--text_length", type=int, default=77)  ########  77
    # parser.add_argument("--vocab_size", type=int, default=49408)

    # ######################## solver ########################
    # parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    # parser.add_argument("--lr", type=float, default=1e-5)
    # parser.add_argument("--bias_lr_factor", type=float, default=2.)
    # parser.add_argument("--weight_decay", type=float, default=4e-5)
    # parser.add_argument("--weight_decay_bias", type=float, default=0.)
    # parser.add_argument("--alpha", type=float, default=0.9)
    # parser.add_argument("--beta", type=float, default=0.999)
    
    # ######################## scheduler ########################
    # parser.add_argument("--num_epoch", type=int, default=60)
    # parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    # parser.add_argument("--gamma", type=float, default=0.1)
    # parser.add_argument("--warmup_factor", type=float, default=0.1)
    # parser.add_argument("--warmup_epochs", type=int, default=5)
    # parser.add_argument("--warmup_method", type=str, default="linear")
    # parser.add_argument("--lrscheduler", type=str, default="cosine")
    # parser.add_argument("--target_lr", type=float, default=0)
    # parser.add_argument("--power", type=float, default=0.9)

    # ######################## dataset ########################
    # parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    # parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    # parser.add_argument("--num_instance", type=int, default=4)
    # parser.add_argument("--root_dir", default="./data")
    # parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args()

    return args

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True