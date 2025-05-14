from my_model import objectives 
from .model_person_search import ALBEF 
import torch 
import torch.nn as nn 
from collections import OrderedDict 
import torch.nn.functional as F 
from transformers import BertModel, BertTokenizer 
import os 
import ruamel.yaml as YAML 
from .vit import interpolate_pos_embed 
import logging


 
def bulid_tokenizer(tokenizer_path="./bert-base-uncased",logger=None): 
    # 设置本地模型路径 
    local_model_path = "./bert-base-uncased" 
    # 检查是否已存在模型文件 
    required_files = ["config.json", "model.safetensors", "vocab.txt"] 
    is_model_ready = all(os.path.exists(os.path.join(local_model_path, f)) for f in required_files) 
    # 如果没有模型文件就下载并保存 
    if not is_model_ready: 
        logger.info("本地模型不存在或不完整，正在从 Hugging Face 下载...") 
        model = BertModel.from_pretrained('bert-base-uncased') 
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
        os.makedirs(local_model_path, exist_ok=True) 
        model.save_pretrained(local_model_path) 
        tokenizer.save_pretrained(local_model_path) 
    else: 
        logger.info("发现本地模型，直接加载。") 
    # 加载模型和 tokenizer（不论是下载的还是本地已有的） 
    model = BertModel.from_pretrained(local_model_path) 
    tokenizer = BertTokenizer.from_pretrained(local_model_path) 
    logger.info("模型和 tokenizer 加载成功。") 
    return tokenizer 
    
 
def build_model(args): 
    logger = logging.getLogger(args.name)
    tokenizer = bulid_tokenizer(args.tokenizer_path,logger) 
    yaml = YAML.YAML(typ='rt') 
    config = yaml.load(open(args.config, 'r')) 
    model = ALBEF(config=config, text_encoder=args.tokenizer_path, tokenizer=tokenizer) 
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu') 
    state_dict = checkpoint['model'] 
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder) 
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped 
    m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], 
                                                model.visual_encoder_m) 
    state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped  
    msg = model.load_state_dict(state_dict, strict=False, assign=True) 
    logger.info('load checkpoint from %s' % args.checkpoint) 
    logger.info(msg) 
    return model
