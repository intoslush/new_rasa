import torch.nn.functional as F
from .faiss_rerank import compute_jaccard_distance
from sklearn.cluster import DBSCAN
import torch

def cluster_begin_epoch(train_loader, model, args,tokenizer = None,logger = None):
    device = "cuda"
    feature_size =256 #cuhk是577,融合之后是768
    max_size = len(train_loader.dataset)  #这个是所有的图片和描述对的数量共计6800对左右     
    image_bank = torch.zeros((max_size, feature_size)).to(device)
    index = 0

    model.to(device)
    model = model.eval()
    #TODO这玩意我以后一定改
    logger.info("开始计算伪标签")
    with torch.no_grad():
        if args.distributed:
            model=model.module
        for i,batch in enumerate(train_loader):
            image1=batch['image1']
            image2=batch['image2']
            text1=batch['caption1']
            text2=batch['caption2']
            idx=batch['person_id']
            replace=batch['replace_flag']
            pseudo_label=batch['pseudo_label']
            image1 = image1.to(device, non_blocking=True)
            # image2 = image2.to(device, non_blocking=True)
            # idx = idx.to(device, non_blocking=True)
            replace = replace.to(device, non_blocking=True)
            # text_input1 = tokenizer(text1, padding='longest', max_length=config['max_words'], return_tensors="pt").to(device)
            # text_input2 = tokenizer(text2, padding='longest', max_length=config['max_words'], return_tensors="pt").to(device)
            
            image_embeds = model.visual_encoder(image1)#(13,577,768)
            # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image1.device)#注意力掩码全一表示所有图像token都应该被关注
            image_feat = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)#用于取cls token的特征,shape(13,577)
            # extract text features
            # text_output = model.text_encoder.bert(text_input2.input_ids, attention_mask=text_input2.attention_mask,
                                                # return_dict=True, mode='text')
            # text_embeds = text_output.last_hidden_state
            # text_feat = F.normalize(model.text_proj(text_embeds[:, 0, :]), dim=-1)#同样是取cls token的特征
            batch_size = image1.shape[0]
            
            # output_pos =model.text_encoder.bert(encoder_embeds=text_embeds,
            #                                 attention_mask=text_input2.attention_mask,
            #                                 encoder_hidden_states=image_embeds,
            #                                 encoder_attention_mask=image_atts,
            #                                 return_dict=True,
            #                                 mode='fusion',
            #                                 )
            # fusion_feat=output_pos.last_hidden_state[:, 0, :]#shape(13,768)
            image_bank[index: index + batch_size] = image_feat
            index = index + batch_size
            

        image_bank = image_bank[:index]  
        if args.distributed:
            logger.info(f"Rank {torch.distributed.get_rank()} | 开始计算不同类之间的距离") 
        else:
            logger.info(f"单卡Rank 0 | 开始计算不同类之间的距离")   

        try:
            
            image_rerank_dist = compute_jaccard_distance(image_bank, k1=30, k2=6, search_option=3)  
        except Exception as e:
            logger.info(f" 计算距离出错：{e}")     
        # image_rerank_dist = compute_jaccard_distance(image_bank, k1=30, k2=6, search_option=0)  

        # DBSCAN cluster
        cluster = DBSCAN(eps= 0.6, min_samples=4, metric='precomputed', n_jobs=-1)

        image_pseudo_labels = cluster.fit_predict(image_rerank_dist)    

        del image_rerank_dist
        # ✅ 打印统计信息
        dataset_len = len(train_loader.dataset)
        num_noise = (image_pseudo_labels == -1).sum()
        num_clusters = len(set(image_pseudo_labels)) - (1 if -1 in image_pseudo_labels else 0)
        logger.info(f"Dataset 总长度: {dataset_len},最终输出的伪标签长度: {len(image_pseudo_labels)}")
        logger.info(f"聚类数（不含 -1）: {num_clusters}")
        logger.info(f"-1 (未归入任何簇) 的数量: {num_noise}\n")
    del image_bank
    return image_pseudo_labels