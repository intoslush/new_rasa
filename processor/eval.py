import torch
import utils.optimizer as utils
import time
import torch.nn.functional as F
import datetime
import torch.distributed as dist

@torch.no_grad()
def itm_eval(scores_t2i, img2person, txt2person, eval_mAP):
    img2person = torch.tensor(img2person)
    txt2person = torch.tensor(txt2person)
    index = torch.argsort(scores_t2i, dim=-1, descending=True)
    pred_person = img2person[index]
    matches = (txt2person.view(-1, 1).eq(pred_person)).long()

    def acc_k(matches, k=1):
        matches_k = matches[:, :k].sum(dim=-1)
        matches_k = torch.sum((matches_k > 0))
        return 100.0 * matches_k / matches.size(0)

    # Compute metrics
    ir1 = acc_k(matches, k=1).item()
    ir5 = acc_k(matches, k=5).item()
    ir10 = acc_k(matches, k=10).item()
    ir_mean = (ir1 + ir5 + ir10) / 3

    if eval_mAP:
        real_num = matches.sum(dim=-1)
        tmp_cmc = matches.cumsum(dim=-1).float()
        order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long)
        tmp_cmc /= order
        tmp_cmc *= matches
        AP = tmp_cmc.sum(dim=-1) / real_num
        mAP = AP.mean() * 100.0
        eval_result = {'r1': ir1,
                       'r5': ir5,
                       'r10': ir10,
                       'r_mean': ir_mean,
                       'mAP': mAP.item()
                       }
    else:
        eval_result = {'r1': ir1,
                       'r5': ir5,
                       'r10': ir10,
                       'r_mean': ir_mean,
                       }
    return eval_result

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config,args):
    # evaluate
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing features for evaluation...')
    start_time = time.time()
    # extract text features
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_words'], return_tensors="pt").to(device)
        text_output = model.text_encoder.bert(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    # extract image features
    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    # compute the feature similarity score for all image-text pairs
    sims_matrix = text_embeds @ image_embeds.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)
    # take the top-k candidates and calculate their ITM score sitm for ranking
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        topk_idx = topk_idx.to(image_feats.device)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder.bert(encoder_embeds=text_feats[start + i].repeat(config['k_test'], 1, 1),
                                         attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                         encoder_hidden_states=encoder_output.to(device),
                                         encoder_attention_mask=encoder_att,
                                         return_dict=True,
                                         mode='fusion'
                                         )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score
    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    return score_matrix_t2i.cpu()