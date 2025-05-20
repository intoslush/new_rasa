import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from .cluster import cluster_begin_epoch
import ruamel.yaml as YAML 
import utils.optimizer as utils
from optim import create_optimizer
from scheduler import create_scheduler
from .eval import itm_eval, evaluation
import os
import json
import torch.distributed as dist

def do_train(start_epoch, args, model, train_loader, evaluator, checkpointer,cluster_loader,test_loader):
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger(args.name)
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "cdm_loss": AverageMeter(),
        "chm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    yaml = YAML.YAML(typ='rt') 
    config = yaml.load(open(args.config, 'r')) 
    # Optimizer and learning rate scheduler
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    scheduler, _ = create_scheduler(arg_sche, optimizer)

    # train config
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0
    best_log = ''
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    # 开始eval
    if args.distributed:
        model_without_ddp=  model.module
    else:
        model_without_ddp = model
    # train
    for epoch in range(start_epoch, num_epoch + 1):
        # 生成为标签,并筛选处理sample
        cluster_loader.dataset.mode = 'cluster'
        image_pseudo_labels = cluster_begin_epoch(cluster_loader, model, args,None,logger)
        image_num_cluster = len(set(image_pseudo_labels)) - (1 if -1 in image_pseudo_labels else 0)
        logger.info("==> Statistics for epoch [{}]: {} image clusters,total{}".format(epoch, image_num_cluster,len(image_pseudo_labels)))
        train_loader.dataset.mode = 'train'
        train_loader.dataset.set_pseudo_labels(image_pseudo_labels)
        if epoch > 0:
            scheduler.step(epoch + warmup_steps)
        if args.distributed:
            dist.barrier()
            train_loader.sampler.set_valid_indices(train_loader.dataset.valid_indices)
            train_loader.sampler.set_epoch(epoch)


        # 开始每个batch的训练
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()
        logger.info("开始epoch {}的mini batch循环".format(epoch))
        for n_iter, batch in enumerate(train_loader):
            batch = {
                k: v.to(device) if hasattr(v, 'to') else v
                for k, v in batch.items()
            }
            if epoch > 0 or not config['warm_up']:
                alpha = config['alpha']
            else:
                alpha = config['alpha'] * min(1.0, n_iter / len(train_loader))
            # logger.info("开始poch {}的第{}个batch的loss计算".format(epoch, n_iter))   
            loss_cl, loss_pitm, loss_mlm, loss_prd, loss_mrtd = model(batch,alpha,config) 
            # logger.info(print(f"{loss_cl.requires_grad}, {loss_pitm.requires_grad}, {loss_mlm.requires_grad}, {loss_prd.requires_grad} ,{loss_mrtd.requires_grad}"))
            # logger.info("完成epoch {}的第{}个batch的loss计算".format(epoch, n_iter)) 
            # 计算总损失
            loss = 0.
            for j, los in enumerate((loss_cl, loss_pitm, loss_mlm, loss_prd, loss_mrtd)):
                loss += config['weights'][j] * los
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == 0 and n_iter % step_size == 0 and n_iter <= warmup_iterations:
                scheduler.step(n_iter // step_size)

        logger.info("----------epoch {}训练完成-------------".format(epoch))
        if epoch >= config['eval_epoch'] or args.evaluate:
            score_test_t2i = evaluation(model_without_ddp, test_loader, model_without_ddp.tokenizer, device, config,args)
            if utils.is_main_process():
                test_result = itm_eval(score_test_t2i, test_loader.dataset.img2person, test_loader.dataset.txt2person, args.eval_mAP)
                print('Test:', test_result, '\n')
                if args.evaluate:
                    log_stats = {'epoch': epoch,
                                 **{f'test_{k}': v for k, v in test_result.items()}
                                 }
                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                else:
                    log_stats = {'epoch': epoch,
                                 **{f'test_{k}': v for k, v in test_result.items()},
                                 }
                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'best': best,
                        'best_epoch': best_epoch
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_epoch%02d.pth' % epoch))
                    if test_result['r1'] > best:
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                        best = test_result['r1']
                        best_epoch = epoch
                        best_log = log_stats

        dist.barrier()
        torch.cuda.empty_cache()

        




