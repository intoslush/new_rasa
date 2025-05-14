import os
import os.path as op
import torch
import numpy as np
import random
import time
from dataset import get_dataloder
from processor.processor import do_train
# from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from my_model import build_model
from utils.metrics import Evaluator
from utils.options import get_args, set_seed    
from utils.comm import get_rank, synchronize

if __name__ == '__main__':
    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    logger = setup_logger(args.name, save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)
    model = build_model(args)
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    

    if args.distributed:
        model=model.to(torch.device("cuda", args.local_rank))
        torch.distributed.barrier()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
        )
        logger.info("模型分布式化成功")
    else:
        logger.info("模型初始化化成功")
        model.to(device)
    #获取dataloader
    train_loader, val_loader, test_loader, cluster_lodaer = get_dataloder(args)
    logger.info("dataloader加载成功")
    start_epoch = 1
    # if args.resume:
    #     checkpoint = checkpointer.resume(args.resume_ckpt_file)
    #     start_epoch = checkpoint['epoch']

    # do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
    do_train(start_epoch, args, model, train_loader, None, None,cluster_lodaer,test_loader)
    
    if args.distributed:
        torch.distributed.destroy_process_group()#多卡结束
    logger.info("Test complete!")