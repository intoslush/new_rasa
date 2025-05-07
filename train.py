import os
import os.path as op
import torch
import numpy as np
import random
import time

from dataset import get_dataloder
# from processor.processor import do_train
# from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
# from solver import build_optimizer, build_lr_scheduler
from my_model import build_model
# from utils.metrics import Evaluator
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
    # get image-text pair datasets dataloader
    # train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args)
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    

    if args.distributed:
        model=model.to(torch.device("cuda", args.local_rank))
        torch.distributed.barrier()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            # output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
        )
        logger.info("模型分布式化成功")
    else:
        model.to(device)
    #获取dataloader
    train_loader, val_loader, test_loader, cluster_lodaer = get_dataloder(args)
    # optimizer = build_optimizer(args, model)
    # scheduler = build_lr_scheduler(args, optimizer)

    # is_master = get_rank() == 0
    # checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    # evaluator = Evaluator(val_img_loader, val_txt_loader)

    # start_epoch = 1
    # if args.resume:
    #     checkpoint = checkpointer.resume(args.resume_ckpt_file)
    #     start_epoch = checkpoint['epoch']

    # do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
    
    if args.distributed:
        torch.distributed.destroy_process_group()#多卡结束
    logger.info("Test complete!")