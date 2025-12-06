import os
import json
import torch

try:
    from .eval import itm_eval, evaluation
except Exception:
    from .eval import itm_eval, evaluation  # type: ignore


def evaluate_and_checkpoint(
    *,
    model_without_ddp,
    test_loader,
    device,
    config,
    args,
    optimizer,
    scheduler,
    epoch: int,
    best: float,
    best_epoch: int,
    output_dir: str,
    logger,
    tb_writer=None,
):
    """运行验证并根据 r1 更新最优模型。返回 (best, best_epoch, best_log:dict)。"""
    test_result = {}
    best_log = {}

    score_test_t2i = evaluation(
        model_without_ddp, test_loader, model_without_ddp.tokenizer, device, config, args
    )
    test_result = itm_eval(
        score_test_t2i, test_loader.dataset.img2person, test_loader.dataset.txt2person, args.eval_mAP
    )

    logger.info(f"Test result: {test_result}")

    # 写日志
    log_stats = {'epoch': epoch, **{f'test_{k}': v for k, v in test_result.items()}}
    with open(os.path.join(output_dir, "log.txt"), "a") as f:
        f.write(json.dumps(log_stats) + "\n")

    # TB 可视化
    if tb_writer is not None:
        for key, value in test_result.items():
            tb_writer.add_scalar(f"Eval/{key}", value, epoch)

    # 保存最优
    if test_result.get('r1', -float('inf')) > best:
        best = test_result['r1']
        best_epoch = epoch
        best_log = log_stats
        save_obj = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'config': config,
            'epoch': epoch,
            'best': best,
            'best_epoch': best_epoch,
        }
        os.makedirs(os.path.join(output_dir, "checkpoint"), exist_ok=True)
        torch.save(save_obj, os.path.join(output_dir, "checkpoint", 'checkpoint_best.pth'))
        

    return best, best_epoch, best_log