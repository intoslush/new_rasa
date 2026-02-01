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
    # === 新增功能 START: 保存 Top-10 预测结果到 JSON ===
    if True:
        logger.info("Saving Top-10 visualization results to json...")
        
        dataset = test_loader.dataset
        
        # 1. 获取完整图片路径列表
        # 根据你的 dataset 代码，self.image 存的是文件名，self.image_root 是根目录
        try:
            full_img_paths = [os.path.join(dataset.image_root, p) for p in dataset.image]
        except AttributeError:
            # 兼容性处理：如果 dataset 被封装了一层（如 Subset），尝试获取原 dataset
            if hasattr(dataset, 'dataset'):
                full_img_paths = [os.path.join(dataset.dataset.image_root, p) for p in dataset.dataset.image]
            else:
                logger.error("Error: Cannot find .image or .image_root in dataset")
                full_img_paths = []

        txt2person = dataset.txt2person
        img2person = dataset.img2person
        texts = dataset.text

        topk_results = []
        
        # 获取 Top-10 的索引
        scores, indices = torch.topk(score_test_t2i, k=10, dim=1)
        indices = indices.cpu().numpy()

        for txt_idx, sorted_img_indices in enumerate(indices):
            query_text = texts[txt_idx]
            query_pid = txt2person[txt_idx]
            
            match_results = []
            for rank, img_idx in enumerate(sorted_img_indices):
                img_pid = img2person[img_idx]
                is_correct = (query_pid == img_pid)
                
                # 获取该图片的完整路径
                current_img_path = full_img_paths[img_idx] if full_img_paths else "PATH_NOT_FOUND"

                match_results.append({
                    "rank": rank,
                    "img_path": current_img_path,
                    "is_correct": bool(is_correct),
                    "img_pid": int(img_pid),
                    "score": float(scores[txt_idx, rank])
                })

            topk_results.append({
                "query_id": txt_idx,
                "query_text": query_text,
                "query_pid": int(query_pid),
                "matches": match_results
            })

        # 保存为 JSON 文件
        json_path = os.path.join(output_dir, f"epoch_{epoch}_top10_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(topk_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved top-10 results to {json_path}")
    # === 新增功能 END ===
    
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