from math import cos, pi
from typing import Dict, Optional

def _interp(base_v: float, target_v: float, t: float, mode: str = "linear") -> float:
    t = max(0.0, min(1.0, t))
    if mode == "cosine":
        # t=0 为 base，t=1 为 target，余弦缓入缓出
        return target_v + (base_v - target_v) * 0.5 * (1.0 + cos(pi * t))
    return base_v + (target_v - base_v) * t


def compute_dynamic_weights(
    epoch: int,
    num_epoch: int,
    base_weights: Dict[str, float],
    schedule: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    base_weights: 例如 {"loss_cl":0.5, "loss_pitm":1, "loss_mlm":1, "loss_prd":0.5, "loss_mrtd":0.5}
    schedule 示例：
    {
      "freeze_epochs": 20,
      "mode": "linear" | "cosine",
      "loss_cl": {"min": 0.1},
      "loss_mlm": {"max": 1.0},
      "loss_pitm": {"max": 0.8},
      "normalize_sum": true
    }
    注意：为保证行为与旧版一致，这里保留 freeze=99 的覆盖（若你希望按配置生效，删除该行即可）。
    """
    if schedule is None:
        schedule = {}

    freeze = int(schedule.get("freeze_epochs", 20))
    mode = schedule.get("mode", "linear")
    # 冻结阶段：直接返回初始权重
    if epoch <= freeze:
        return dict(base_weights)

    # 进度 t：从冻结结束到训练结束线性映射到 [0,1]
    denom = max(1, (num_epoch - freeze))
    t = (epoch - freeze) / denom
    t = max(0.0, min(1.0, t))

    w = dict(base_weights)

    # loss_cl：下降到最小值
    cl_cfg = schedule.get("loss_cl", {})
    if "min" in cl_cfg:
        w["loss_cl"] = _interp(base_weights.get("loss_cl", 0.0), float(cl_cfg["min"]), t, mode)

    # # loss_mlm：上升到最大值
    # mlm_cfg = schedule.get("loss_mlm", {})
    # if "max" in mlm_cfg:
    #     w["loss_mlm"] = _interp(base_weights.get("loss_mlm", 0.0), float(mlm_cfg["max"]), t, mode)

    # # loss_pitm：下降/上升（旧代码写的是 "max"，保持一致）
    # pitm_cfg = schedule.get("loss_pitm", {})
    # if "max" in pitm_cfg:
    #     w["loss_pitm"] = _interp(base_weights.get("loss_pitm", 0.0), float(pitm_cfg["max"]), t, mode)



    return w