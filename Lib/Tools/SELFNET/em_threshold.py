import numpy as np
import torch


def get_class_adaptive_threshold(em_u_s1, mask_u_w, num_classes,percent):
    """
    计算类别自适应阈值

    Args:
        em_u_s1 (torch.Tensor): 熵图，shape [H, W]
        mask_u_w (torch.Tensor): 伪标签，shape [H, W]
        num_classes (int): 类别数
        epoch (int): 当前 epoch
        total_epochs (int): 总 epoch 数

    Returns:
        mask_reliable (torch.Tensor): 可靠区域掩码，shape [H, W]
    """
    # 初始化类别阈值
    class_thresholds = np.full(num_classes, np.inf)
    em_u_s1_np = em_u_s1.detach()  # 分离梯度
    mask_u_w_np = mask_u_w.detach()  # 分离梯度
    for c in range(num_classes):
        # 获取当前类别的熵值（分离梯度后操作）
        class_mask = (mask_u_w_np == c)
        class_entropy = em_u_s1_np[class_mask].cpu().numpy()  # 安全转换为 numpy
        if len(class_entropy) > 0:
            class_thresholds[c] = np.percentile(class_entropy, percent)
        # 生成阈值图（直接使用原始设备上的 Tensor）
    class_thresholds_tensor = torch.tensor(class_thresholds, dtype=em_u_s1.dtype, device=em_u_s1.device)
    threshold_map = class_thresholds_tensor[mask_u_w.long()]  # 使用原始 mask_u_w 索引
    # 可靠区域掩码
    mask_reliable = (em_u_s1 <= threshold_map)
    return mask_reliable