import numpy as np
import torch
import torch.nn.functional as F

def generate_cutmix_mask(img_size, ratio=2):
    """生成CutMix矩形区域掩码"""
    H, W = img_size
    cut_area = H * W // ratio
    aspect_ratio = np.random.uniform(0.5, 2.0)
    w = int(np.sqrt(cut_area * aspect_ratio))
    h = int(cut_area / w)
    w, h = min(w, W), min(h, H)
    x_start = np.random.randint(0, W - w + 1)
    y_start = np.random.randint(0, H - h + 1)
    mask = torch.ones((H, W), dtype=torch.long)
    mask[y_start:y_start + h, x_start:x_start + w] = 0
    return mask


def generate_hybrid_cutmix(img_size, epoch, total_epochs, label_ratio):
    H, W = img_size
    mask = torch.ones(H, W)
    # 1. 动态计算基础参数
    base_regions = max(1, round(4 - 10 * label_ratio))
    max_ratio = 0.5 * (1 - label_ratio) ** 1.5
    min_size = max(16, 32 * (label_ratio ** 0.5))
    # 2. 衰减策略（余弦退火）
    current_regions = base_regions * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    current_regions = int(max(1, current_regions))

    # 3. 区域生成逻辑
    for _ in range(current_regions):
        ratio = np.random.uniform(0.1, max_ratio)
        cut_area = H * W * ratio

        aspect = np.random.uniform(0.3, 3.0)
        w = int(np.sqrt(cut_area * aspect))
        h = int(cut_area / w)

        # 应用最小尺寸约束
        w = max(w, min_size)
        h = max(h, min_size)

        # 边界保护
        w = min(w, W - 1)
        h = min(h, H - 1)
        x = np.random.randint(0, max(1, W - w))
        y = np.random.randint(0, max(1, H - h))
        mask[y:y + h, x:x + w] = 0

    return mask

def hybrid_cutmix_aug(conf_w, mask_w, data_s1, data_s2, epoch, total_epochs, label_ratio,dynamic_conf_threshold,fixed_conf_threshold):
    # coming soon
    
    return torch.cat(new_conf), torch.cat(new_mask), torch.cat(new_data)

