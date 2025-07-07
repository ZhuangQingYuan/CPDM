import random
import numpy as np
import logging
import os
import pandas as pd
import torch


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    return cmap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()


# def init_log(name, level=logging.INFO):
#     if (name, level) in logs:
#         return
#     logs.add((name, level))
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     ch = logging.StreamHandler()
#     ch.setLevel(level)
#     print(os.environ)
#     if "SLURM_PROCID" in os.environ:
#         rank = int(os.environ["SLURM_PROCID"])
#         logger.addFilter(lambda record: rank == 0)
#     else:
#         rank = 0
#     format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
#     formatter = logging.Formatter(format_str)
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)
#     return logger

def init_log(name, level=logging.INFO, log_file=None):
    if (name, level) in logs:
        return logging.getLogger(name)
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file:
        # 确保目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def label_onehot(inputs, num_class):
    '''
    inputs is class label
    return one_hot label
    dim will be increasee
    '''
    batch_size, image_h, image_w = inputs.shape
    device = inputs.device
    masked_inputs = torch.where(inputs == 255,torch.tensor(-1, device=device),inputs)
    outputs = torch.zeros([batch_size, num_class, image_h, image_w], device=device)
    valid_mask = (masked_inputs >= 0) & (masked_inputs < num_class)
    valid_indices = masked_inputs[valid_mask].long()
    if valid_indices.numel() > 0:
        # 计算对应的三维索引
        batch_indices, h_indices, w_indices = torch.where(valid_mask)
        outputs[batch_indices, valid_indices, h_indices, w_indices] = 1.0
    return outputs

    # batch_size, image_h, image_w = inputs.shape
    # device = inputs.device
    # inputs = torch.relu(inputs)
    # outputs = torch.zeros([batch_size, num_class, image_h, image_w]).to(inputs.device)

    # test = inputs.cpu().numpy()
    # with pd.ExcelWriter('test_output.xlsx', engine='openpyxl') as writer:
    #     for i in range(batch_size):
    #         # Flatten each sample and save it as a separate sheet in the Excel file
    #         test_flat = test[i].reshape(image_h, image_w)
    #         df = pd.DataFrame(test_flat)
    #         # Write the dataframe to a new sheet named 'Sample_X'
    #         df.to_excel(writer, sheet_name=f'Sample_{i + 1}', index=False, header=False)
    # return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)



def set_seed(seed):
    random.seed(seed) 
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU，设置所有CUDA设备的种子
    torch.use_deterministic_algorithms(True, warn_only=True)  # 确保cudnn的确定性、仅警告而非终止
    # torch.backends.cudnn.enabled = False  # 禁用cudnn
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False  # 关闭cudnn的基准测试模式