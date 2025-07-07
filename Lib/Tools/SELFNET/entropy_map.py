import torch

"""文件说明：计算熵图代码"""
def generate_entropy_map(prob):
    """
    :param prob: 模型输出的预测概率值 通道进行softmax之后 (1,6,320,320)
    """
    entropy_map = - torch.sum(prob * torch.log(prob + 1e-10), dim=1)
    return entropy_map

def calc_entropy_map(inputs):
    if type(inputs) is list:
        temp_var = None
        for aux in inputs:
            entropy_map = generate_entropy_map(aux.softmax(dim=1))
            if temp_var is None:
                temp_var = entropy_map
            else:
                temp_var = torch.cat((temp_var, entropy_map), dim=0)
        all_entropy_map,_ = torch.min(temp_var, dim=0)
    else:
        all_entropy_map = generate_entropy_map(inputs.softmax(dim=1))
    return all_entropy_map
