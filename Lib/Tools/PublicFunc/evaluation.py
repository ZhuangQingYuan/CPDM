import numpy as np
import torch
from Lib.Tools.PublicFunc.utils import AverageMeter, intersectionAndUnion
import torch.nn.functional as F
"""
文件说明：用于进行模型的验证
"""

def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:

            img = img.to(cfg['device'])

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).to(cfg['device'])
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
            
                if cfg['model_name']=='unimatch':
                    pred = model(img).argmax(dim=1)
                elif cfg['model_name']=='uccl':
                    pred = model(img)[1].argmax(dim=1)
                elif cfg['model_name']=='corrmatch':
                    pred = model(img)['out'].argmax(dim=1)
                elif cfg['model_name']=='selfnet':
                    pred_dict = model(img,mode='eval')
                    pred = pred_dict['out'].argmax(dim=1)
                elif cfg['model_name']=='wscl':
                    pred = model(img).argmax(dim=1)
                elif cfg['model_name']=='lsst':
                    pred = model(img).argmax(dim=1)
                elif cfg['model_name']=='fixmatch':
                    pred = model(img).argmax(dim=1)
                elif cfg['model_name']=='selfnet_v1':
                    pred_dict = model(img, mode='eval')
                    pred = pred_dict['out'].argmax(dim=1)
                elif cfg['model_name']=='scalematch':
                    pred = model(img).argmax(dim=1)
                else:
                    raise NameError('Evaluation.py中模型名称设置有误')

            intersection, union, target = intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['num_classes'], 255)
            reduced_intersection = torch.from_numpy(intersection).to(cfg['device'])
            reduced_union = torch.from_numpy(union).to(cfg['device'])
            reduced_target = torch.from_numpy(target).to(cfg['device'])

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)
    return mIOU, iou_class


def self_evaluate(model, loader, mode, cfg,**kwargs):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:

            img = img.to(cfg['device'])

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).to(cfg['device'])
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img,model = kwargs['self_para']['model'],mode = kwargs['self_para']['mode']).argmax(dim=1)

            intersection, union, target = intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['num_classes'], 255)
            reduced_intersection = torch.from_numpy(intersection).to(cfg['device'])
            reduced_union = torch.from_numpy(union).to(cfg['device'])
            reduced_target = torch.from_numpy(target).to(cfg['device'])

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)
    return mIOU, iou_class