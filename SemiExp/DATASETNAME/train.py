import os
import sys
import argparse
import pprint
import warnings
from datetime import datetime
import logging
import numpy as np
from matplotlib import pyplot as plt
from torch.optim import AdamW, Adam, SGD,lr_scheduler
from torch import optim
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from Lib.Criterion.CriterionConfig import CriterionSet
from Lib.GetDatasetsFunc.SemiSupervised.SELFNET.selfnet import SemiDataset
from Lib.SegModels.SemiSupervised.models_Interface import SSegmentationSet
from Lib.Tools.PublicFunc.classes import CLASSES
from Lib.Tools.PublicFunc.evaluation import evaluate
from Lib.Tools.PublicFunc.utils import AverageMeter, init_log
from Lib.Tools.SELFNET.augmethods import hybrid_cutmix_aug
from Lib.Tools.SELFNET.em_threshold import get_class_adaptive_threshold
from Lib.Tools.SELFNET.entropy_map import calc_entropy_map
warnings.filterwarnings('ignore')


def parse_args():
    # 可更改参数
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--model', default='selfnet', type=str)


    "改标记率实现数据集的标记切换----该数据集名称实现数据集切换"
    parser.add_argument('--save_path', default='SaveModels/logging/DFC22/semi_supervised/selfnet/1_8/fix50_proto_classem60',help='path where to tensorboard log')
    parser.add_argument('--dataset_name', default='DFC22', type=str, help='lower alphabet is must')
    parser.add_argument('--dataset_root', default='RsDatasets/DFC22', type=str, help='数据集路径')
    parser.add_argument('--num_classes', type=int, default=12)
    parser.add_argument('--labeled_id_path', default='RsDatasets/splits/DFC22/1_8/labeled.txt', type=str,help = 'labeled file path')
    parser.add_argument('--unlabeled_id_path', default='RsDatasets/splits/DFC22/1_8/unlabeled.txt', type=str, help='unlabeled file path')
    parser.add_argument('--label_ratio', default=0.125, type=float, help='unlabeled file path')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--percent', default=60, type=int)
    parser.add_argument('--fixed_conf_threshold', default=0.50)
    parser.add_argument('--dynamic_conf_threshold', default=0.01)

    parser.add_argument('--backbone', default='resnet101', type=str)
    parser.add_argument('--crop_size', default=320, type=int,help='数据集大小')
    parser.add_argument('--ignore_index',type=int,default=255,help='path where to tensorboard log')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (absolute lr)')
    parser.add_argument('--lr_multi', type=float, default=10.0)
    parser.add_argument('--evaluate_mode', type=str, default='original')

    # 基本不更改参数
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--BENCHMARK', type=bool, default=False)
    parser.add_argument('--DETERMINISTIC', type=bool, default=False)
    parser.add_argument('--ENABLED', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=512)
    args = parser.parse_args()
    return args

def main():
    # 统一设置
    args = parse_args()
    """文件存储设置"""
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.save_path,formatted_time+'.log')
    """关闭日志传播->防止输出重复日志"""
    logger = init_log('global', logging.INFO,log_file)
    logger.propagate = 0
    all_args = {**vars(args), 'device': args.device}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(args.save_path)
    if args.seed > 0:
        import random
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    cudnn.benchmark = args.BENCHMARK
    cudnn.deterministic = args.DETERMINISTIC
    cudnn.enabled = args.ENABLED

    device = torch.device(args.device)

    """读取数据形成dataloader"""
    trainset_u = SemiDataset(args.dataset_name,args.dataset_root , 'train_u',args.crop_size, args.unlabeled_id_path)
    trainset_l = SemiDataset(args.dataset_name, args.dataset_root, 'train_l',args.crop_size, args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(args.dataset_name,args.dataset_root , 'val')
    trainloader_l = DataLoader(trainset_l, batch_size=args.batch_size,shuffle=True,pin_memory=False, num_workers=1, drop_last=True)
    trainloader_u = DataLoader(trainset_u, batch_size=args.batch_size,shuffle=True,pin_memory=False, num_workers=1, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, pin_memory=False,shuffle=False, num_workers=1,drop_last=False)

    model = SSegmentationSet(model=args.model,cfg={'backbone':args.backbone,'dilations':[6, 12, 18],'num_classes':args.num_classes,'replace_stride_with_dilation':[False, False, True]}).to(device)

    """优化器定义"""
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},{'params': [param for name, param in model.named_parameters() if 'backbone' not in name],'lr': args.lr * args.lr_multi}], lr=args.lr, momentum=0.9, weight_decay=1e-4)
    """损失函数定义"""
    criterion_l = CriterionSet(loss='cross_entropy_ignore')
    criterion_u = CriterionSet(loss='cross_entropy_ignore')
    criterion_proto = CriterionSet(loss='proto_learning_loss')
    """固定参数"""
    total_iters = len(trainloader_u) * args.epochs
    previous_best = 0.0
    epoch = -1
    """检查点定义"""
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epoch + 1, args.epochs):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(epoch, optimizer.param_groups[0]['lr'], previous_best))
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_proto = AverageMeter()
        total_loss_proto_seg = AverageMeter()
        loader = zip(trainloader_l, trainloader_u)
        for i, ((img_x, mask_x), (img_u_w, img_u_s1, img_u_s2)) in enumerate(loader):
            img_x, mask_x = img_x.to(device), mask_x.to(device)
            img_u_w = img_u_w.to(device)
            img_u_s1, img_u_s2 = img_u_s1.to(device), img_u_s2.to(device)
            with torch.no_grad():
                model.eval()
                pred_u_w = model(img_u_w)['out'].detach()
                prob_u_w = pred_u_w.softmax(dim=1)
                conf_u_w, mask_u_w = prob_u_w.max(dim=1)
            model.train()
            if np.random.uniform(0, 1) < 0.5:
                conf_u_w, mask_u_w, img_u_s1 = hybrid_cutmix_aug(
                                                conf_w=conf_u_w,
                                                mask_w=mask_u_w,
                                                data_s1=img_u_s1,
                                                data_s2=img_u_s2,
                                                epoch=epoch,
                                                total_epochs=args.epochs,
                                                label_ratio=args.label_ratio,
                                                dynamic_conf_threshold = args.dynamic_conf_threshold,
                                                fixed_conf_threshold=args.fixed_conf_threshold,
                                                )
                num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
                pred = model(torch.cat((img_x, img_u_s1)),mode='train',need_prototype = True,gt_seg = mask_x)
                pred_l, pred_u_s1 = pred['out'].split([num_lb, num_ulb])
                pred_u_s1 = pred_u_s1.clone()
                em_u_s1 = calc_entropy_map(pred_u_s1)
                # em_s1_threshold = np.percentile(em_u_s1.detach().cpu().numpy().flatten(), args.percent)
                # 动态阈值
                mask_reliable = get_class_adaptive_threshold(
                    em_u_s1=em_u_s1,
                    mask_u_w=mask_u_w,
                    num_classes=args.num_classes,
                    percent=args.percent
                )
                loss_x = criterion_l(pred_l, mask_x, args.ignore_index, device)
                loss_u_s1 = criterion_u(pred_u_s1, mask_u_w, args.ignore_index, device)
                # loss_u_s1 = loss_u_s1 * (em_u_s1 <= em_s1_threshold)
                loss_u_s1 = loss_u_s1 * mask_reliable
                loss_u_s1 = torch.mean(loss_u_s1)

                # 进行原型学习
                contrast_logits = pred['contrast_logits']
                contrast_target = pred['contrast_target']
                proto_seg_ul = pred['out_proto_seg_u']
                proto_seg_l = pred['out_proto_seg_l']
                loss_proto = criterion_proto(contrast_logits,contrast_target,args.ignore_index)

                loss_proto_seg_ul = criterion_u(proto_seg_ul, mask_u_w, args.ignore_index, device)
                loss_proto_seg_l = criterion_l(proto_seg_l,mask_x,args.ignore_index, device)
                loss_proto_seg = (loss_proto_seg_ul + loss_proto_seg_l)/2.0

            else:
                num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
                pred = model(torch.cat((img_x, img_u_s1)), mode='train',need_prototype = True,gt_seg = mask_x)
                pred_l, pred_u_s1 = pred['out'].split([num_lb, num_ulb])
                pred_u_s1 = pred_u_s1.clone()
                em_u_s1 = calc_entropy_map(pred_u_s1)
                # em_s1_threshold = np.percentile(em_u_s1.detach().cpu().numpy().flatten(), args.percent)
                # 动态阈值
                mask_reliable = get_class_adaptive_threshold(
                    em_u_s1=em_u_s1,
                    mask_u_w=mask_u_w,
                    num_classes=args.num_classes,
                    percent=args.percent
                )
                loss_x = criterion_l(pred_l, mask_x, args.ignore_index, device)
                loss_u_s1 = criterion_u(pred_u_s1, mask_u_w, args.ignore_index, device)
                # loss_u_s1 = loss_u_s1 * (em_u_s1 <= em_s1_threshold)
                loss_u_s1 = loss_u_s1 * mask_reliable
                loss_u_s1 = torch.mean(loss_u_s1)

                contrast_logits = pred['contrast_logits']
                contrast_target = pred['contrast_target']
                proto_seg_ul = pred['out_proto_seg_u']
                proto_seg_l = pred['out_proto_seg_l']
                loss_proto = criterion_proto(contrast_logits,contrast_target,args.ignore_index)
                
                loss_proto_seg_ul = criterion_u(proto_seg_ul, mask_u_w, args.ignore_index, device)
                loss_proto_seg_l = criterion_l(proto_seg_l,mask_x,args.ignore_index, device)
                loss_proto_seg = (loss_proto_seg_ul + loss_proto_seg_l)/2.0


            loss =loss_x + loss_proto + loss_u_s1  + loss_proto_seg


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1).item())
            total_loss_proto.update(loss_proto.item())
            total_loss_proto_seg.update(loss_proto_seg.item())
            """-----------进行优化器参数更新------------"""
            iters = epoch * len(trainloader_u) + i
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * args.lr_multi
            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_x', loss_x.item(), iters)
            writer.add_scalar('train/loss_s1',(loss_u_s1).item(), iters)
            writer.add_scalar('train/loss_proto', loss_proto.item(), iters)
            writer.add_scalar('train/loss_proto_seg', loss_proto_seg.item(), iters)
            if i % (len(trainloader_u) // 8) == 0:
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss proto: {:.3f}, Loss protoseg: {:.3f}'.format(i, total_loss.avg,total_loss_x.avg,total_loss_s.avg,total_loss_proto.avg,total_loss_proto_seg.avg))
        mIoU, iou_class = evaluate(model, valloader, 'original',{'crop_size': args.crop_size, 'num_classes': args.num_classes, 'device': device,'model_name': args.model})
        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] ''IoU: {:.2f}'.format(cls_idx,CLASSES[args.dataset_name][cls_idx], iou))
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(args.evaluate_mode, mIoU))
        writer.add_scalar('eval/mIoU', mIoU, epoch)
        for i, iou in enumerate(iou_class):
            writer.add_scalar('eval/%s_IoU' % (CLASSES[args.dataset_name][i]), iou, epoch)
        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))

if __name__ == '__main__':
    main()

























