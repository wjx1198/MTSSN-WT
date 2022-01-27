###########################################################################
# Created by: Yuan Hu
# Email: huyuan@radi.ac.cn
# Copyright (c) 2019
###########################################################################

import os
import time

import numpy as np
from tqdm import tqdm

import torch
from utils import utils
from utils import helpers
from utils import CFScore
from torch.utils import data
from utils.log import create_logger
from torch.nn import BatchNorm2d
from WTCNN_DATA import EdgeDetection
from models.fusion_side5_att import Fusion
import cv2
from option import Options
import pdb

torch.cuda.set_device(2)
computer_device = 2


def test(args):
    args.log_name = str(args.checkname)
    logger = create_logger(args.log_root, args.log_name)
    logger.info(args)
    # dataset
    testset = EdgeDetection(split='test')
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False, **loader_kwargs)

    # model
    wvlt_name = 'db1'
    model = Fusion(
        nclass=6,
        backbone=args.backbone,
        norm_layer=BatchNorm2d,
        wvlt_transform=wvlt_name,
        computer_device=computer_device
    )

    # resuming checkpoint
    if args.resume is None or not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    model = torch.load(args.resume)

    if args.cuda:
        model = model.cuda()
    print(model)

    model.eval()
    class_names_list, label_values = helpers.get_label_info(os.path.join("./road_dataset_edge_new/class_dict.csv"))
    tbar_val = tqdm(test_data)
    # tbar = self.valloader

    avg_scores_per_epoch = []
    avg_iou_per_epoch = []
    class_scores_list = []
    iou_list = []
    n = 0
    truePos_currImg_list = [0.0] * 698
    falsePos_currImg_list = [0.0] * 698
    falseNeg_currImg_list = [0.0] * 698
    avg_scores_edge_per_epoch = []
    avg_iou_edge_per_epoch = []
    class_scores_edge_list = []
    iou_edge_list = []
    truePos_currImg_edge_list = [0.0] * 698
    falsePos_currImg_edge_list = [0.0] * 698
    falseNeg_currImg_edge_list = [0.0] * 698
    for i, (image, target, target_edge, paths) in enumerate(tbar_val):
        if args.cuda:
            image = image.cuda()
            with torch.no_grad():
                outputs, outputs_edge = model(image.float())
            output, side5 = tuple(outputs_edge)
            num = min(args.test_batch_size, len(paths))
            for j in range(num):
                path = paths[j]
                gt = target[j, :, :, :]
                gt = helpers.reverse_one_hot(helpers.one_hot_it(np.transpose(gt.numpy(), [1, 2, 0]), label_values))
                gt_edge = target_edge[j, :, :, :]
                gt_edge = helpers.reverse_one_hot(
                    helpers.one_hot_it(np.transpose(gt_edge.numpy(), [1, 2, 0]), label_values))
                # 语义分割结果验证集
                outputs_image = outputs[j, :, :, :].cuda().data.cpu().numpy()
                outputs_image = np.transpose(outputs_image, (1, 2, 0))
                outputs_image = helpers.reverse_one_hot(outputs_image)
                outs_vis_image = helpers.colour_code_segmentation(outputs_image, label_values)
                # 边缘语义分割结果验证集
                output_image = output[j, :, :, :].cuda().data.cpu().numpy()
                output_image = np.transpose(output_image, (1, 2, 0))
                output_image = helpers.reverse_one_hot(output_image)
                out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

                side5_image = side5[j, :, :, :].cuda().data.cpu().numpy()
                side5_image = np.transpose(side5_image, (1, 2, 0))
                side5_image = helpers.reverse_one_hot(side5_image)
                side5_vis_image = helpers.colour_code_segmentation(side5_image, label_values)
                # 语义分割验证结果
                class_accuracies, iou, tpos, fpos, fneg = utils.evaluate_segmentation(pred=outputs_image, label=gt,
                                                                                      num_classes=6,
                                                                                      resize_height=512,
                                                                                      resize_width=512)
                # 边缘语义分割验证结果
                class_accuracies_edge, iou_edge, tpos_edge, fpos_edge, fneg_edge = utils.evaluate_segmentation(
                    pred=output_image, label=gt_edge,
                    num_classes=6,
                    resize_height=512,
                    resize_width=512)

                class_scores_list.append(class_accuracies)
                gt = helpers.colour_code_segmentation(gt, label_values)
                iou_list.append(iou)
                truePos_currImg_list[n] = list(tpos)
                falsePos_currImg_list[n] = list(fpos)
                falseNeg_currImg_list[n] = list(fneg)
                cv2.imwrite("%s/%s_pred.png" % ("test_Results", path),
                            cv2.cvtColor(np.uint8(outs_vis_image), cv2.COLOR_RGB2BGR))
                cv2.imwrite("%s/%s_gt.png" % ("test_Results", path), cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

                class_scores_edge_list.append(class_accuracies_edge)
                gt_edge = helpers.colour_code_segmentation(gt_edge, label_values)
                iou_edge_list.append(iou_edge)
                truePos_currImg_edge_list[n] = list(tpos_edge)
                falsePos_currImg_edge_list[n] = list(fpos_edge)
                falseNeg_currImg_edge_list[n] = list(fneg_edge)
                cv2.imwrite("%s/%s_edge_pred.png" % ("test_Results_edge", path),
                            cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
                cv2.imwrite("%s/%s_edge_side_pred.png" % ("test_Results_edge", path),
                            cv2.cvtColor(np.uint8(side5_vis_image), cv2.COLOR_RGB2BGR))
                cv2.imwrite("%s/%s_edge_gt.png" % ("test_Results_edge", path),
                            cv2.cvtColor(np.uint8(gt_edge), cv2.COLOR_RGB2BGR))
                n = n + 1

    # 计算语义分割定量结果
    # 计算精度
    class_scores_list_new = list(map(list, zip(*class_scores_list)))  # python3的map函数输出为map型，故需转list，python2不用
    avg_pre = []
    new = class_scores_list_new
    for i in range(len(class_scores_list_new)):
        # Remove noises in GT
        new[i] = np.delete(new[i], np.where(new[i] == 8.8))
        new[i] = np.delete(new[i], np.where(new[i] == 8.8))
        if class_scores_list_new[i].size == 0:
            continue
        else:
            avg_pre.append(np.mean(new[i]))
            print("%d %f" % (i, np.mean(new[i])))
    print("Average accuracy  = %f" % (np.mean(avg_pre[1:])))
    logger.info("Average accuracy = %f" % (np.mean(avg_pre[1:])))
    avg_scores_per_epoch.append(np.mean(avg_pre[1:]))
    # 计算IOU
    class_iou_list = list(map(list, zip(*iou_list)))
    avg_iou = []
    new_ = class_iou_list
    for i in range(len(class_iou_list)):
        new_[i] = np.delete(new_[i], np.where(new_[i] == 8.8))
        new_[i] = np.delete(new_[i], np.where(new_[i] == 8.8))
        if new_[i].size != 0:
            avg_iou.append(np.mean(new_[i]))
            print("%d %f" % (i, np.mean(new_[i])))
    print("Average IoU = %f" % (np.mean(avg_iou[1:])))
    logger.info("Average IoU = %f" % (np.mean(avg_iou[1:])))
    avg_iou_per_epoch.append(np.mean(avg_iou[1:]))
    # 计算验证集的F_Score
    fscore_macro = CFScore.compute_F_Score(truePos_currImg_list, falsePos_currImg_list, falseNeg_currImg_list,
                                           num_classes=6)
    logger.info(" Macro F1: %.7f" % fscore_macro)

    # 计算边缘语义分割定量结果
    # 计算精度
    class_scores_edge_list_new = list(
        map(list, zip(*class_scores_edge_list)))  # python3的map函数输出为map型，故需转list，python2不用
    avg_pre_edge = []
    new_edge = class_scores_edge_list_new
    for i in range(len(class_scores_edge_list_new)):
        # Remove noises in GT
        new_edge[i] = np.delete(new_edge[i], np.where(new_edge[i] == 8.8))
        new_edge[i] = np.delete(new_edge[i], np.where(new_edge[i] == 8.8))
        if class_scores_edge_list_new[i].size == 0:
            continue
        else:
            avg_pre_edge.append(np.mean(new_edge[i]))
            print("%d %f" % (i, np.mean(new_edge[i])))
    print("Average accuracy for edge  = %f" % (np.mean(avg_pre_edge[1:])))
    logger.info("Average accuracy for edge  = %f" % (np.mean(avg_pre_edge[1:])))
    avg_scores_edge_per_epoch.append(np.mean(avg_pre_edge[1:]))
    # 计算IOU
    class_iou_edge_list = list(map(list, zip(*iou_edge_list)))
    avg_iou_edge = []
    new_edge_ = class_iou_edge_list
    for i in range(len(class_iou_edge_list)):
        new_edge_[i] = np.delete(new_edge_[i], np.where(new_edge_[i] == 8.8))
        new_edge_[i] = np.delete(new_edge_[i], np.where(new_edge_[i] == 8.8))
        if new_edge_[i].size != 0:
            avg_iou_edge.append(np.mean(new_edge_[i]))
            print("%d %f" % (i, np.mean(new_edge_[i])))
    print("Average IoU for edge = %f" % (np.mean(avg_iou_edge[1:])))
    logger.info("Average IoU for edge = %f" % (np.mean(avg_iou_edge[1:])))
    avg_iou_edge_per_epoch.append(np.mean(avg_iou_edge[1:]))
    # 计算验证集的F_Score
    fscore_macro_edge = CFScore.compute_F_Score(truePos_currImg_edge_list,
                                                falsePos_currImg_edge_list,
                                                falseNeg_currImg_edge_list,
                                                num_classes=6)
    logger.info(" Macro F1: %.7f" % fscore_macro_edge)


def eval_model(args):
    if args.resume_dir is None:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume_dir))

    if os.path.splitext(args.resume_dir)[1] == '.tar':
        args.resume = args.resume_dir
        assert os.path.exists(args.resume_dir)
        test(args)


if __name__ == "__main__":
    args = Options().parse()
    args.test_batch_size = args.test_batch_size
    eval_model(args)
