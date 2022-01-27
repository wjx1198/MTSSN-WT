###########################################################################
# Created by: Yuan Hu
# Email: huyuan@radi.ac.cn
# Copyright (c) 2019
###########################################################################

import os
import pdb

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
from torch.utils import data
from tqdm import tqdm

import option
from WTCNN_DATA import EdgeDetection
from models.dff_side5_att import get_dff
from models.fusion_side5_att import Fusion
from utils import CFScore
from utils import helpers
from utils import utils
from utils.log import create_logger

torch.cuda.set_device(2)
computer_device = 2


# torch_ver = torch.__version__[:3]
# if torch_ver == '0.3':
#     from torch.autograd import Variable

class Trainer():
    def __init__(self, args):
        self.args = args
        args.log_name = str(args.checkname)
        self.logger = create_logger(args.log_root, args.log_name)
        self.logger.info(args)

        # # data transforms
        # input_transform = transform.Compose([
        #     transform.ToTensor(),
        #     transform.Normalize([.485, .456, .406], [.229, .224, .225])]) # ImageNet训练集的mean及std
        #
        # # dataset
        # data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
        #                'crop_size': args.crop_size, 'logger': self.logger,
        #                'scale': args.scale}

        trainset = EdgeDetection(split='train')
        # testset = EdgeDetection(split='test')
        valtest = EdgeDetection(split='val')

        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} \
            if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=False, shuffle=True, **kwargs)
        # self.testloader = data.DataLoader(testset, batch_size=args.batch_size,
        #                                  drop_last=False, shuffle=False, **kwargs)
        self.valloader = data.DataLoader(valtest, batch_size=args.batch_size,
                                         drop_last=False, shuffle=True, **kwargs)
        self.nclass = trainset.num_class
        # num_vals = min(args.num_val_images, len(self.valloader)*args.batch_size)
        #
        # random.seed(16)
        # self.val_indices = random.sample(range(0, len(self.valloader)*args.batch_size), num_vals)

        # model
        wvlt_name = 'db1'
        model = Fusion(
            nclass=self.nclass,
            backbone=args.backbone,
            norm_layer=BatchNorm2d,
            wvlt_transform=wvlt_name,
            computer_device=computer_device
        )

        # elif args.model_edge == 'casenet':
        #     model_edge = get_casenet(
        #         nclass=self.nclass,
        #         backbone=args.backbone,
        #         norm_layer=BatchNorm2d
        #     )
        self.logger.info(model)

        # for param in list(model.pretrained.parameters()):
        #     param.requires_grad = False

        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},
                       # {'params': model.dff.ada_learner.parameters(), 'lr': args.lr * 10},
                       {'params': model.dff.side1.parameters(), 'lr': args.lr * 10},
                       {'params': model.dff.side2.parameters(), 'lr': args.lr * 10},
                       {'params': model.dff.side3.parameters(), 'lr': args.lr * 10},
                       {'params': model.dff.side5.parameters(), 'lr': args.lr * 10},
                       # {'params': model.dff.side5_w.parameters(), 'lr': args.lr * 10},
                       {'params': model.wavelet.parameters(), 'lr':args.lr * 10}]
        # params_list = [{'params': model.dff.ada_learner.parameters(), 'lr': args.lr * 10},
        #                {'params': model.dff.side1.parameters(), 'lr': args.lr * 10},
        #                {'params': model.dff.side2.parameters(), 'lr': args.lr * 10},
        #                {'params': model.dff.side3.parameters(), 'lr': args.lr * 10},
        #                {'params': model.dff.side5.parameters(), 'lr': args.lr * 10},
        #                {'params': model.dff.side5_w.parameters(), 'lr': args.lr * 10},
        #                {'params': model.wavelet.parameters(), 'lr': args.lr * 10}]

        optimizer = torch.optim.SGD(params_list,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        # finetune from a trained model
        # pdb.set_trace()
        if args.ft:
            model_cityscape = get_dff(
                nclass=19,
                backbone=args.backbone,
                norm_layer=BatchNorm2d
            )

            self.model_cityscape = model_cityscape
            args.start_epoch = 0
            checkpoint = torch.load('dff_cityscapes_resnet101.pth.tar')
            self.model_cityscape.load_state_dict(checkpoint['state_dict'], strict=False)
            model = model.to('cpu')
            for k1, v1 in model.pretrained.named_parameters():
                for k2, v2 in model_cityscape.pretrained.named_parameters():
                    if k1 == k2:
                        v1.data = v2.data
            # if args.cuda:
            #     self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            # else:
            #     self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            # model_dict = model.state_dict()
            # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict('pretrained')}
            # model_dict('pretrained').update(pretrained_dict)
            # model.load_state_dict(model_dict)
            # if args.cuda:
            #     model = model.cuda()
            self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.ft_resume, checkpoint['epoch']))

        # Instantiate the gradient descent optimizer - use Adam optimizaer with default parameters
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
        # self.criterion1 = utils.SoftIoULoss(self.nclass)
        # self.criterion_edge = nn.CrossEntropyLoss(reduction='none')
        # self.model_edge, self.optimizer_edge, self.model, self.optimizer = model_edge, optimizer_edge, model, optimizer

        # resuming checkpoint
        if args.resume:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            model = torch.load(args.resume)
            args.start_epoch = 85
            # if args.cuda:
            #     self.model_edge.module.load_state_dict(checkpoint['state_dict'])
            # else:
            #     self.model_edge.load_state_dict(checkpoint['state_dict'])
            # if not args.ft:
            #     self.optimizer_edge.load_state_dict(checkpoint['optimizer'])
            self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, args.start_epoch))

        # lr scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.model = model
        self.optimizer = optimizer
        # using cuda
        if args.cuda:
            self.model = model.cuda()
            self.criterion = self.criterion.cuda()
        # self.scheduler_edge = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.trainloader),
        #                               logger=self.logger, lr_step=args.lr_step)

    def training(self, epoch):
        self.model.train()
        class_names_list, label_values = helpers.get_label_info(os.path.join("./road_dataset_edge_new/class_dict.csv"))
        class_names_string = ""
        for class_name in class_names_list:
            if not class_name == class_names_list[-1]:
                class_names_string = class_names_string + class_name + ", "
            else:
                class_names_string = class_names_string + class_name

        # The softmax cross entropy loss with the weighting mechanism
        if os.path.exists('01.txt'):
            class_edge_weights = np.loadtxt('01.txt')
        else:
            print("Computing class edge weights for", args.dataset, "...")
            class_edge_weights = utils.compute_class_weights(labels_dir='./road_dataset_edge_new/train_sedge_labels',
                                                        label_values=label_values)
            np.savetxt('01.txt', class_edge_weights)
            print(class_edge_weights)

        if os.path.exists('02.txt'):
            class_weights = np.loadtxt('02.txt')
        else:
            print("Computing class weights for", args.dataset, "...")
            class_weights = utils.compute_class_weights(labels_dir='./road_dataset_edge_new/train_labels',
                                                        label_values=label_values)
            np.savetxt('02.txt', class_weights)
            print(class_weights)

        # Define the loss criterion and instantiate the gradient descent optimizer
        class_edge_weights = torch.from_numpy(class_edge_weights)
        class_edge_weights = torch.tensor(class_edge_weights, dtype=torch.float32)
        class_weights = torch.from_numpy(class_weights)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        tbar = tqdm(self.trainloader)
        train_loss = 0.
        train_loss_all = 0.
        for i, (image, target, target_edge, _) in enumerate(tbar):
            target_new = np.float32(
                helpers.one_hot_it(label=np.transpose(target.numpy(), [0, 2, 3, 1]), label_values=label_values))
            target_new = torch.from_numpy(target_new)
            target_new = np.transpose(target_new, [0, 3, 1, 2])
            target = np.float32(
                helpers.one_hot_it(label=np.transpose(target.numpy(), [0, 2, 3, 1]), label_values=label_values))
            # target = np.transpose(target, (0, 2, 3, 1))
            target = helpers.reverse_one_hot(target)
            target = torch.from_numpy(target)

            target_edge_new = np.float32(
                helpers.one_hot_it(label=np.transpose(target_edge.numpy(), [0, 2, 3, 1]), label_values=label_values))
            target_edge_new = torch.from_numpy(target_edge_new)
            target_edge_new = np.transpose(target_edge_new, [0, 3, 1, 2])
            target_edge = np.float32(
                helpers.one_hot_it(label=np.transpose(target_edge.numpy(), [0, 2, 3, 1]), label_values=label_values))
            # target_edge = np.transpose(target_edge, (0, 2, 3, 1))
            target_edge = helpers.reverse_one_hot(target_edge)
            target_edge = torch.from_numpy(target_edge)

            if args.cuda:
                image = image.cuda()
                target = target.cuda()
                target_edge = target_edge.cuda()
                # target_new = target_new.cuda()
                # target_edge_new = target_edge_new.cuda()

            self.optimizer.zero_grad()

            output, outputs_edge = self.model(image.float())
            output_edge, side5 = tuple(outputs_edge)
            # pdb.set_trace()

            loss1 = self.criterion(output_edge, target_edge.long())
            loss1 = utils.cp_loss(class_edge_weights, target_edge_new, loss1, args.batch_size)
            loss2 = self.criterion(side5, target_edge.long())
            loss2 = utils.cp_loss(class_edge_weights, target_edge_new, loss2, args.batch_size)
            loss_tensor = self.criterion(output, target.long()).cpu()
            loss_tensor = utils.cp_loss(class_weights, target_new, loss_tensor, args.batch_size)

            # loss1_iou = self.criterion1(output, target_new)
            # loss2_iou = self.criterion1(side5, target_new)
            # loss = (2 + (loss1_iou + loss2_iou)) * 100 + loss1 + loss2
            # loss1_dice = self.criterion(output, target_new)
            # loss2_dice = self.criterion(side5, target_new)
            # loss = loss1_dice + loss2_dice
            loss = loss1 + loss2 + loss_tensor
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_loss_all += loss.item()
            if i == 0 or (i + 1) % 20 == 0:
                train_loss = train_loss / min(20, i + 1)

                self.logger.info('Epoch [%d], Batch [%d],\t train-loss: %.4f' % (
                    epoch + 1, i + 1, train_loss))
                train_loss = 0.

        self.scheduler.step()
        # avg_loss_per_epoch.append(train_loss_all / (i + 1))
        self.logger.info('-> Epoch [%d], Train epoch loss: %.3f' % (
            epoch + 1, train_loss_all / (i + 1)))

        if not args.no_val:
            # save checkpoint every 20 epoch
            filename = "checkpoint_%s.pth.tar" % (epoch + 1)
            if epoch % 3 == 0 or epoch == args.epochs - 1:
                directory = "./runs_side5_attention/%s/%s/%s/" % (args.dataset, args.model, args.checkname)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + filename
                torch.save(self.model, filename)
        return train_loss_all / (i + 1)

    def validation(self, epoch):
        self.model.eval()
        class_names_list, label_values = helpers.get_label_info(os.path.join("./road_dataset_edge_new/class_dict.csv"))
        tbar_val = tqdm(self.valloader)
        # tbar = self.valloader

        avg_scores_per_epoch = []
        avg_iou_per_epoch = []
        class_scores_list = []
        iou_list = []
        n = 0
        truePos_currImg_list = [0.0] * args.num_val_images
        falsePos_currImg_list = [0.0] * args.num_val_images
        falseNeg_currImg_list = [0.0] * args.num_val_images
        avg_scores_edge_per_epoch = []
        avg_iou_edge_per_epoch = []
        class_scores_edge_list = []
        iou_edge_list = []
        truePos_currImg_edge_list = [0.0] * args.num_val_images
        falsePos_currImg_edge_list = [0.0] * args.num_val_images
        falseNeg_currImg_edge_list = [0.0] * args.num_val_images

        for i, (image, target, target_edge, paths) in enumerate(tbar_val):
            # if torch_ver == "0.3":
            #     image = Variable(image, volatile=True)
            # else:
            # pdb.set_trace()
            if args.cuda:
                image = image.cuda()
            with torch.no_grad():
                outputs, outputs_edge = self.model(image.float())
            output, side5 = tuple(outputs_edge)
            num = min(args.test_batch_size, len(paths))
            for j in range(num):
                # pdb.set_trace()
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
                                                                                      num_classes=self.nclass,
                                                                                      resize_height=512,
                                                                                      resize_width=512)
                # 边缘语义分割验证结果
                class_accuracies_edge, iou_edge, tpos_edge, fpos_edge, fneg_edge = utils.evaluate_segmentation(
                    pred=side5_image, label=gt_edge,
                    num_classes=self.nclass,
                    resize_height=512,
                    resize_width=512)

                class_scores_list.append(class_accuracies)
                gt = helpers.colour_code_segmentation(gt, label_values)
                iou_list.append(iou)
                truePos_currImg_list[n] = list(tpos)
                falsePos_currImg_list[n] = list(fpos)
                falseNeg_currImg_list[n] = list(fneg)
                cv2.imwrite("%s/%s_pred.png" % ("Results_side5att", path),
                            cv2.cvtColor(np.uint8(outs_vis_image), cv2.COLOR_RGB2BGR))
                cv2.imwrite("%s/%s_gt.png" % ("Results_side5att", path), cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

                class_scores_edge_list.append(class_accuracies_edge)
                gt_edge = helpers.colour_code_segmentation(gt_edge, label_values)
                iou_edge_list.append(iou_edge)
                truePos_currImg_edge_list[n] = list(tpos_edge)
                falsePos_currImg_edge_list[n] = list(fpos_edge)
                falseNeg_currImg_edge_list[n] = list(fneg_edge)
                cv2.imwrite("%s/%s_edge_pred.png" % ("Results_edge_side5att", path),
                            cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
                cv2.imwrite("%s/%s_edge_side_pred.png" % ("Results_edge_side5att", path),
                            cv2.cvtColor(np.uint8(side5_vis_image), cv2.COLOR_RGB2BGR))
                cv2.imwrite("%s/%s_edge_gt.png" % ("Results_edge_side5att", path),
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
        print("Average accuracy for epoch # %04d = %f" % (epoch, np.mean(avg_pre[1:])))
        self.logger.info("Average accuracy for epoch # %04d = %f" % (epoch, np.mean(avg_pre[1:])))
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
        print("Average IoU for epoch # %04d = %f" % (epoch, np.mean(avg_iou[1:])))
        self.logger.info("Average IoU for epoch # %04d = %f" % (epoch, np.mean(avg_iou[1:])))
        avg_iou_per_epoch.append(np.mean(avg_iou[1:]))
        # 计算验证集的F_Score
        fscore_macro = CFScore.compute_F_Score(truePos_currImg_list, falsePos_currImg_list, falseNeg_currImg_list,
                                               self.nclass)
        self.logger.info(" Macro F1: %.7f" % fscore_macro)

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
        print("Average accuracy for epoch # %04d = %f" % (epoch, np.mean(avg_pre_edge[1:])))
        self.logger.info("Average accuracy for epoch # %04d = %f" % (epoch, np.mean(avg_pre_edge[1:])))
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
        print("Average IoU for epoch # %04d = %f" % (epoch, np.mean(avg_iou_edge[1:])))
        self.logger.info("Average IoU for epoch # %04d = %f" % (epoch, np.mean(avg_iou_edge[1:])))
        avg_iou_edge_per_epoch.append(np.mean(avg_iou_edge[1:]))
        # 计算验证集的F_Score
        fscore_macro_edge = CFScore.compute_F_Score(truePos_currImg_edge_list,
                                                    falsePos_currImg_edge_list,
                                                    falseNeg_currImg_edge_list,
                                                    self.nclass)
        self.logger.info(" Macro F1: %.7f" % fscore_macro_edge)


if __name__ == "__main__":
    a = option.Options()
    args = a.parse()
    torch.manual_seed(args.seed)

    trainer = Trainer(args)
    trainer.logger.info(['Starting Epoch:', str(args.start_epoch)])
    trainer.logger.info(['Total Epoches:', str(args.epochs)])

    avg_loss_per_epoch = []

    for epoch in range(args.start_epoch, args.epochs):
        # trainer.validation(epoch)
        a = trainer.training(epoch)
        avg_loss_per_epoch.append(a)
        # 绘制loss曲线并保存
        fig2, ax2 = plt.subplots(figsize=(11, 8))
        ax2.plot(range(epoch + 1), avg_loss_per_epoch)
        # ax2.plot(range(85, epoch + 1), avg_loss_per_epoch)
        ax2.set_title("Average loss vs epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Current loss")
        plt.savefig('loss_vs_epochs.png')
        if not args.no_val:
            trainer.validation(epoch)
