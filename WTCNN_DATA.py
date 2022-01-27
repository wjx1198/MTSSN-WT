from __future__ import print_function, division

import datetime
import os
import sys

import cv2
import numpy as np
import utils.utils as U
import torch
import torch.utils.data as data
import pdb


class EdgeDetection(data.Dataset):
    NUM_CLASS = 6

    def __init__(self, split, root='./road_dataset_edge_new'):
        super(EdgeDetection, self).__init__()
        self.split = split
        self.images, self.masks, self.masks_edge = prepare_data(root, self.split)
        if split != 'vis':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, idx):
        paths = U.filepath_to_name(self.images[idx])
        img = U.load_image1(self.images[idx])
        mask = U.load_image2(self.masks[idx])
        mask_edge = U.load_image2(self.masks_edge[idx])
        img, _ = U.resize_data(img, img, 512, 512)
        mask, _ = U.resize_data(mask, mask, 512, 512)
        img = np.float32(img) / 255.0
        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        mask_edge = np.transpose(mask_edge, (2, 0, 1))
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        mask_edge = torch.from_numpy(mask_edge)
        return img, mask, mask_edge, paths

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        return self.NUM_CLASS


def prepare_data(dataset_dir, mode):
    train_input_names = []
    train_output_names = []
    train_edge_names = []
    val_input_names = []
    val_output_names = []
    val_edge_names = []
    test_input_names = []
    test_output_names = []
    test_edge_names = []
    if mode == 'train':
        for file in os.listdir(dataset_dir + "/train"):
            # cwd = os.getcwd()
            train_input_names.append(dataset_dir + "/train/" + file)
        for file in os.listdir(dataset_dir + "/train_labels"):
        #     # cwd = os.getcwd()
            train_output_names.append(dataset_dir + "/train_labels/" + file)
        for file in os.listdir(dataset_dir + "/train_sedge_labels"):
            # cwd = os.getcwd()
            train_edge_names.append(dataset_dir + "/train_sedge_labels/" + file)

        train_input_names.sort(), train_output_names.sort(), train_edge_names.sort()
        return train_input_names, train_output_names, train_edge_names

    if mode == 'val':
        for file in os.listdir(dataset_dir + "/val"):
            # cwd = os.getcwd()
            val_input_names.append(dataset_dir + "/val/" + file)
        for file in os.listdir(dataset_dir + "/val_labels"):
        #     cwd = os.getcwd()
            val_output_names.append(dataset_dir + "/val_labels/" + file)
        for file in os.listdir(dataset_dir + "/val_sedge_labels"):
            # cwd = os.getcwd()
            val_edge_names.append(dataset_dir + "/val_sedge_labels/" + file)

        val_input_names.sort(), val_output_names.sort(), val_edge_names.sort()
        return val_input_names, val_output_names, val_edge_names

    if mode == 'test':
        for file in os.listdir(dataset_dir + "/test"):
            # cwd = os.getcwd()
            test_input_names.append(dataset_dir + "/test/" + file)
        for file in os.listdir(dataset_dir + "/test_labels"):
        #     # cwd = os.getcwd()
            test_output_names.append(dataset_dir + "/test_labels/" + file)
        for file in os.listdir(dataset_dir + "/test_sedge_labels"):
            # cwd = os.getcwd()
            test_edge_names.append(dataset_dir + "/test_sedge_labels/" + file)

        test_input_names.sort(), test_output_names.sort(), test_edge_names.sort()
        return test_input_names, test_output_names, test_edge_names
    # train_input_names.sort(), train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort(), train_edge_names.sort(), val_edge_names.sort(), test_edge_names.sort(),
    # return train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names, train_edge_names, val_edge_names, test_edge_names


# # Input images & GT boundaries -- gray-scale images
# def load_image1(path):
#     image = cv2.cvtColor(cv2.imread(path, 0), cv2.COLOR_GRAY2RGB)
#     return image
#
#
# # GT -- color images
# def load_image2(path):
#     image = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
#     return image
#
#
# # GT -- color images
# def load_image3(path):
#     image = cv2.imread(path, 0)
#     return image
#
#
# def cp_loss(weights, labels, loss):
#     labels = torch.squeeze(labels)
#     weights_batch = torch.tensor(labels)
#     a = weights.shape
#     for i in range(a[0]):
#         weights_batch[i, :, :] = weights[i]
#     weights_new = weights_batch * labels
#     weights_new = (weights_new.sum(dim=0))
#     weighted_loss = torch.squeeze(loss) * weights_new
#     weighted_loss = torch.sum(weighted_loss)
#     return weighted_loss
#
#
# # Takes an absolute file path and returns the name of the file without th extension
# def filepath_to_name(full_name):
#     file_name = os.path.basename(full_name)
#     file_name = os.path.splitext(file_name)[0]
#     return file_name
#
#
# # Print with time
# def LOG(X, f=None):
#     time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
#     if not f:
#         print(time_stamp + " " + X)
#     else:
#         f.write(time_stamp + " " + X)
#
#
# # Resize the image
# def resize_data(image, label, resize_height, resize_width):
#     image = cv2.resize(image, (resize_height, resize_width))
#     label = cv2.resize(label, (resize_height, resize_width))
#     return image, label
#
#
# def compute_cpa(pred, label, num_classes, resize_height, resize_width):
#     pred = np.reshape(pred, (resize_height, resize_width))
#     label = np.reshape(label, (resize_height, resize_width))
#     total = [0.0] * num_classes
#     for val in range(num_classes):
#         total[val] = (label == val).sum()
#
#     count = [0.0] * num_classes
#     for i in range(num_classes):
#         idx, idy = np.where(pred == i)
#         for j in range(len(idx)):
#             xx = idx[j]
#             yy = idy[j]
#             if pred[xx, yy] == label[xx, yy]:
#                 count[i] = count[i] + 1.0
#
#     accuracies = []
#     for i in range(len(total)):
#         # Remove noises in GT
#         if total[i] <= 20.1:
#             accuracies.append(8.8)
#         else:
#             accuracies.append(count[i] / total[i])
#     return accuracies
#
#
# def compute_iou(pred, label, num_classes, resize_height, resize_width):
#     pred = np.reshape(pred, (resize_height, resize_width))
#     label = np.reshape(label, (resize_height, resize_width))
#     total1 = [0.0] * num_classes
#     total2 = [0.0] * num_classes
#     for val in range(num_classes):
#         total1[val] = (label == val).sum()
#         total2[val] = (pred == val).sum()
#     count = [0.0] * num_classes
#     for i in range(num_classes):
#         idx, idy = np.where(pred == i)
#         for j in range(len(idx)):
#             xx = idx[j]
#             yy = idy[j]
#             if pred[xx, yy] == label[xx, yy]:
#                 count[i] = count[i] + 1.0
#
#     IoU = []
#     total = np.subtract(np.add(total1, total2), count)
#     for i in range(len(total)):
#         if total1[i] <= 20.1:
#             IoU.append(8.8)
#         else:
#             IoU.append(count[i] / total[i])
#     return IoU
#
#
# # def evaluate_segmentation(pred, label, num_classes, resize_height, resize_width):
# #     flat_pred = pred.flatten()
# #     flat_label = label.flatten()
# #
# #     cpa = compute_cpa(flat_pred, flat_label, num_classes, resize_height, resize_width)
# #     iou = compute_iou(flat_pred, flat_label, num_classes, resize_height, resize_width)
# #     tpos, fpos, fneg = CFScore.compute_F_Score_e(flat_pred, flat_label, num_classes, resize_height, resize_width)
# #
# #     return cpa, iou, tpos, fpos, fneg
#
#
# def compute_class_weights(labels_dir, label_values):
#     '''
#     Arguments:
#         labels_dir(list): Directory where the image segmentation labels are
#         num_classes(int): the number of classes of pixels in all images
#
#     Returns:
#         class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.
#
#     '''
#     image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]
#
#     num_classes = len(label_values)
#
#     class_pixels = np.zeros(num_classes)
#
#     for n in range(len(image_files)):
#         image = imread(image_files[n])
#
#         for index, colour in enumerate(label_values):
#             class_map = np.all(np.equal(image, colour), axis=-1)
#             class_map = class_map.astype(np.float32)
#             class_pixels[index] += np.sum(class_map)
#
#         print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
#         sys.stdout.flush()
#
#     total_pixels = float(np.sum(class_pixels))
#     index_to_delete = np.argwhere(class_pixels == 0.0)
#     class_pixels = np.delete(class_pixels, index_to_delete)
#
#     class_weights = total_pixels / class_pixels
#     class_weights = class_weights / np.sum(class_weights)
#
#     return class_weights
#
#
# def mkdir(path):
#     isExists = os.path.exists(path)  # 判断路径是否存在，若存在则返回True，若不存在则返回False
#     if not isExists:  # 如果不存在则创建目录
#         os.makedirs(path)
#         return True
#     else:
#         return False
#
#
# def save_feature_to_img(feature_images, name):
#     features = feature_images  # 返回一个指定层输出的特征图,属于四维张量[batch,channel,width,height]
#     for i in range(features.shape[1]):
#         feature = features[:, i, :, :]  # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
#         feature = feature.view(feature.shape[1], feature.shape[2])  # batch为1，所以可以直接view成二维张量
#         feature = feature.cuda().data.cpu().numpy()  # 转为numpy
#
#         # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
#         feature = (feature - np.amin(feature)) / (np.amax(feature) - np.amin(feature) + 1e-5)  # 注意要防止分母为0！
#         feature = np.round(feature * 255)  # [0, 1]——[0, 255],为cv2.imwrite()函数而进行
#
#         mkdir('/home/wujunxian/data/WTCNN-Net/wavwlet_cnn_master/featuremap/' + name)  # 创建保存文件夹，以选定可视化层的序号命名
#         cv2.imwrite('/home/wujunxian/data/WTCNN-Net/wavwlet_cnn_master/featuremap/' + name + '/' + str(i) + '.jpg',
#                     feature)  # 保存当前层输出的每个channel上的特征图为一张图像
