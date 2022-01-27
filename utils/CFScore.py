import numpy as np
from utils.log import create_logger
import pdb

my_label_switcher_APD = {
    0: "Background",
    1: "Crack",
    2: "patch",
    3: "repair",
    4: "slab",
    5: "light",
    6: "Unknown Item 3",
    7: "Unknown Item 4"
}


def compute_F_Score_e(pred, label, num_classes, resize_height, resize_width, nTolerance=0):
    nNumOfDistess = num_classes
    # flat_pred = pred.flatten()
    # flat_label = label.flatten()
    pred = np.reshape(pred, (resize_height, resize_width))
    label = np.reshape(label, (resize_height, resize_width))
    total_gt = [0.0] * nNumOfDistess
    for val in range(nNumOfDistess):
        total_gt[val] = (label == val).sum()

    truePos_currImg = [0.0] * nNumOfDistess
    # trueNeg_currImg = [0] * nNumOfDistess  # 不影响计算PR值
    falsePos_currImg = [0.0] * nNumOfDistess
    falseNeg_currImg = [0.0] * nNumOfDistess

    nImage_w, nImage_h = pred.shape
    nLabel_w, nLabel_h = label.shape
    if nImage_w != nLabel_w or nImage_h != nLabel_h:
        print("\nImage Size Not Matched, Please Check Your File!")

    # -- Check PR-Curve for Curr image
    # https://www.zhihu.com/question/304639772
    # https://zhuanlan.zhihu.com/p/64315175 (used)
    matImageData_optr = pred
    matLabelData_optr = label
    for nRow in range(nImage_h):
        for nCol in range(nImage_w):
            nVal_NetOutput = matImageData_optr[nCol, nRow]
            nVal_NetLabel = matLabelData_optr[nCol, nRow]
            if nVal_NetOutput == nVal_NetLabel:
                truePos_currImg[nVal_NetOutput] += 1.0
                # truePos_allImgs[nVal_NetOutput] += 1.0
            else:
                bIsNeighCorrect = False
                for nRow_Offset in range(-nTolerance, nTolerance + 1):
                    for nCol_Offset in range(-nTolerance, nTolerance + 1):
                        nCol_Neigh = nCol + nCol_Offset
                        nRow_Neigh = nRow + nRow_Offset
                        if nCol_Neigh < 0 or nCol_Neigh > nImage_w - 1 or nRow_Neigh < 0 or nRow_Neigh > nImage_h - 1:
                            continue
                        nVal_Neigh = matLabelData_optr[nCol_Neigh, nRow_Neigh]
                        if nVal_NetOutput == nVal_Neigh:
                            bIsNeighCorrect = True
                        # if bIsNeighCorrect = True 可打破循环出去

                if bIsNeighCorrect:
                    # 若周围邻域内找到了该标签，那么它是truePos
                    truePos_currImg[nVal_NetOutput] += 1.0
                    # truePos_allImgs[nVal_NetOutput] += 1.0
                else:
                    # 若周围邻域内也未找到该标签，说明原始标签中该处像素没被检测出来，置为 falseNeg
                    falseNeg_currImg[nVal_NetLabel] += 1.0  # Recall相关
                    # falseNeg_allImgs[nVal_NetLabel] += 1.0  # Recall相关
                    # 若周围邻域内也未找到该标签，说明网络输出中该处像素检测错误，置为 falseNeg
                    falsePos_currImg[nVal_NetOutput] += 1.0  # Precision相关
                    # falsePos_allImgs[nVal_NetOutput] += 1.0  # Precision相关

    truePos_currImg_new = []
    falsePos_currImg_new = []
    falseNeg_currImg_new = []
    for i_ in range(len(total_gt)):
        if total_gt[i_] <= 20.1:
            truePos_currImg_new.append(8.8)
            falsePos_currImg_new.append(8.8)
            falseNeg_currImg_new.append(8.8)
        else:
            truePos_currImg_new.append(truePos_currImg[i_])
            falsePos_currImg_new.append(falsePos_currImg[i_])
            falseNeg_currImg_new.append(falseNeg_currImg[i_])
    return truePos_currImg_new, falsePos_currImg_new, falseNeg_currImg_new


def compute_F_Score(tpos_list, fpos_list, fneg_list, num_classes):

    print("\nCompute_F_Score(Percentage point):")
    print("\n")

    # -- Compute PR-Curve for All Images -------------------------------------------------------------------------------
    for i_prt in range(50):
        print("*", end='')

    nNumOfDistess = num_classes
    fscore_allImgs = [0.0] * nNumOfDistess
    precision_allImgs = [0.0] * nNumOfDistess
    recall_allImgs = [0.0] * nNumOfDistess
    truePos_allImgs = [0.0] * nNumOfDistess
    # trueNeg_allImgs = [0.0] * nNumOfDistess  # 不影响计算PR值
    falsePos_allImgs = [0.0] * nNumOfDistess
    falseNeg_allImgs = [0.0] * nNumOfDistess
    fSmallDivHelper = 1.e-9

    truePos_currImg_list = tpos_list
    falsePos_currImg_list = fpos_list
    falseNeg_currImg_list = fneg_list
    truePos_currImg_list_new = list(map(list, zip(*truePos_currImg_list)))
    # truePos_currImg_list_new = list(map(list, zip(*truePos_currImg_list)))
    falsePos_currImg_list_new = list(map(list, zip(*falsePos_currImg_list)))
    falseNeg_currImg_list_new = list(map(list, zip(*falseNeg_currImg_list)))
    new1 = truePos_currImg_list_new
    new2 = falsePos_currImg_list_new
    new3 = falseNeg_currImg_list_new
    for n1 in range(len(truePos_currImg_list_new)):
        # Remove noises in GT
        new1[n1] = np.delete(new1[n1], np.where(new1[n1] == 8.8))
        new1[n1] = np.delete(new1[n1], np.where(new1[n1] == 8.8))
    for n2 in range(len(truePos_currImg_list_new)):
        new2[n2] = np.delete(new2[n2], np.where(new2[n2] == 8.8))
        new2[n2] = np.delete(new2[n2], np.where(new2[n2] == 8.8))
    for n3 in range(len(truePos_currImg_list_new)):
        new3[n3] = np.delete(new3[n3], np.where(new3[n3] == 8.8))
        new3[n3] = np.delete(new3[n3], np.where(new3[n3] == 8.8))
    for n in range(len(truePos_currImg_list_new)):
        if new1[n].size != 0:
            truePos_allImgs[n] = float(np.sum(new1[n]))
            falsePos_allImgs[n] = float(np.sum(new2[n]))
            falseNeg_allImgs[n] = float(np.sum(new3[n]))
            recall_allImgs[n] = truePos_allImgs[n] / (
                    truePos_allImgs[n] + falseNeg_allImgs[n] + fSmallDivHelper)
            precision_allImgs[n] = truePos_allImgs[n] / (
                    truePos_allImgs[n] + falsePos_allImgs[n] + fSmallDivHelper)
            fscore_allImgs[n] = 2 * recall_allImgs[n] * precision_allImgs[n] / (
                    recall_allImgs[n] + precision_allImgs[n] + fSmallDivHelper)
            strDistName = my_label_switcher_APD.get(n, "Unknown Item")
            print("\nFile: All Images", " , Dist: ", strDistName, " , Precision: %.7f" % (precision_allImgs[n] * 100.0),
                  " , Recall: %.7f" % (recall_allImgs[n] * 100.0), " , F1: %.7f" % (fscore_allImgs[n] * 100.0))
        else:
            continue

    # -- Compute PR-Curve for Each Image
    #
    # Recall = True_Pos / (True_Pos + False_Neg)
    # Precision = True_Pos / (True_Pos + False_Pos)
    # F = 2 * Recall * Precision / (Recall + Precision)
    #
    # for i_class in range(nNumOfDistess):
    #     recall_currImg[i_class] = truePos_currImg[i_class] / (
    #                 truePos_currImg[i_class] + falseNeg_currImg[i_class] + fSmallDivHelper)
    #     precision_currImg[i_class] = truePos_currImg[i_class] / (
    #                 truePos_currImg[i_class] + falsePos_currImg[i_class] + fSmallDivHelper)
    #     fscore_currImg[i_class] = 2 * recall_currImg[i_class] * precision_currImg[i_class] / (
    #                 recall_currImg[i_class] + precision_currImg[i_class] + fSmallDivHelper)
    #     strDistName = my_label_switcher_APD.get(i_class, "Unknown Item")
    #     print("File: ", img_name, " , Dist: ", strDistName,
    #           " , Precision: %.2f" % (precision_currImg[i_class] * 100),
    #           " , Recall: %.2f" % (recall_currImg[i_class] * 100), " , F1: %.2f " % (fscore_currImg[i_class] * 100))

    # end of for i in range(0, len(listImages)):

    # -- Compute Mirco F1 ----------------------------------------------------------------------------------------------
    print("\n")
    for i in range(50):
        print("*", end='')
    print("\nMicro F1 Eval. Results (No Background)\n")

    truePos_allClasses = 0.0
    falseNeg_allClasses = 0.0
    falsePos_allClasses = 0.0
    fSmallDivHelper = 1.e-9
    for i_class in range(1, nNumOfDistess):  # No Background Counted
        truePos_allClasses += truePos_allImgs[i_class]
        falseNeg_allClasses += falseNeg_allImgs[i_class]
        falsePos_allClasses += falsePos_allImgs[i_class]
    recall_micro = truePos_allClasses / (truePos_allClasses + falseNeg_allClasses + fSmallDivHelper)
    precision_micro = truePos_allClasses / (truePos_allClasses + falsePos_allClasses + fSmallDivHelper)
    fscore_micro = 2 * recall_micro * precision_micro / (
            recall_micro + precision_micro + fSmallDivHelper)
    print("\nMicro Precision: %.7f" % (precision_micro * 100), " , Micro Recall: %.7f" % (recall_micro * 100),
          " , Micro F1: %.7f" % (fscore_micro * 100))

    # -- Compute Marco F1 ----------------------------------------------------------------------------------------------
    print("\nMacro F1 Eval. Results (No Background)")
    fscore_macro = 0
    for i_class in range(1, nNumOfDistess):  # No Background Counted
        if new1[i_class].size != 0:
            fscore_macro += fscore_allImgs[i_class]
            print("Class ", i_class, " F1: %.7f" % (fscore_allImgs[i_class] * 100))
        else:
            continue
    fscore_macro = fscore_macro / (nNumOfDistess-1)
    print(" Macro F1: %.7f" % (fscore_macro * 100))

    return fscore_macro * 100
    # return recall_allImgs, precision_allImgs, fscore_allImgs, fscore_micro, fscore_macro
