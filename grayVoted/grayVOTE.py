from __future__ import print_function
from multiprocessing.pool import ThreadPool
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import PreTreatment
import operator 



def grayVoted(Img,sizeM,k):
    w,h=Img.shape# 行、列。旋转90度后的宽高
    step=1
    sizeM=sizeM# 奇数比较好，这样可以有整数中心点
    numMM=sizeM*sizeM
    r=sizeM//2
    listCenterIndex=sizeM*sizeM//2
    startY=r
    endY=h-r
    startX=r
    endX=w-r
    comparation=0
    newImage=Img.copy()
    for x in range(startX,endX,step):
        for y in range(startY, endY, step):
            Num1=0
            Num2=0
            comparation=Img[x][y]-k
            smallROI=Img[x-r:x+r+1,y-r:y+r+1] #array
            # 计算领域中的Max和Min
            Roi=smallROI.reshape(1,numMM)
            Roi2List=Roi.tolist()[0]  # 元组转换为列表后，是嵌套列表，通过索引将嵌套列表变成一个单纯的列表
            #print('Roi2List',Roi2List)
            Roi2List.pop(listCenterIndex)
            neighbor=Roi2List
            Num1=len([i for i in neighbor if i>=comparation])
            Num2=numMM-1-Num1
            min_number = np.min(neighbor)
            max_number = np.max(neighbor)
            # max.append(max_number)
            # min.append(min_number)
            # neighbors.append(neighbor)
            # ROI.append(smallROI)
            L=max_number/(numMM-1)
            N=-min_number/(numMM-1)
            # 灰度投票
            newImage[x][y]=Num1*L+Num2*N
    return newImage.astype(np.uint8)
            



if __name__ == '__main__':

    # 原图
    Img = cv2.imread("D:/ddddata/DRIVE/training/images/"+"23"+"_training.tif") #注意这儿是原图
    greenImg = cv2.split(Img)[1] # 绿色通道
    grayImg = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    #grayImg=grayImg[50:400,50:400]

      # 提取掩膜
    ret0, th0 = cv2.threshold(grayImg, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(th0, np.ones((7, 7), np.uint8))


# 对比度=======================================================
    # 高斯滤波
    blurImg = cv2.GaussianBlur(grayImg, (5, 5), 0)
    # CLAHE 光均衡化+对比度增强
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 10))
    claheImg = clahe.apply(blurImg)
    ImageV1=grayVoted(greenImg,sizeM=11,k=-2)
    #ImageV2=grayVoted(ImageV1,sizeM=11,k=-5)
    ret1, otsu1= cv2.threshold(ImageV1, 0, 255, cv2.THRESH_OTSU)
    #ret1, otsu2= cv2.threshold(ImageV2, 0, 255, cv2.THRESH_OTSU)
    #compare1=np.hstack([ImageV1,ImageV2])
    #compare2=np.hstack([otsu1,otsu2])
    #cv2.imshow('second voted',compare1)
    #cv2.imshow('otsu second voted ',compare2)
    
    otsu1INV=cv2.bitwise_not(otsu1)
    otsu1INV = PreTreatment.pass_mask(mask, otsu1INV)
    #cv2.imwrite('FirstVoted.jpg',otsu1INV)
    PreTreatment.showImg('FirstVoted.jpg',otsu1INV)




    cv2.waitKey(0)
    
