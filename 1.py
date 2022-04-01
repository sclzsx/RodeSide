import cv2
import numpy as np


def caleGrayHist(image):
    # 灰度图像的高、宽
    rows, cols = image.shape
    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint64)  # 图像的灰度级范围是0~255
    for r in range(rows):

        for c in range(cols):
            grayHist[image[r][c]] += 1

    return grayHist


def threshTwoPeaks(image):
    # 计算灰度直方图
    histogram = caleGrayHist(image)

    # 找到灰度直方图的最大峰值对应得灰度值
    maxLoc = np.where(histogram == np.max(histogram))
    firstPeak = maxLoc[0][0]  # 取第一个最大的灰度值

    # 寻找灰度直方图的第二个峰值对应得灰度值
    measureDists = np.zeros([256], np.float32)
    for k in range(256):
        measureDists[k] = pow(k - firstPeak, 2) * histogram[k]
    maxLoc2 = np.where(measureDists == np.max(measureDists))
    secondPeak = maxLoc2[0][0]

    # 找到两个峰值之间的最小值对应的灰度值，作为阈值
    thresh = 0
    if firstPeak > secondPeak:  # 第一个峰值在第二个峰值右侧
        temp = histogram[int(secondPeak):int(firstPeak)]
        minLoc = np.where(temp == np.min(temp))
        thresh = secondPeak + minLoc[0][0] + 1  # 有多个波谷取左侧的波谷
    else:
        temp = histogram[int(firstPeak):int(secondPeak)]
        minLoc = np.where(temp == np.min(temp))
        thresh = firstPeak + minLoc[0][0] + 1

    # 找到阈值后进行阈值处理，得到二值图
    threshImage_out = image.copy()
    threshImage_out[threshImage_out > thresh] = 255
    threshImage_out[threshImage_out <= thresh] = 0

    return (thresh, threshImage_out)


# THRESH_TRIANGLE与直方图技术法类似(效果更好)

img = cv2.imread(r'Data/2.JPG', cv2.IMREAD_GRAYSCALE)
the, dst = threshTwoPeaks(img)
the1 = 200
maxval = 255
the1, dst1 = cv2.threshold(img, the1, maxval, cv2.THRESH_TRIANGLE + cv2.THRESH_BINARY)
print('The thresh is :', the)
print('The thresh1 is :', the1)
cv2.imshow("image", img)
cv2.imshow('thresh_out', dst)
cv2.imshow('thresh_out1', dst1)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('3.png',dst)#保存图片