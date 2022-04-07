from tabnanny import verbose
import cv2
import math
import numpy as np


def print_img_info(img, gain=1, verbose=0):
    print(np.min(img), np.max(img), np.mean(img), img.shape, img.dtype)
    if verbose:
        print(img)
    cv2.imshow('tmp', img * gain)
    cv2.waitKey()


bgr = cv2.imread('Data2/6.JPG')



l_X = [88,90,91,95,98,99,105,106,112,115,117,118,120,123,124,125,127,128,131,133,136,138,139,140,142,143,144,145,146,147,148,149,150,151,152,155,158
]
l_Y = [17,20,22,28,33,34,42,44,51,55,58,59,62,66,67,69,72,73,78,82,87,91,93,95,100,102,104,107,109,112,114,117,120,123,126,135,145
]
r_X = [142,143,145,146,150,153,154,155,156,165,166,173,183,186,193,194,195,196,200,204,207,208,211,215,216,217,221,222,224,226,227,229,230,236,239,246,247,249,256,258,260,261,262,263,265,266,267,268,269,270,272,273,274,276,277,278,279,280,282,286,287,290,291,292,293,294,295,301,306,311,313,316,319,322,325,328,329,330,331,333
]
r_Y = [39,39,39,39,39,39,39,39,39,40,41,43,47,49,53,53,54,55,57,60,63,63,66,69,70,71,74,75,77,79,80,82,83,88,91,98,99,101,108,110,113,114,115,116,118,119,120,121,122,123,125,126,127,129,130,131,132,133,135,139,140,143,144,145,146,147,148,153,158,162,164,166,169,171,173,175,176,176,177,178
]

assert len(r_X) == len(r_Y)
assert len(l_X) == len(l_Y)
# print_img_info(bgr)
H,W,_=bgr.shape
print(H,W)
print(len(l_X), min(l_X), max(l_X))
print(len(l_Y), min(l_Y), max(l_Y))
print(len(r_X), min(r_X), max(r_X))
print(len(r_Y), min(r_Y), max(r_Y))
for i in range(len(l_X)-1):
       if l_Y[i]>H:
              l_Y[i]=H
       cv2.line(bgr, (l_X[i], l_Y[i]), (l_X[i+1], l_Y[i+1]), (0, 255, 0), 3)
for i in range(len(r_X) - 1):
       if r_Y[i] > H:
              r_Y[i] = H
       cv2.line(bgr, (r_X[i], r_Y[i]), (r_X[i + 1], r_Y[i + 1]), (0, 0, 255), 3)
       # cv2.line(bgr, (r_X[i], r_Y[i]), (r_X[i + 1], r_Y[i + 1]), (0, 255, 0), 3)
# print_img_info(bgr)
cv2.imwrite('62.png', bgr)