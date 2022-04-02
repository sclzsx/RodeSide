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

bgr = cv2.imread('Data/5.JPG')

def even_light(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2] / 360
    H,W = h.shape
    r = 11
    gau1 = cv2.GaussianBlur(v, (r,r), 15 / math.sqrt(2))
    gau2 = cv2.GaussianBlur(v, (r,r), 80 / math.sqrt(2))
    gau3 = cv2.GaussianBlur(v, (r,r), 250 / math.sqrt(2))
    gau = (gau1 + gau2 + gau3) / 3
    m = np.mean(gau)
    out = v
    for i in range(H):
        for j in range(W):
            out[i, j] = math.pow(v[i, j], math.pow(0.5, ((m - gau[i, j]) / m)))
    hsv2 = hsv
    hsv2[:,:,0] = h
    hsv2[:,:,1] = s
    hsv2[:,:,2] = out * 360
    bgr2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    return bgr2

bgr2 = even_light(bgr)

def gen_hue_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0] / 360
    H,W = h.shape
    hue_mask = np.zeros_like(h)
    for i in range(H):
        for j in range(W):
            t = h[i, j]
            if t < 0.25:
                hue_mask[i, j] = 13.6 * t * t
            elif t < 0.42:
                hue_mask[i, j] = -18.5 * (t - 0.3333) * (t - 0.3333) + 1
            else:
                hue_mask[i, j] = 0.003624 * math.pow(t, -6.29)
    # hue_mask = hue_mask * 0.3
    hue_mask = cv2.resize(hue_mask, None, fx=0.25, fy=0.25)
    # print_img_info(hue_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    hue_mask = cv2.dilate(hue_mask, kernel)
    
    hue_mask = cv2.resize(hue_mask, (W, H))
    # hue_mask = hue_mask * hue_mask
    # hue_mask = np.clip(hue_mask, 0, 1)
    # hue_mask = 1 - hue_mask
    # print_img_info(hue_mask)
    hue_mask[hue_mask < 0.95] = 0
    hue_mask[hue_mask >= 0.95] = 1
    hue_mask = hue_mask.astype('bool')
    # hue_mask = np.where(hue_mask < 0.9, 1, 0)
    # print_img_info(hue_mask)
    # print_img_info(hue_mask)
    # hue_mask[hue_mask > 0.1] = 255
    # hue_mask = np.where(hue_mask < 0.1, 1, 0)
    # print_img_info(hue_mask, verbose=1)
    return hue_mask


hue_mask = gen_hue_mask(bgr2)
# print_img_info(hue_mask, verbose=1)

gray = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray, (3, 3))

# kernel = np.array([
# [1, 0, -1],
# [2, 0, -2],
# [1, 0, -1]], dtype="float32")
# X = cv2.filter2D(gray, -1, kernel)
# kernel = np.array([
# [1, 2, 1],
# [0, 0, 0],
# [-1, -2, -1]], dtype="float32")
# Y = cv2.filter2D(gray, -1, kernel)
# sob = np.power(np.power(X, 2) + np.power(Y, 2), 0.5)

gx = cv2.Sobel(gray,cv2.CV_16S,1,0)
gy = cv2.Sobel(gray,cv2.CV_16S,0,1)
gx = cv2.convertScaleAbs(gx)
gy = cv2.convertScaleAbs(gy)
sob = cv2.addWeighted(gx,0.5,gy,0.5,0)
th = np.mean(sob) * 1.5
sob[sob < th] = 0
sob[sob >= th] = 255
# print_img_info(sob)
sob = cv2.GaussianBlur(sob,(5,5), 3)

sob = sob * hue_mask

print_img_info(sob)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
sob_big = cv2.erode(sob, kernel)
sob_small = cv2.dilate(sob, kernel)
sob_diff = sob_big - sob_small
# print_img_info(sob_diff)

crop_mask = np.ones_like(gray)
H,W = crop_mask.shape
crop_mask[1:int(H/4),:] = 0
crop_mask[:, 1:int(W/10)] = 0
crop_mask[:, W-int(W/10):] = 0
edge = sob_diff * hue_mask
edge = edge * crop_mask

t_x_l,t_y_l,b_x_l,b_y_l = 0, 0, 0, 0
t_x_r,t_y_r,b_x_r,b_y_r = 0, 0, 0, 0

for i in range(H):
    for j in range(W):
        if edge[i, j] > 0:
            t_y_l = i
            t_x_l = j
            break
t_y_r = t_y_l

for i in range(W-1, 0, -1):
    if edge[t_x_l, t_y_l] > 0:
        t_x_r = i
        break

for i in range(H):
    for j in range(W-1, 0, -1):
        if edge[i, j] > 0:
            b_y_l = i
            b_x_l = j
            break
b_y_r = b_y_l

for i in range(W-1, 0, -1):
    if edge[b_y_l, i] > 0:
        b_x_r = i
        break

t_x = (t_x_l + t_x_r) // 2
t_y = (t_y_l + t_y_r) // 2
b_x = (b_x_l + b_x_r) // 2
b_y = (b_y_l + b_y_r) // 2

left_mask = np.zeros_like(edge)
right_mask = np.zeros_like(edge)
for i in range(H):
    for j in range(W):
        t = ((b_y - t_y) / (b_x - t_x)) * (j - t_x) + t_y
        if t > i:
            left_mask[i, j] = 1
        else:
            right_mask[i, j] = 1

left_edge = edge
left_edge[left_mask < 1] = 0

right_edge = edge
right_edge[right_mask < 1] = 0

