from tabnanny import verbose
from turtle import bgcolor
import cv2
import math
import numpy as np
from scipy.optimize import leastsq
from skimage import morphology
from matplotlib import pyplot as plt


def print_img_info(img, img_name, verbose=0, save_name=None):
    print(np.min(img), np.max(img), np.mean(img), img.shape, img.dtype)
    if verbose:
        print(img)
    if np.max(img) < 2:
        img = (img * 255).astype('uint8')
    else:
        img = (img * 1).astype('uint8')
    if save_name is not None:
        cv2.imwrite(img_name + '_' + save_name + '.jpg', img)
    else:
        cv2.imshow(img_name, img)
        cv2.waitKey()


def even_light(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    H, W = h.shape

    # r = min(H, W)
    r = 19
    gau1 = cv2.GaussianBlur(v, (r, r), 15 / math.sqrt(2)).astype('float32')
    gau2 = cv2.GaussianBlur(v, (r, r), 80 / math.sqrt(2)).astype('float32')
    gau3 = cv2.GaussianBlur(v, (r, r), 250 / math.sqrt(2)).astype('float32')
    gau = (gau1 + gau2 + gau3) / 3 / 255
    m = np.mean(gau)
    v = v.astype('float32') / 255
    out = v.copy()
    for i in range(H):
        for j in range(W):
            out[i, j] = math.pow(v[i, j], math.pow(0.5, ((m - gau[i, j]) / m)))
    hsv2 = hsv.copy()
    hsv2[:, :, 0] = h
    hsv2[:, :, 1] = s
    hsv2[:, :, 2] = (out * 255).astype('uint8')
    bgr2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    return bgr2


def rgb2hsv(r, g, b):
    # assert r<1 float32
    h, w = r.shape
    H = np.zeros((h, w), np.float32)
    S = np.zeros((h, w), np.float32)
    V = np.zeros((h, w), np.float32)
    for i in range(0, h):
        for j in range(0, w):
            mx = max((b[i, j], g[i, j], r[i, j]))
            mn = min((b[i, j], g[i, j], r[i, j]))
            dt = mx - mn

            if mx == mn:
                H[i, j] = 0
            elif mx == r[i, j]:
                if g[i, j] >= b[i, j]:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / dt)
                else:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / dt) + 360
            elif mx == g[i, j]:
                H[i, j] = 60 * ((b[i, j]) - r[i, j]) / dt + 120
            elif mx == b[i, j]:
                H[i, j] = 60 * ((r[i, j]) - g[i, j]) / dt + 240

            if mx == 0:
                S[i, j] = 0
            else:
                S[i, j] = 1 - mn / mx

            V[i, j] = mx
    H = H / 360
    return H, S, V


def gen_hue_mask(bgr):
    r = bgr[:, :, 2].astype('float32') / 255
    g = bgr[:, :, 1].astype('float32') / 255
    b = bgr[:, :, 0].astype('float32') / 255

    h, _, _ = rgb2hsv(r, g, b)

    H, W = h.shape
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

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    hue_mask = cv2.dilate(hue_mask, kernel)
    hue_mask = cv2.dilate(hue_mask, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    hue_mask = cv2.erode(hue_mask, kernel)
    hue_mask = cv2.erode(hue_mask, kernel)
    hue_mask = cv2.erode(hue_mask, kernel)
    hue_mask = cv2.erode(hue_mask, kernel)
    hue_mask = cv2.erode(hue_mask, kernel)
    hue_mask = cv2.erode(hue_mask, kernel)

    hue_mask = np.where(hue_mask < 0.5, 0, 1)

    hue_mask = 1 - hue_mask

    return hue_mask


def gen_crop_mask(gray):
    crop_mask = np.ones_like(gray)
    H, W = crop_mask.shape
    crop_mask[1:int(H / 8), :] = 0
    crop_mask[H - int(H / 14):, :] = 0
    crop_mask[:, 1:int(W / 14)] = 0
    crop_mask[:, W - int(W / 14):] = 0
    crop_mask = crop_mask.astype('float32')
    return crop_mask


def gen_left_right_mask(edge):
    t_x_l, t_y_l, b_x_l, b_y_l = 0, 0, 0, 0
    t_x_r, t_y_r, b_x_r, b_y_r = 0, 0, 0, 0
    H, W = edge.shape

    flag = 0
    for i in range(H):
        for j in range(W):
            if edge[i, j] > 0:
                t_y_l = i
                t_x_l = j
                flag = 1
                break
        if flag:
            break
    t_y_r = t_y_l

    for i in range(W - 1, 0, -1):
        if edge[t_y_l, i] > 0:
            t_x_r = i
            break

    flag = 0
    for i in range(H):
        for j in range(W - 1, 0, -1):
            if edge[i, j] > 0:
                b_y_l = i
                b_x_l = j
                break
        if flag:
            break
    b_y_r = b_y_l

    for i in range(W - 1, 0, -1):
        if edge[b_y_l, i] > 0:
            b_x_r = i
            break

    # print(t_x_l,t_y_l,b_x_l,b_y_l)
    # print(t_x_r,t_y_r,b_x_r,b_y_r)

    t_x = (t_x_l + t_x_r) // 2
    t_y = (t_y_l + t_y_r) // 2
    b_x = (b_x_l + b_x_r) // 2
    b_y = (b_y_l + b_y_r) // 2

    left_mask = np.zeros_like(edge)
    right_mask = np.zeros_like(edge)
    for i in range(H):
        for j in range(W):
            t = ((b_y - t_y) / (b_x - t_x)) * (j - t_x) + t_y
            if t < i:
                left_mask[i, j] = 1
            else:
                right_mask[i, j] = 1
    # print_img_info(left_mask, gain=255, save=1)
    return left_mask, right_mask


def filt_forground_with_area_thresh(edge, hard_thresh=0):
    edge1 = (edge * 255).astype('uint8')
    edge2 = np.zeros_like(edge1)
    contours, _ = cv2.findContours(edge1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not hard_thresh:
        areas = []
        for cont in contours:
            area = cv2.contourArea(cont)
            areas.append(area)
        area_thresh = np.mean(areas)
    else:
        area_thresh = 1000

    for cont in contours:
        area = cv2.contourArea(cont)
        if area > area_thresh:
            cv2.drawContours(edge2, [cont], -1, 255, -1)

    edge = np.where(edge2 > 0, 1, 0).astype('float32')
    # print_img_info(edge, gain=255, save=1)
    return edge


def fit_curve(x, y):
    x = np.array(x)
    y = np.array(y)

    def Fun(p, x):
        a1, a2, a3 = p
        return a1 * x ** 2 + a2 * x + a3

    def error(p, x, y):
        return Fun(p, x) - y

    p0 = [0.1, -0.01, 100]
    para = leastsq(error, p0, args=(x, y))
    y_fitted = Fun(para[0], x)
    return y_fitted


if __name__ == '__main__':
    img_name = '3'

    # 读原图和边缘图
    bgr = cv2.imread('Data2/' + img_name + '.JPG')
    edge = cv2.imread('Data2/' + img_name + '1.png', 0)
    edge_ori_u8 = edge.copy()
    edge = np.where(edge > 0, 1, 0).astype('float32')
    edge_ori = edge.copy()

    # 可视化边缘图
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edge_ori_u8 = cv2.dilate(edge_ori_u8, kernel)
    edge_ori_u8 = np.where(edge_ori_u8 > 0, 1, 0)
    gray2 = gray * (~edge_ori_u8.astype('bool'))
    print_img_info(gray2, img_name=img_name, save_name='gray2')

    # 亮度均衡
    bgr2 = even_light(bgr)
    print_img_info(bgr2, img_name=img_name, save_name='bgr2')

    # 绿植掩码
    hue_mask = gen_hue_mask(bgr2)
    print_img_info(hue_mask, img_name=img_name, save_name='hue_mask')

    # 裁剪掩码
    crop_mask = gen_crop_mask(gray)
    print_img_info(crop_mask, img_name=img_name, save_name='crop_mask')

    # 结合掩码后的边缘区域
    edge = edge * hue_mask
    edge = edge * crop_mask
    print_img_info(edge, img_name=img_name, save_name='edge_masked')

    # 边缘图形态学处理
    edge = cv2.GaussianBlur(edge, (3, 3), 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge = cv2.dilate(edge, kernel)
    edge = cv2.dilate(edge, kernel)
    edge = cv2.dilate(edge, kernel)
    edge = cv2.dilate(edge, kernel)
    edge = cv2.erode(edge, kernel)
    edge = np.where(edge > 0, 1, 0)
    print_img_info(edge, img_name=img_name, save_name='edge_morph')

    # 边缘区域骨架细化
    skeleton = morphology.skeletonize(edge)
    print_img_info(skeleton, img_name=img_name, save_name='edge_skeleton')

    # 左右掩码
    left_mask, right_mask = gen_left_right_mask(skeleton)

    # 对左右掩码分别筛除面积小的区域
    edge_left = edge * left_mask
    edge_left = filt_forground_with_area_thresh(edge_left, hard_thresh=0)
    print_img_info(edge_left, img_name=img_name, save_name='edge_left')
    edge_right = edge * right_mask
    edge_right = filt_forground_with_area_thresh(edge_right, hard_thresh=0)
    print_img_info(edge_right, img_name=img_name, save_name='edge_right')

    # 取边缘区域的边缘点
    edge_left = edge_left * edge_ori
    print_img_info(edge_left, img_name=img_name, save_name='edge_left_ori')
    edge_right = edge_right * edge_ori
    print_img_info(edge_right, img_name=img_name, save_name='edge_right_ori')

    # 获取左右边缘点的坐标
    H, W = gray.shape
    left_X = []
    left_Y = []
    right_X = []
    right_Y = []
    for i in range(W):
        flag = 0
        for j in range(H):
            if flag:
                break
            if edge_left[j, i] > 0:
                left_X.append(i)
                left_Y.append(j)
                flag = 1
    for i in range(W):
        flag = 0
        for j in range(H):
            if flag:
                break
            if edge_right[j, i] > 0:
                right_X.append(i)
                right_Y.append(j)
                flag = 1

    # 拟合坐标
    left_Y2 = fit_curve(left_X, left_Y).astype('uint8')
    right_Y2 = fit_curve(right_X, right_Y).astype('uint8')
    plt.plot(left_X, H - np.array(left_Y))
    plt.plot(left_X, H - np.array(left_Y2))
    plt.plot(right_X, H - np.array(right_Y))
    plt.plot(right_X, H - np.array(right_Y2))
    plt.savefig(img_name + 'curve.png')

    # 于原图画出拟合结果
    print(len(left_X), min(left_X), max(left_X))
    print(len(left_Y2), min(left_Y2), max(left_Y2))
    print(len(right_X), min(right_X), max(right_X))
    print(len(right_Y2), min(right_Y2), max(right_Y2))
    for i in range(len(left_X) - 1):
        if left_Y2[i] > H:
            left_Y2[i] = H
        cv2.line(bgr, (left_X[i], left_Y2[i]), (left_X[i + 1], left_Y2[i + 1]), (0, 255, 0), 3)
    for i in range(len(right_X) - 1):
        if right_Y2[i] > H:
            right_Y2[i] = H
        cv2.line(bgr, (right_X[i], right_Y2[i]), (right_X[i + 1], right_Y2[i + 1]), (0, 0, 255), 3)
    print_img_info(bgr, img_name=img_name, save_name='final')
