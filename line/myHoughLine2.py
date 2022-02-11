# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import math


def myHoughLine(imag, rho, theta, threshold):
    gray_img = cv.cvtColor(imag, cv.COLOR_RGB2GRAY)
    blur_img = cv.GaussianBlur(gray_img, (5, 5), 0)
    edge = cv.Canny(blur_img, 70, 140, apertureSize=5)

    width = imag.shape[1]
    height = imag.shape[0]

    R = round(math.sqrt(pow(height - 1, 2) + pow(width - 1, 2))) + 1
    rhos = np.linspace(-R, R, 2 * R)
    thetas = np.deg2rad(np.arange(0, 180, 1))  # 划分角度，精度为1，并生成一个数组,再转化为弧度制

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)  # 初始化投票器

    # 创建sin、cos列表
    sinTable = np.sin(thetas)
    cosTable = np.cos(thetas)

    # 投票计数
    for x in range(height):
        for y in range(width):
            if edge[x][y] != 0:
                for t in range(len(thetas)):
                    r = int(round(y * cosTable[t] + x * sinTable[t]) + R)
                    accumulator[(r, t)] += 1

    res = np.argwhere(accumulator > threshold)
    rhos_idx = res[:, 0]  # 获得符合要求的rho，即距离
    theta_idx = res[:, 1]  # 获得符合要求的theta，即角度
    # print (rhos_idx,theta_idx)
    # print (rhos[rhos_idx],thetas[theta_idx])

    res = [[], []]
    for i in range(1, len(rhos) - 1):
        for j in range(1, len(thetas) - 1):
            a = accumulator[i][j]
            if a > threshold:
                left = accumulator[i][j - 1]
                right = accumulator[i][j + 1]
                top = accumulator[i + 1][j]
                bottom = accumulator[i - 1][j]
                if a > left and a > right and a > top and a > bottom:
                    print(i, j)
                    res[0].append(rhos[i])
                    res[1].append(thetas[j])

    return rhos[rhos_idx], thetas[theta_idx]


if __name__ == '__main__':
    src = cv.imread("images/01.jpg")
    cv.imshow("images", src)
    rhos, thetas = myHoughLine(src, 1, 1, 195)
    for rho, theta in zip(rhos, thetas):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv.line(src, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv.imshow("line_detection", src)
    cv.waitKey()
    cv.destroyAllWindows()
