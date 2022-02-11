# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import math
from PIL import Image


def myHoughLine(imag, rho, theta, threshold):
    gray_img = cv.cvtColor(imag, cv.COLOR_RGB2GRAY)
    blur_img = cv.GaussianBlur(gray_img, (5, 5), 0)
    edge = cv.Canny(blur_img, 50, 150, apertureSize=5)

    height, width = imag.shape[:2]

    R = round(math.sqrt(pow(height - 1, 2) + pow(width - 1, 2))) + 1  # 计算出最大的r
    rhos = np.linspace(-R, R, 2 * R)
    thetas = np.deg2rad(np.arange(0, 180, 1))  # 划分角度，精度为1，并生成一个数组,再转化为弧度制

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)  # 初始化投票器

    # 创建sin、cos列表
    sinTable = [np.sin(t * np.pi / 180) for t in range(len(thetas))]
    cosTable = [np.cos(t * np.pi / 180) for t in range(len(thetas))]

    # 投票计数
    for x in range(height):
        for y in range(width):
            if edge[x][y] != 0:
                for t in range(len(thetas)):
                    r = int(round((x * cosTable[t] + y * sinTable[t]) / rho))
                    accumulator[(r, t)] += 1

    accum = np.uint8(accumulator.T)
    cv.imshow("accum", accum)  # 显示霍夫空间

    # 阈值化并查找局部最大值
    res = [[], []]
    for i in range(len(rhos)):
        for j in range(len(thetas)):
            a = accumulator[i][j]
            if a > threshold:
                '''
                left = accumulator[i][j-1]
                right = accumulator[i][j+1]
                top = accumulator[i+1][j]
                bottom = accumulator[i-1][j]
                '''
                # if a>left and a >right and a>top and a>bottom:
                res[0].append(i)
                res[1].append(j)
    print(res[0], res[1])
    # 绘制直线
    for i in range(0, len(res[1])):
        rho = res[0][i]
        theta = math.pi / 2 - res[1][i] * math.pi / 180  # 转为弧度制，且是90°的余角   对于含≥90°角且单条直线 or ＜90°且多条直线，不转化则显示出对称直线！！

        a = np.cos(theta)
        b = np.sin(theta)
        if theta > math.pi / 2 or theta < 0:  # 对于含≥90°角的直线，忽略ttheta<0，则无法检测到！！
            x0 = a * (rho - 2 * R)
            y0 = b * (rho - 2 * R)
        else:
            x0 = a * rho
            y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv.line(imag, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv.imshow("line_detection", imag)


if __name__ == '__main__':
    src = cv.imread("images/01.jpg")
    cv.imshow("src", src)
    myHoughLine(src, 1, 1, 200)

    cv.waitKey()
    cv.destroyAllWindows()
