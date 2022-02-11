import cv2 as cv
import numpy as np
import math


def myHoughCircle(image, threshold=150):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edge = cv.Canny(gray, 50, 100, apertureSize=3)
    cv.imshow("Edged image", edge)
    height = image.shape[0]
    width = image.shape[1]

    a_cir = np.arange(0, width, 1)  # 细分圆心参数a,b
    b_cir = np.arange(0, height, 1)

    r_min = 0  # 细分圆半径r
    r_max = round(math.sqrt(pow(height - 1, 2) + pow(width - 1, 2))) + 1
    r_size = 1
    r_cir = np.arange(r_min, r_max, r_size)

    accumulator = np.zeros((len(a_cir), len(b_cir), len(r_cir)))  # 初始化投票器

    for x in range(height):  # 投票计数
        for y in range(width):
            if (edge[x][y] != 0):
                for a in range(len(a_cir)):
                    for b in range(len(b_cir)):
                        r = int(math.sqrt((x - a_cir[a]) ** 2 + (y - b_cir[b]) ** 2))
                        accumulator[(a, b, r)] += 1

    res = np.argwhere(accumulator > threshold)
    a0 = res[:, 0]
    b0 = res[:, 1]
    r0 = res[:, 2]

    for i in zip(*(a0, b0, r0)):
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 3)
        cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv.imshow('result', image)


if __name__ == '__main__':
    pic = cv.imread("images/01.jpg")
    cv.imshow("Initial image", pic)
    myHoughCircle(pic, 150)

    cv.waitKey(0)
    cv.destroyAllWindows()

