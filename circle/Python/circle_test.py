# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 20:35:21 2021

@author: thebin
"""

import cv2 as cv
import numpy as np

planets = cv.imread("images/01.jpg")
gay_img = cv.cvtColor(planets, cv.COLOR_BGRA2GRAY)
img = cv.medianBlur(gay_img, 7)  # 进行中值模糊，去噪点
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=30, maxRadius=220)

circles = np.uint16(np.around(circles))
print(circles)

for i in circles[0, :]:  # 遍历矩阵每一行的数据
    cv.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 3)
    cv.circle(planets, (i[0], i[1]), 2, (0, 255, 0), 3)

cv.imshow("gay_img", planets)
cv.waitKey(0)
cv.destroyAllWindows()