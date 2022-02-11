# -*- coding: utf-8 -*-
"""
Created on Sun May 23 12:31:12 2021

@author: 19749
"""

import cv2 as cv
import numpy as np


def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edge = cv.Canny(gray, 50, 150, 5)
    lines = cv.HoughLines(edge, 1, np.pi / 180, 150)

    # print(lines[0])
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv.imshow("line_detection", image)


image = cv.imread("images/01.jpg")
line_detection(image)
cv.waitKey(0)
cv.destroyAllWindows()
