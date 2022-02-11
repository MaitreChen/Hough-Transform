import cv2 as cv
import numpy as np
from math import sqrt
from time import process_time


def houghGradint(image, dp, minDist, param1, param2, minRadius, maxRadius):
    """
    :param image:8位灰度图
    :param dp: 累加器分辨率
    :param minDist: 最小圆间距
    :param param1: 较高的通过Canny边缘检测器的两个阈值
    :param param2: 圆心累加器的阈值
    :param minRadius:  最小圆间距
    :param maxRadius:  最大圆间距
    """
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edge = cv.Canny(gray_img, 50, 150)

    # canny_thresh = param1  # canny的阈值
    vacuum_thresh = param2  # 中心累加器阈值

    sobel_x = cv.Sobel(edge, cv.CV_64F, dx=1, dy=0, ksize=3)  # x方向的梯度
    sobel_y = cv.Sobel(edge, cv.CV_64F, dx=0, dy=1, ksize=3)  # y方向的梯度

    rows = image.shape[0]
    cols = image.shape[1]
    accumulator = np.zeros((rows, cols))  # 建立圆心累加器
    none_zero = np.zeros((rows, cols))

    gaussian_filter = 1.0 / 20 * np.array([[1, 2, 1], [2, 8, 2], [1, 2, 1]])  # 高斯卷积核
    sobel_x_pad = np.pad(sobel_x, ((1, 1), (1, 1)), 'constant')
    sobel_y_pad = np.pad(sobel_y, ((1, 1), (1, 1)), 'constant')

    for i in range(rows):
        for j in range(cols):
            if not edge[i][j] == 0:  # 对非零点处理
                dx_fields = sobel_x_pad[i:i + 3, j:j + 3]
                dy_fields = sobel_y_pad[i:i + 3, j:j + 3]
                dx = (dx_fields * gaussian_filter).sum()
                dy = (dy_fields * gaussian_filter).sum()
                if dx != 0 and dy != 0:  # 导数值为零无意义
                    none_zero[(i, j)] = 1
                    k = dy / (dx + 0.00000001)
                    if abs(k) < 0.05 or abs(k) > 50:  # 筛选K
                        continue
                    b = j - k * i
                    for t in range(rows):  # 以一个像素为单位不断增加x值,计算y值
                        y = int(k * t + b)
                        if 0 < y < cols:  # 筛选y,不超过图像尺寸
                            accumulator[(t, y)] += 1

    cv.imshow("accumulator", accumulator)

    points = [[], []]
    for i in range(rows):
        for j in range(cols):
            a = accumulator[(i, j)]
            if a > vacuum_thresh:
                points[0].append(i)
                points[1].append(j)
                # left = accumulator[(i, j-1)]
                # right = accumulator[(i, j+1)]
                # bottom = accumulator[(i-1, j)]
                # right = accumulator[(i+1, j)]
                # if (a > left) & (a > right) & (a > bottom) & (a > right):
    # print(len(points[0]))
    sort_centers = []  # 处理坐标并加入累加值进行排序
    for k in range(len(points[0])):
        sort_centers.append([])
        sort_centers[-1].append(points[0][k])
        sort_centers[-1].append(points[1][k])
        sort_centers[-1].append(accumulator[(points[0][k], points[1][k])])

    print(sort_centers[:10])
    sort_centers.sort(key=lambda x: x[2], reverse=True)
    possible_centers = sort_centers[:(len(sort_centers))]  # 取出投票值较大的点作为圆心疑似点

    radii = maxRadius - minRadius  # 划分半径，建立累加器
    radius_accumulator = np.zeros(radii)

    for i in range(rows):
        for j in range(cols):
            if not edge[i, j] == 0:
                for t in range(len(possible_centers)):
                    x0 = possible_centers[t][0]
                    y0 = possible_centers[t][1]
                    r = int(sqrt((x0 - i) ** 2 + (y0 - j) ** 2))  # 计算每一个圆心疑似点与非零点的距离
                    if minRadius < r < maxRadius:
                        radius_accumulator[r - minRadius] += 1

    res = radius_accumulator.argsort()[::-1]  # 对半径降序排列

    true_radius = res[0] + minRadius
    print(true_radius)

    for i in range(rows):
        for j in range(cols):
            if not edge[i, j] == 0:  # 确定一个边缘点，求与圆心疑似点的距离，一旦大于设定值，则舍弃
                for k in range(len(possible_centers) - 1, -1, -1):  # 便于删除操作，故采用逆访问
                    a = possible_centers[k][0]
                    b = possible_centers[k][1]
                    dist = int(sqrt((a - i) ** 2 + (b - j) ** 2))
                    delta = 5  # 作为圆心与边缘点之间的差值
                    if abs(dist - true_radius) > delta:  # 距离大于设定值则舍弃
                        possible_centers.remove(possible_centers[k])

    print(possible_centers)

    circles = [[], [], []]
    length = len(possible_centers)
    if length != 0:
        a_mean = sum([possible_centers[i][0] for i in range(length)]) / length
        b_mean = sum([possible_centers[i][1] for i in range(length)]) / length
        circles[0].append(round(a_mean))
        circles[1].append(round(b_mean))
        circles[2].append(true_radius)
    else:
        print("no exit circle")

    return circles


def draw_circle(image, circle):
    cv.circle(image, (circle[1][0], circle[0][0]), circle[2][0], (0, 255, 0), 3)
    cv.circle(image, ((circle[0][0]), circle[1][0]), 2, (0, 255, 0), 3)
    cv.imshow("final_image", image)


if __name__ == "__main__":
    img = cv.imread("images/01.jpg")
    # cv.imshow("img", img)
    blur = cv.GaussianBlur(img, (3, 3), 0)  # 高斯滤波去噪

    t1 = process_time()
    circle = houghGradint(blur, 1, 0, 0, 3, 1, 120)
    t2 = process_time()
    print(t2 - t1)

    draw_circle(img, circle)

    cv.waitKey()
    cv.destroyAllWindows()
