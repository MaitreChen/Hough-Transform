from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv

# read image
image = cv.imread('test1.jpg')
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image_blur = cv.medianBlur(image_gray,5)

# get circles
circles = cv.HoughCircles(image_gray,cv.HOUGH_GRADIENT,1,300,param1=100,param2=50,minRadius=110,maxRadius=250)
circles = np.uint16(np.around(circles))

# get radius of circles
radius_set = circles[0,:][:,2]
radius_set = np.array(radius_set).reshape(-1,1)

# exclude circles with large radius differences
# method: k-means
num_clusters = 2
km = KMeans(n_clusters=num_clusters).fit(radius_set)

labels = km.labels_
d = {}
for label in labels:
    d[label] = d.get(label, 0) + 1

labels_sort = sorted(d.items(),key=lambda x:x[1],reverse=True)
true_label = labels_sort[0][0]
print(f'true label is: {true_label}')
print(circles[0,:])

# count the real circle
count = 0
for i, v in enumerate(circles[0,:]):
    if labels[i] != true_label:
        continue

    cv.circle(image,(v[0],v[1]),v[2],(0,255,0),12)
    cv.circle(image,(v[0],v[1]),3,(0,255,0),14)
    count += 1


# plot the result
plt.title("Coin Detection Result")
plt.text(0.05,0.95,f'The number of coins is: {count}',fontsize=12,
        verticalalignment='top',color='red')
image_rgb = cv.cvtColor(image,cv.COLOR_BGR2RGB) 
plt.imshow(image_rgb)
plt.show()  

