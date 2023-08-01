import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img6.png')
blurred_image = cv2.GaussianBlur(img,(5,5),0)
edges = cv2.Canny(blurred_image,100,200)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_cp = img.copy()
for contour in contours:
        if cv2.contourArea(contour) < 200 and cv2.contourArea(contour) > 100:
            cv2.drawContours(img_cp, [contour], -1, (0,255,0), 2)
plt.imshow(img_cp[:,:,::-1])

plt.show()