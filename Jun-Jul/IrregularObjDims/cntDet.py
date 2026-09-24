import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img6.png')
img_cp = img.copy()
gray = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
gray_inverted = cv2.bitwise_not(gray)

_, binary = cv2.threshold(gray_inverted, 100, 255, cv2.THRESH_BINARY)
img_cp2 = img.copy()
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) > 65000:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_cp2, (x,y), (x+w, y+h), (255,0,0), 2)

plt.imshow(img_cp2[:,:,::-1])
plt.show()