import cv2
import numpy as np
import utils

scale = 3
wP = 210 * scale
wH = 297 * scale


path = 'img3.png'
img = cv2.imread(path)
img, finalContours = utils.getContours(img, minArea=50000, filter=4)

if len(finalContours) != 0:
    biggest = finalContours[0][2]
    imgWarp = utils.warpImg(img, biggest, wP, wH)
    img2, finalContours2 = utils.getContours(imgWarp, minArea=1000, cnyThres=[1000,1000])

    if len(finalContours2) != 0:
        for obj in finalContours2:
            cv2.polylines(img2, [obj[2]], True, (0,255,0), 2)
            newPoints = utils.reorder(obj[2])
            nW = round(utils.findDis(newPoints[0][0]//scale, newPoints[1][0]//scale)/10, 1)
            nH = round(utils.findDis(newPoints[0][0]//scale, newPoints[2][0]//scale)/10, 1)
            cv2.arrowedLine(img2, (newPoints[0][0][0], newPoints[0][0][1]), (newPoints[1][0][0], newPoints[1][0][1]), (255,0,255), 3, 8, 0, 0.05)
            cv2.arrowedLine(img2, (newPoints[0][0][0], newPoints[0][0][1]), (newPoints[2][0][0], newPoints[2][0][1]), (255,0,255), 3, 8, 0, 0.05)
            x, y, w, h = obj[3]
            cv2.putText(img2, '{}cm'.format(nW), (x+30, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,0,255), 2)
            cv2.putText(img2, '{}cm'.format(nH), (x-70, y+h//2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,0,255), 2)

    cv2.imshow('A4', img2)

img = cv2.resize(img, (0,0), None, 0.5, 0.5)
# cv2.imshow('Image', img)
cv2.waitKey(0)