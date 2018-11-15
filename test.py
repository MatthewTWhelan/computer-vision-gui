#!/usr/bin/python3

import cv2 as cv

img = cv.imread("arrow.png")
img_resize = cv.resize(img, (25,25))
cv.imwrite("arrow.jpg", img_resize)
