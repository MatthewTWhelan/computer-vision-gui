#!/usr/bin/python3

import cv2 as cv

img = cv.imread("images/smoothing_blurring_help.png")
img_resize = cv.resize(img, (541,1623))
cv.imwrite("images/smoothing_blurring_help.jpg", img_resize)
