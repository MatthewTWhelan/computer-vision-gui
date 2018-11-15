import numpy as np
import cv2 as cv

class openCV_testing:

    def __init__(self):
        self.img = cv.imread('image.jpg')

    def img_show(self, img):
        cv.imshow('image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def smoothing_operations(self, kernel_size):
        img = self.img
        self.img_show(img)
        kernel = np.ones((kernel_size,kernel_size), np.float32) / kernel_size**2
        img_smooth = cv.filter2D(img, -1, kernel)

    def gaussian_blur(self, kernel_size):
        img = self.img
        self.img_show(img)
        img_gauss = cv.GaussianBlur(img, (5,5), 0)

    def edge_detection(self):
        img = self.img
        self.img_show(img)
        img_edge = cv.Canny(img, 100, 200)

    def colour_segmentation(self, lower_range, upper_range):
        # colour is RGB
        img = self.img
        self.img_show(img)
        img_mask = cv.inRange(img, lower_range, upper_range)
        img_result = cv.bitwise_and(img, img, mask=img_mask)

    def image_resize(self):
        img  = self.img
        self.img_show(img)
        img_resize = cv.resize(img, (100,100))