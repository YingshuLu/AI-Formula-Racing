import os

import cv2
import numpy as np


class ImageProcessor(object):
    @staticmethod
    def show_image(img, name="image", scale=1.0):
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, img)
        cv2.waitKey(1)

    @staticmethod
    def save_image(folder, img, prefix="img", date_str="", suffix="", filename=""):
        if not filename:
            filename = "%s-%s%s.jpg" % (prefix, date_str, suffix)
        cv2.imwrite(os.path.join(folder, filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    @staticmethod
    def rad2deg(radius):
        return radius / np.pi * 180.0

    @staticmethod
    def deg2rad(degree):
        return degree / 180.0 * np.pi

    @staticmethod
    def bgr2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def preprocess(img):
        img = cv2.GaussianBlur(img, (5, 5), 0)
        return img

    @staticmethod
    def count_color(img_color, mask_color1, mask_color2=None):
        mask1 = cv2.inRange(img_color, mask_color1[0], mask_color1[1])
        if mask_color2:
            mask2 = cv2.inRange(img_color, mask_color2[0], mask_color2[1])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = mask1
        mask_color_count = cv2.countNonZero(mask)
        return mask_color_count
