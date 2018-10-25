from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
import os
from PIL import Image
import numpy as np
from sklearn.externals import joblib
import cv2
import sys
import rsign
import sign_svm

OBSTACLE_THRESHOLD = 20

class Sign:
    def __init__(self):
        cwd = os.getcwd()
        self._model = sign_svm.Classify()
        self._model.load_model()
        self._roi_mask = None
 
    def detect(self, img):
        height, width = img.shape[:2]
        img0 = img[0:int(height/2),:]
        detected, mask = rsign.detect(img0)
        return detected, mask

    def roi_mask(self):
        return self._roi_mask
    #-1: no sign
    #0: right fork
    #1: left fork
    def predict(self, src_img):
        self._roi_mask = None
        detected, mask = self.detect(src_img)
        if not detected:
            return -1
        height, width = src_img.shape[:2]
        img = src_img[0:int(height/2),:]
        locs = rsign.location(mask)
        if locs[0] <= 0:
            return -1
        roi = img[locs[3]:locs[1], locs[2]: locs[0]]
        self._roi_mask = mask[locs[3]:locs[1], locs[2]: locs[0]]
        #cv2.imshow("ROI", roi)
        return self._model.predict(roi)

    def predict_debug(self, img, dirname):
        self._roi_mask = None
        detected, mask = rsign.detect(img)
        if not detected:
            return -1

        locs = rsign.location(mask)
        if locs[0] <= 0:
            return -1
        roi = img[locs[3]:locs[1], locs[2]: locs[0]]
        self._roi = mask[locs[3]:locs[1], locs[2]: locs[0]]
        return self._model.predict_debug(roi, dirname)

def hsv_mask(img):
    blur = cv2.GaussianBlur(img, (5,5), 0)
    blur = cv2.GaussianBlur(blur, (5,5), 0)
    blur = cv2.GaussianBlur(blur, (5,5), 0)

    hsv_low = np.array([85, 0, 50])
    hsv_high = np.array([140, 65, 255])

    height, width = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    return mask

def track_umask(img):
    flat = rgb_flatten(img)
    r, g, b = cv2.split(flat)

    r_channel = 255 - r
    g_channel = 255 - g
    b_channel = 255 - b

    mask = cv2.bitwise_and(r_channel, b_channel)
    mask = cv2.bitwise_and(mask, g_channel)

    mask = cv2.blur(mask, (5, 5))
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    return mask


def rgb_flatten(img):
    r, g, b = cv2.split(img)
    r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
    g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
    b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
    y_filter = ((r >= 128) & (g >= 128) & (b < 100))

    r[y_filter], g[y_filter] = 255, 255
    b[np.invert(y_filter)] = 0

    b[b_filter], b[np.invert(b_filter)] = 255, 0
    r[r_filter], r[np.invert(r_filter)] = 255, 0
    g[g_filter], g[np.invert(g_filter)] = 255, 0

    flattened = cv2.merge((r, g, b))
    return flattened


# detect if sign of stand is blank
# 1: blank
# 0: not blank
# -1: unknown
def detect_obstacle(img):

    invalid = [-1, -1]
    h, w = img.shape[:2]
    img = img[0:int(h/2),:]
    hmask = hsv_mask(img)
    #cv2.imshow("hsv mask", hmask)
    #mask = hmask
    tmask = track_umask(img)
    #cv2.imshow("track umask", tmask)
    mask = cv2.bitwise_and(tmask, hmask)
    #print("hsv sum:", np.sum(mask > 0))
    stand_on = np.sum(mask > 0) > OBSTACLE_THRESHOLD
    if not stand_on:
        return invalid

    sn_half = img[0:int(h/2), :]
    sign_mask = rsign._mask(sn_half)
#    print("sign num:", np.sum(sign_mask > 0))
    sign_on = np.sum(sign_mask > 0) > 10

    if sign_on:
        return invalid

    locs = np.where(mask > 0)
    h = int(np.sum(locs[0]) / len(locs[0].tolist()))
    w = int(np.sum(locs[1]) / len(locs[1].tolist()))

    if h < 50:
        return invalid
    loc = [w, h]
    #cv2.circle(img, (w,h), 4, (0,0,255), -1)
    return loc
