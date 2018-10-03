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
import obstacle

SIGN_STRING = ["NONE", "Right Fork", "Left Fork"]

class Sign:
    def __init__(self):
        cwd = os.getcwd()
        self._model = sign_svm.Classify()
        self._model.load_model()
 
    def detect(self, img):
        height, width = img.shape[:2]
        img0 = img[0:int(height/2),:]
        detected, mask = rsign.detect(img0)
        return detected, mask

    #-1: no sign
    #0: right fork
    #1: left fork
    def predict(self, src_img):
        detected, mask = self.detect(src_img)
        if not detected:
            return -1
        height, width = src_img.shape[:2]
        img = src_img[0:int(height/2),:]
        locs = rsign.location(mask)
        if locs[0] <= 0:
            return -1
        roi = img[locs[3]:locs[1], locs[2]: locs[0]]
        #cv2.imshow("ROI", roi)
        return self._model.predict(roi)

    def predict_debug(self, img, dirname):
        detected, mask = rsign.detect(img)
        if not detected:
            return -1

        locs = rsign.location(mask)
        if locs[0] <= 0:
            return -1
        roi = img[locs[3]:locs[1], locs[2]: locs[0]]
        return self._model.predict_debug(roi, dirname)
    

# detect if sign of stand is blank
# 1: blank
# 0: not blank
# -1: unknown
def blank(src_img):
        height, width = src_img.shape[:2]
        half = src_img[0:int(height*2/6), :]
        gray = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        umask = 255 - mask
        hsv = obstacle.hsv_mask(half)
        mask = cv2.bitwise_and(umask, hsv)
        stand_on = np.sum(mask[mask == 255].size) > 200

        sn_half = src_img[0:int(height/2), :]
        color = rsign._mask(sn_half)
        sn = cv2.bitwise_and(sn_half, sn_half, mask = color)
        sign_mask = cv2.threshold(sn, 200, 255, cv2.THRESH_BINARY)[1]
        sign_on = np.sum(sign_mask[sign_mask == 255].size) > 10

        on = -1
        if sign_on:
            on = 0
        elif stand_on:
            on = 1
        return on

        
if __name__=='__main__':
    if len(sys.argv) < 2:
        exit(0)

    file_name = sys.argv[1]
    img = cv2.imread(file_name)

    #predict sign class
    sn = Sign()
    idx = sn.predict(img)
    print("sign: ", SIGN_STRING[idx+1])
    
