from __future__ import print_function

from time import time
from PIL  import Image
from io   import BytesIO

import os
import cv2
import math
import numpy as np
import base64
import logging
import collections

class ImageProcessor(object):
    @staticmethod
    def show_image(img, name = "image", scale = 1.0):
        if scale and scale != 1.0:
            img = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC) 

        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, ImageProcessor.bgr2rgb(img))
        cv2.waitKey(1)

    @staticmethod
    def show_hsv_image(img, name = "image", scale = 1.0):
        if scale and scale != 1.0:
            img = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC) 

        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, img)
        cv2.waitKey(1)

    @staticmethod
    def save_image(folder, img, prefix = "img", suffix = ""):
        from datetime import datetime
        filename = "%s-%s%s.jpg" % (prefix, datetime.now().strftime('%Y%m%d-%H%M%S-%f'), suffix)
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
    def _flatten_rgb(img):
        r, g, b = cv2.split(img)
        r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
        g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
        b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
        y_filter = ((r >= 128) & (g >= 128) & (b < 100))
        w_filter = ((r > 200) & (g > 200) & (b > 200))

        # r[y_filter], g[y_filter] = 255, 255
        # b[np.invert(y_filter)] = 0

        b[b_filter], b[np.invert(b_filter)] = 255, 0
        r[r_filter], r[np.invert(r_filter)] = 255, 0
        g[g_filter], g[np.invert(g_filter)] = 255, 0

        r[w_filter], g[w_filter], b[w_filter] = 255, 255, 255 # show white

        flattened = cv2.merge((r, g, b))
        return flattened

    @staticmethod
    def preprocess(img):
        img = ImageProcessor._flatten_rgb(img)
        return img
    
    @staticmethod
    def showimg_console(img):
        for i in range (0,240, 5):
            for j in range(0, 320, 5):
                if is_blue(img[i,]):
                    print('1', end='')
                else:
                    print('0', end='')
            print()

    @staticmethod
    def which_color_max(img):
        red1, red2, green1, green2, blue1, blue2 = ImageProcessor.color_segmentation(img)
        red_l = red2 - red1
        green_l = green2 - green1
        blue_l = blue2 - blue1

        l = max(red_l, green_l, blue_l)

        if blue_l > 80:
            return 2
        if l == red_l:
            return 0
        if l == green_l:
            return 1
        if l == blue_l:
            return 2

    @staticmethod
    def image_filter(img):
        return cv2.medianBlur(img, 5)

    @staticmethod
    def color_segmentation(img, irow=130):
        red1 = -1
        red2 = -1
        green1 = -1
        green2 = -1
        blue1 = -1
        blue2 = -1
        for i in range(0, 320):
            # is red
            if is_red(img[irow, i]):
                red1 = i
                for j in range(i, 320):
                    # first not red
                    if not is_red(img[irow, j]):
                        red2 = j
                        break
                if red1 > -1 and red2 == -1:
                    red2 = 319
                break
        for i in range(0, 320):
            # is green
            if is_green(img[irow, i]):
                green1 = i
                for j in range(i, 320):
                    # first not green
                    if not is_green(img[irow, j]):
                        green2 = j
                        break
                if green1 > -1 and green2 == -1:
                    green2 = 319
                break
        for i in range(0, 320):
            # is blue
            if is_blue(img[irow, i]):
                blue1 = i
                for j in range(i, 320):
                    # first not blue
                    if not is_blue(img[irow, j]):
                        blue2 = j
                        break
                if blue1 > -1 and blue2 == -1:
                    blue2 = 319
                break
        return red1, red2, green1, green2, blue1, blue2

    @staticmethod
    def which_black_wall_close(img):
        black_l_1 = -1
        black_l_2 = -1
        black_r_1 = -1
        black_r_2 = -1
        for i in range(0, 240):
            point = img[i, 0]
            if point[0] < 50 and point[1] < 50 and point[2] < 50:
                black_l_1 = i
                for j in range(black_l_1, 240):
                    point = img[j, 0]
                    if point[0] > 200 or point[1] > 200 or point[2] > 200:
                        black_l_2 = j
                        break
                break
        for i in range(0, 240):
            point = img[i, 319]
            if point[0] < 50 and point[1] < 50 and point[2] < 50:
                black_r_1 = i
                for j in range(black_r_1, 240):
                    point = img[j, 319]
                    if point[0] > 200 or point[1] > 200 or point[2] > 200:
                        black_r_2 = j
                        break 
                break    
        #print( black_l_1, black_l_2, black_r_1, black_r_2)
        black_l = abs(black_l_2 - black_l_1)
        black_r = abs(black_r_2 - black_r_1)
        if abs(black_r-black_l) < 5 or black_l == 0 or black_r == 0:
            return False, 0, 0
        return True, black_l, black_r

    @staticmethod
    def which_color_and_track(img):
        '''
        color: 
        - 0 red
        - 1 green
        - 2 blue
        - -1 unknown, border
        track:
        -1, 0, 2, 0, 1, 2, 1, -1
        -1, r, b, r, g, b, g, -1
        which track:
        1 2 3 4 5 6
        '''
        red1, red2, green1, green2, blue1, blue2 = ImageProcessor.color_segmentation(img, irow=238)
        # no blue track in camera, red and green split the camera
        if blue1 == -1 and blue2 == -1 and \
            green1 != -1 and green2 != -1 and \
            red1 != -1 and red2 != -1:
            if red2+red1 < green1+green2:
                #  -1, r, b, r, g, b, g, -1, (3rd, red track)
                return -1, 0, 2, 0, 1, 2, 1, -1, 3
            else:
                return -1, 1, 2, 1, 0, 2, 0, -1, 3
        
        # all green color in camera
        if green2 - green1 == 319:
            for i in range(239, 70, -1):
                # right border
                point1 = img[i, 300]
                # bottom to top, first should be green color
                if is_green(point1):
                    continue
                else:
                    # encounter a red color
                    if is_red(point1):
                        return -1, 1, 2, 1, 0, 2, 0, -1, 3
                    # encounter blue color
                    elif is_blue(point1):
                        return -1, 0, 2, 0, 1, 2, 1, -1, 3
        
        # all red color in camera
        if red2 - red1 == 319:
            for i in range(239, 70, -1):
                # right border
                point1 = img[i, 300]
                # bottom to top, first should be red color        
                if is_red(point1):
                    continue
                else:
                    # encounter blue
                    if is_blue(point1):
                        return -1, 1, 2, 1, 0, 2, 0, -1, 3
                    # encounter green
                    elif is_green(point1):
                        return -1, 0, 2, 0, 1, 2, 1, -1, 3   

        # all blue in camera
        if blue2 != -1 or blue1 != -1:
            sure, left_black, right_black = ImageProcessor.which_black_wall_close(img)
            midblue = (blue2+blue1)/2
            midgreen = (green1+green2)/2
            midred = (red1+red2)/2
            if sure == True:     
                if left_black < right_black:
                    # car is close to right border
                    for i in range(239, 70, -1):
                        # left border, 
                        point1 = img[i, 25]
                        # bottom to top, ignore blue color        
                        if is_blue(point1):
                            continue
                        else:
                            # encounter red
                            if is_red(point1):
                                if midred == -1:
                                    return -1, 1, 2, 1, 0, 2, 0, -1, 4
                                if midblue > midred:
                                    # closer to right
                                    return -1, 1, 2, 1, 0, 2, 0, -1, 4
                                else:
                                    # closer to centre
                                    return -1, 1, 2, 1, 0, 2, 0, -1, 5
                            # encounter green
                            elif is_green(point1):
                                if midgreen == -1:
                                    return -1, 0, 2, 0, 1, 2, 1, -1, 4
                                if midblue > midgreen:
                                    # closer to centre
                                    return -1, 0, 2, 0, 1, 2, 1, -1, 4   
                                else:
                                    return -1, 0, 2, 0, 1, 2, 1, -1, 5              
                else:
                    # car is close to left border
                    for i in range(239, 70, -1):
                        # right border, 
                        point1 = img[i, 300]
                        # bottom to top, ignore blue color        
                        if is_blue(point1):
                            continue
                        else:
                            # encounter red
                            if is_red(point1):
                                if midred == -1:
                                    return -1, 0, 2, 0, 1, 2, 1, -1, 3
                                if midblue < midred:
                                    # closer to left
                                    return -1, 0, 2, 0, 1, 2, 1, -1, 3
                                else:
                                    return -1, 0, 2, 0, 1, 2, 1, -1, 2
                            # encounter green
                            elif is_green(point1):
                                if midgreen == -1:
                                    return -1, 1, 2, 1, 0, 2, 0, -1, 3
                                if midblue < midgreen:
                                    return -1, 1, 2, 1, 0, 2, 0, -1, 3  
                                else:
                                    return -1, 1, 2, 1, 0, 2, 0, -1, 2                     
            else:
                # some error?
                return -1, -1, -1, -1, -1, -1, -1, -1, -1 

        # some error?
        return -1, -1, -1, -1, -1, -1, -1, -1, -1


def is_red(point):
    r = point[0]
    g = point[1]
    b = point[2]
    return (r >= 220) & (g < 50) & (b < 50)


def is_green(point):
    r = point[0]
    g = point[1]
    b = point[2]
    return (g >= 220) & (r < 50) & (b < 50)


def is_blue(point):
    r = point[0]
    g = point[1]
    b = point[2]
    return (b >= 220) & (g < 50) & (r < 50)


def is_black(point):
    r = point[0]
    g = point[1]
    b = point[2]
    return (b < 10) & (g < 10) & (r < 10)


def is_yellow(point):
    r = point[0]
    g = point[1]
    b = point[2]    
    return (r >= 128) & (g >= 128) & (b < 100)


def is_black_hsv(point):
    r = point[0]
    g = point[1]
    b = point[2]
    return (b < 10) & (g < 10) & (r < 10)