# coding=utf-8
from time import time
from PIL  import Image
from io   import BytesIO
from datetime import datetime
import os
import cv2
import math
import numpy as np
import base64
import logger
import Settings
import Globals
from imutils import contours
from skimage import measure
#from ImageProcessor import ImageProcessor
import argparse
import imutils
from collections import Counter

logger = logger.get_logger(__name__)

def logit(msg):
    pass
    #logger.info("%s" % msg)

class ImageProcessor(object):
    @staticmethod
    def show_image(img, name = "image", scale = 1.0, newsize = None):
        if scale and scale != 1.0:
            img = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC) 

        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, img)
        cv2.waitKey(1)


    @staticmethod
    def save_image(img, prefix = "img", suffix = ""):
        if Globals.RecordFolder is None:
            return
        
        filename = "%s-%s-%s.jpg" % (datetime.now().strftime('%Y%m%d-%H%M%S-%f'), prefix, suffix)
        cv2.imwrite(os.path.join(Globals.RecordFolder, filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    @staticmethod
    def force_save_image_to_log_folder(img, prefix = "img", suffix = ""):
        filename = "%s-%s-%s.jpg" % (datetime.now().strftime('%Y%m%d-%H%M%S-%f'), prefix, suffix)
        cv2.imwrite(os.path.join('/log/', filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    @staticmethod
    def force_save_bmp_to_log_folder(img, suffix = ""):
        filename = "%s-%s.bmp" % (datetime.now().strftime('%Y%m%d-%H%M%S-%f'), suffix)
        cv2.imwrite(os.path.join('/log/', filename), img)

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
    def _normalize_brightness(img):
        maximum = img.max()
        if maximum == 0:
            return img
        adjustment = min(255.0/img.max(), 3.0)
        normalized = np.clip(img * adjustment, 0, 255)
        normalized = np.array(normalized, dtype=np.uint8)
        return normalized


    @staticmethod
    def _flatten_rgb(img):
        r, g, b = cv2.split(img)
        r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
        g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
        b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
        w_filter = ((r >= 128) & (g >= 128) & (b >= 128))

        b[b_filter], b[np.invert(b_filter)] = 255, 0
        r[r_filter], r[np.invert(r_filter)] = 255, 0
        #b[r_filter], g[r_filter] = 0, 0
        g[g_filter], g[np.invert(g_filter)] = 255, 0

        r[w_filter], g[w_filter],b[w_filter] = 255, 255, 255
        flattened = cv2.merge((r, g, b))
        return flattened
    @staticmethod
    def _flatten_rgb_old(img):
        r, g, b = cv2.split(img)
        r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 100) & (g < 180) & (b < 180)
        g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 100) & (r < 180) & (b < 180)
        b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 100) & (r < 180) & (g < 180)
        y_filter = ((r >= 128) & (g >= 128) & (b < 100))

        r[y_filter], g[y_filter] = 255, 255
        b[np.invert(y_filter)] = 0

        b[b_filter], b[np.invert(b_filter)] = 255, 0
        r[r_filter], r[np.invert(r_filter)] = 255, 0
        g[g_filter], g[np.invert(g_filter)] = 255, 0

        flattened = cv2.merge((r, g, b))
        return flattened

    @staticmethod
    def _crop_image(img, down_cut_ratio):
        bottom_half_ratios = (down_cut_ratio, 1.0)
        bottom_half_slice  = slice(*(int(x * img.shape[0]) for x in bottom_half_ratios))
        bottom_half        = img[bottom_half_slice, :, :]
        return bottom_half

    @staticmethod
    def preprocess(img, down_cut_ratio = 0.65):
        img = ImageProcessor._crop_image(img, down_cut_ratio)
        #img = ImageProcessor._normalize_brightness(img)
        img = ImageProcessor._flatten_rgb(img)
        return img

    @staticmethod
    def find_lines(img):
        grayed      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred     = cv2.GaussianBlur(grayed, (3, 3), 0)
        #edged      = cv2.Canny(blurred, 0, 150)

        sobel_x     = cv2.Sobel(blurred, cv2.CV_16S, 1, 0)
        sobel_y     = cv2.Sobel(blurred, cv2.CV_16S, 0, 1)
        sobel_abs_x = cv2.convertScaleAbs(sobel_x)
        sobel_abs_y = cv2.convertScaleAbs(sobel_y)
        edged       = cv2.addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0)

        lines       = cv2.HoughLinesP(edged, 1, np.pi / 180, 10, 5, 5)
        return lines


    @staticmethod
    def _find_best_matched_line(thetaA0, thetaB0, tolerance, vectors, matched = None, start_index = 0):
        if matched is not None:
            matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
            matched_angle = abs(np.pi/2 - matched_thetaB)

        for i in xrange(start_index, len(vectors)):
            distance, length, thetaA, thetaB, coord = vectors[i]

            if (thetaA0 is None or abs(thetaA - thetaA0) <= tolerance) and \
               (thetaB0 is None or abs(thetaB - thetaB0) <= tolerance):
                
                if matched is None:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue

                heading_angle = abs(np.pi/2 - thetaB)

                if heading_angle > matched_angle:
                    continue
                if heading_angle < matched_angle:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue
                if distance < matched_distance:
                    continue
                if distance > matched_distance:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue
                if length < matched_length:
                    continue
                if length > matched_length:
                    matched = vectors[i]
                    matched_distance, matched_length, matched_thetaA, matched_thetaB, matched_coord = matched
                    matched_angle = abs(np.pi/2 - matched_thetaB)
                    continue

        return matched

    @staticmethod
    def find_steering_angle_by_line(img, last_steering_angle, debug = True):        
        steering_angle = 0.0
        lines          = ImageProcessor.find_lines(img)

        if lines is None:
            return steering_angle

        image_height = img.shape[0]
        image_width  = img.shape[1]
        camera_x     = image_width / 2
        camera_y     = image_height
        vectors      = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                thetaA   = math.atan2(abs(y2 - y1), (x2 - x1))
                thetaB1  = math.atan2(abs(y1 - camera_y), (x1 - camera_x))
                thetaB2  = math.atan2(abs(y2 - camera_y), (x2 - camera_x))
                thetaB   = thetaB1 if abs(np.pi/2 - thetaB1) < abs(np.pi/2 - thetaB2) else thetaB2

                length   = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distance = min(math.sqrt((x1 - camera_x) ** 2 + (y1 - camera_y) ** 2),
                               math.sqrt((x2 - camera_x) ** 2 + (y2 - camera_y) ** 2))

                vectors.append((distance, length, thetaA, thetaB, (x1, y1, x2, y2)))

                if debug:
                    # draw the edges
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

        #the line of the shortest distance and longer length will be the first choice
        vectors.sort(lambda a, b: cmp(a[0], b[0]) if a[0] != b[0] else -cmp(a[1], b[1]))

        best = vectors[0]
        best_distance, best_length, best_thetaA, best_thetaB, best_coord = best
        tolerance = np.pi / 180.0 * 10.0

        best = ImageProcessor._find_best_matched_line(best_thetaA, None, tolerance, vectors, matched = best, start_index = 1)
        best_distance, best_length, best_thetaA, best_thetaB, best_coord = best

        if debug:
            #draw the best line
            cv2.line(img, best_coord[:2], best_coord[2:], (0, 255, 255), 2)

        if abs(best_thetaB - np.pi/2) <= tolerance and abs(best_thetaA - best_thetaB) >= np.pi/4:
            #print "*** sharp turning"
            best_x1, best_y1, best_x2, best_y2 = best_coord
            f = lambda x: int(((float(best_y2) - float(best_y1)) / (float(best_x2) - float(best_x1)) * (x - float(best_x1))) + float(best_y1))
            left_x , left_y  = 0, f(0)
            right_x, right_y = image_width - 1, f(image_width - 1)

            if left_y < right_y:
                best_thetaC = math.atan2(abs(left_y - camera_y), (left_x - camera_x))

                if debug:
                    #draw the last possible line
                    cv2.line(img, (left_x, left_y), (camera_x, camera_y), (255, 128, 128), 2)
                    cv2.line(img, (left_x, left_y), (best_x1, best_y1), (255, 128, 128), 2)
            else:
                best_thetaC = math.atan2(abs(right_y - camera_y), (right_x - camera_x))

                if debug:
                    #draw the last possible line
                    cv2.line(img, (right_x, right_y), (camera_x, camera_y), (255, 128, 128), 2)
                    cv2.line(img, (right_x, right_y), (best_x1, best_y1), (255, 128, 128), 2)

            steering_angle = best_thetaC
        else:
            steering_angle = best_thetaB

        if (steering_angle - np.pi/2) * (last_steering_angle - np.pi/2) < 0:
            last = ImageProcessor._find_best_matched_line(None, last_steering_angle, tolerance, vectors)

            if last:
                last_distance, last_length, last_thetaA, last_thetaB, last_coord = last
                steering_angle = last_thetaB

                if debug:
                    #draw the last possible line
                    cv2.line(img, last_coord[:2], last_coord[2:], (255, 128, 128), 2)
        steering_angle = steering_angle
        if debug:
            #draw the steering direction
            r = 60
            x = image_width / 2 + int(r * math.cos(steering_angle))
            y = image_height    - int(r * math.sin(steering_angle))
            cv2.line(img, (image_width / 2, image_height), (x, y), (255, 0, 255), 5)
            #logit("line angle: %0.2f, steering angle: %0.2f, last steering angle: %0.2f" % (ImageProcessor.rad2deg(best_thetaA), ImageProcessor.rad2deg(np.pi/2-steering_angle), ImageProcessor.rad2deg(np.pi/2-last_steering_angle)))

        return (np.pi/2 - steering_angle)

    # @staticmethod
    # def find_steering_angle_by_line(img, last_steering_angle, debug = True):
    #     steering_angle = 0.0
    #     lines          = ImageProcessor.find_lines(img)

    #     if lines is None:
    #         return steering_angle

    #     image_height = img.shape[0]
    #     image_width  = img.shape[1]
    #     camera_x     = image_width / 2
    #     camera_y     = image_height
    #     vectors      = []

    #     for line in lines:
    #         for x1, y1, x2, y2 in line:
    #             thetaA   = math.atan2(abs(y2 - y1), (x2 - x1))
    #             thetaB1  = math.atan2(abs(y1 - camera_y), (x1 - camera_x))
    #             thetaB2  = math.atan2(abs(y2 - camera_y), (x2 - camera_x))
    #             thetaB   = thetaB1 if abs(np.pi/2 - thetaB1) < abs(np.pi/2 - thetaB2) else thetaB2

    #             length   = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    #             distance = min(math.sqrt((x1 - camera_x) ** 2 + (y1 - camera_y) ** 2),
    #                            math.sqrt((x2 - camera_x) ** 2 + (y2 - camera_y) ** 2))

    #             vectors.append((distance, length, thetaA, thetaB, (x1, y1, x2, y2)))

    #             if debug:
    #                 # draw the edges
    #                 cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

    #     #the line of the shortest distance and longer length will be the first choice
    #     vectors.sort(lambda a, b: cmp(a[0], b[0]) if a[0] != b[0] else -cmp(a[1], b[1]))

    #     best = vectors[0]
    #     best_distance, best_length, best_thetaA, best_thetaB, best_coord = best
    #     tolerance = np.pi / 180.0 * 10.0

    #     best = ImageProcessor._find_best_matched_line(best_thetaA, None, tolerance, vectors, matched = best, start_index = 1)
    #     best_distance, best_length, best_thetaA, best_thetaB, best_coord = best

    #     if debug:
    #         #draw the best line
    #         cv2.line(img, best_coord[:2], best_coord[2:], (0, 255, 255), 2)

    #     if abs(best_thetaB - np.pi/2) <= tolerance and abs(best_thetaA - best_thetaB) >= np.pi/4:
    #         #print "*** sharp turning"
    #         best_x1, best_y1, best_x2, best_y2 = best_coord
    #         f = lambda x: int(((float(best_y2) - float(best_y1)) / (float(best_x2) - float(best_x1)) * (x - float(best_x1))) + float(best_y1))
    #         left_x , left_y  = 0, f(0)
    #         right_x, right_y = image_width - 1, f(image_width - 1)

    #         if left_y < right_y:
    #             best_thetaC = math.atan2(abs(left_y - camera_y), (left_x - camera_x))

    #             if debug:
    #                 #draw the last possible line
    #                 cv2.line(img, (left_x, left_y), (camera_x, camera_y), (255, 128, 128), 2)
    #                 cv2.line(img, (left_x, left_y), (best_x1, best_y1), (255, 128, 128), 2)
    #         else:
    #             best_thetaC = math.atan2(abs(right_y - camera_y), (right_x - camera_x))

    #             if debug:
    #                 #draw the last possible line
    #                 cv2.line(img, (right_x, right_y), (camera_x, camera_y), (255, 128, 128), 2)
    #                 cv2.line(img, (right_x, right_y), (best_x1, best_y1), (255, 128, 128), 2)

    #         steering_angle = best_thetaC
    #     else:
    #         steering_angle = best_thetaB

    #     if (steering_angle - np.pi/2) * (last_steering_angle - np.pi/2) < 0:
    #         last = ImageProcessor._find_best_matched_line(None, last_steering_angle, tolerance, vectors)

    #         if last:
    #             last_distance, last_length, last_thetaA, last_thetaB, last_coord = last
    #             steering_angle = last_thetaB

    #             if debug:
    #                 #draw the last possible line
    #                 cv2.line(img, last_coord[:2], last_coord[2:], (255, 128, 128), 2)

    #     if debug:
    #         #draw the steering direction
    #         r = 60
    #         x = image_width / 2 + int(r * math.cos(steering_angle))
    #         y = image_height    - int(r * math.sin(steering_angle))
    #         cv2.line(img, (image_width / 2, image_height), (x, y), (255, 0, 255), 2)
    #         #logit("line angle: %0.2f, steering angle: %0.2f, last steering angle: %0.2f" % (ImageProcessor.rad2deg(best_thetaA), ImageProcessor.rad2deg(np.pi/2-steering_angle), ImageProcessor.rad2deg(np.pi/2-last_steering_angle)))

    #     return (np.pi/2 - steering_angle)


    @staticmethod
    def find_steering_angle_by_color(img, last_steering_angle, debug = True):
        r, g, b      = cv2.split(img)
        image_height = img.shape[0]
        image_width  = img.shape[1]
        camera_x     = image_width / 2
        image_sample = slice(0, int(image_height * 0.2))
        sr, sg, sb   = r[image_sample, :], g[image_sample, :], b[image_sample, :]
        track_list   = [sr, sg, sb]
        tracks       = map(lambda x: len(x[x > 20]), [sr, sg, sb])
        tracks_seen  = filter(lambda y: y > 50, tracks)

        if len(tracks_seen) == 0:
            return 0.0

        maximum_color_idx = np.argmax(tracks, axis=None)
        _target = track_list[maximum_color_idx]
        _y, _x = np.where(_target == 255)
        px = np.mean(_x)
        steering_angle = math.atan2(image_height, (px - camera_x))

        if debug:
            #draw the steering direction
            r = 60
            x = image_width / 2 + int(r * math.cos(steering_angle))
            y = image_height    - int(r * math.sin(steering_angle))
            cv2.line(img, (image_width / 2, image_height), (x, y), (255, 0, 255), 2)
            #logit("steering angle: %0.2f, last steering angle: %0.2f" % (ImageProcessor.rad2deg(steering_angle), ImageProcessor.rad2deg(np.pi/2-last_steering_angle)))

        return (np.pi/2 - steering_angle) * 2.0


class ImageLines:
    def __init__(self, image):
        self._srcimage = image
        self._image = image

        self._thresh_dilate = []
        self._thresh_erode = []
        self._thresh = []
        self._kernel = (3,1)
        self._thresh = []

    def ImageBasicProcess(self):
        #image = cv2.imread('./IMG/front-t5-2018_10_09_14_00_28_487.jpg')
        image = ImageProcessor._crop_image(self._srcimage, 0.5)
        self._image = image
        #image = self._image

        img_hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # lower mask (0-10)
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([170,50,50])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
        
        # join my masks
        mask = mask0+mask1

        # set my output img to zero everywhere except my mask
        output_img = image.copy()
        #output_img[np.where(mask==0)] = 0
        output_img = cv2.bitwise_and(output_img, output_img, mask= mask)
        #cv2.imshow("output_img", np.hstack([img, output_img]))
        
        gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray", gray)
        
        blurred = cv2.GaussianBlur(gray, self._kernel, 0)
        #cv2.imshow("blurred", blurred)
        
        # threshold the image to reveal light regions in the
        # blurred image
        thresh = cv2.threshold(blurred, 4, 255, cv2.THRESH_BINARY)[1]
        self._thresh = thresh
        #cv2.imshow("thresh", thresh)
        
        self._thresh_erode = cv2.erode(thresh, self._kernel, iterations=1)
        
        self._thresh_dilate = cv2.dilate(self._thresh_erode, self._kernel, iterations=1)
        # if Settings.DEBUG_IMG:
        #     cv2.imshow("thresh2", self._thresh_dilate)

    def GetAllEdgeLine(self):
        if np.size(self._thresh_dilate) != 0:
            # thresh1 = cv2.erode(self._thresh_dilate, self._kernel, iterations=1)
            # if Settings.DEBUG_IMG:
            #     cv2.imshow("thresh1", thresh1)
            
            #thresh_dilate = cv2.dilate(thresh1, (1, 1), iterations=1)

            edgelines = np.bitwise_xor(self._thresh, self._thresh_dilate)
            indexs = np.transpose(edgelines.nonzero())
            
            # if Settings.DEBUG_IMG:
            #     cv2.imshow("edgelines", edgelines)

            return indexs[:indexs.shape[0], :]
        else:
            return None

    def GetMidlleLine(self, yaxis = 10):
        # perform a connected component analysis on the thresholded
        # image, then initialize a mask to store only the "large"
        # components
        image = self._image
        thresh = self._thresh_dilate
        labels = measure.label(thresh, neighbors=4, background=0)

        mask = np.zeros(thresh.shape, dtype="uint8")
        sharpcount = 0
        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue
        
            # otherwise, construct the label mask and count the
            # number of pixels 
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            #cv2.bitwise_and(labels, label, mask= labelMask)
            numPixels = cv2.countNonZero(labelMask)
            #print ("numPixels: %s" % numPixels)
            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            if numPixels > 100:
                mask = cv2.add(mask, labelMask)
                sharpcount +=1
        
        # find the contours in the mask, then sort them from left to
        # right
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
       
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        candidatecount = len(cnts)
 
        cdpos, cdcnt, cdxy = -1, [], [0, 0]

        #print "cant count: %s" % candidatecount
        if candidatecount < 2:
            return cdpos, cdcnt, cdxy, sharpcount
 
        cnts = contours.sort_contours(cnts)[0]
       
        # loop over the contours
        candidatestd = [[1, 0.2, 1.8],
                    [2, 0.4, 4.5],
                    [3, 0.6, 5.5],
                    [4, 0.8, 12.5],
                    [5, 3, 13],
                    [6, 6, 13.5]]
      
        mapc = Counter()
        for (i, cnt) in enumerate(cnts):
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,False)
            
            #(cX, cY), radius) = cv2.minEnclosingCircle(cnt)
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"]) if M["m00"] !=0 else 0
            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
            width = area/perimeter if perimeter != 0 else 0
            
            pos = int(cY / 20)
            
            if area > 60:
                mapc[cX] = cY, pos, cnt

            #print ("index[%s] : %sth (x,y: %s %s) contours [area: %s] [perimeter: %s] [width: %s]" %(pos,i+1, cX, cY, area, perimeter, width))

            if (pos >= 5 and candidatecount >= 3 and width > 13) or width > 15 or (pos >= 5 and area < 3000):
                #print "should not be middle line"
                continue
            
            if width >= candidatestd[pos][1] and width <= candidatestd[pos][2]:            
                #print ("matched middleline: %sth (x,y: %s %s) contours [area: %s] [perimeter: %s] [width: %s]" %(i+1, cX, cY, area, perimeter, width))
                cdcnt = cnt
                cdpos = pos
        
        mapcount = len(mapc)
        if mapcount >= 2:
            keylist = sorted(mapc.keys())
            if mapcount >= 3:
                cY, cdpos, cdcnt = mapc[keylist[1]]
                #print "map>=3 (%s, %s)"%(cY, cdpos)
            elif mapcount == 2:
                cY, pos, cnt = mapc[keylist[0]]
                cY1, pos1, cnt1 = mapc[keylist[1]]
                if cY - cY1 > 0:
                    cdcnt = cnt
                    cdpos = pos
                else:
                    cdcnt = cnt1
                    cdpos = pos1
                #print "map == 2 (%s-%s, %s)"%(cY,cY1, cdpos)

        if cdpos != -1:
            npcnt = cdcnt.transpose(0, 1, 2).reshape(-1, 2)
            rows = np.where(npcnt[:, 1] < yaxis)
            #print "npcnt: %s " % npcnt
            indexs1 = npcnt[rows]
            #print "indexs1: %s " % indexs1
            xy = np.mean(indexs1, axis = 0) if len(indexs1) > 0 else [0, 0]
            xy = [int(xy[0]), int(xy[1])]
            #print "xy: %s " % xy

            # if Settings.DEBUG_IMG:
            #     # cv2.putText(image, "#{}".format(1), (xy[0], xy[1]),
            #     # cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 3)
            #     cv2.circle(image,(xy[0], xy[1]),4,(0,255,0),2)
            #     cv2.imshow("Image", image)
            
            cdxy = xy
            
        return cdpos, cdcnt, cdxy, sharpcount