import cv2
import sys
import math
import numpy as np
import lane

ROI_THRESHOLD=[10, 100, 200]

def _mask(img):
    ga = cv2.GaussianBlur(img, (5,5), 0)
    rgb = lane.flatten(img)
    b, g, r = cv2.split(rgb)
    mask = cv2.threshold(r, 200, 255, cv2.THRESH_BINARY)[1]
    blur = cv2.blur(mask, (5,5))
    mask = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow("mask", mask)
    return mask


def r_mask(img):
    color_low = np.array([10, 10, 120])
    color_high =np.array([70, 60, 200])
    ga = cv2.GaussianBlur(img, (5,5), 0)
    mask = cv2.inRange(ga, color_low, color_high)

    blur = cv2.blur(mask, (5,5))
    mask = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[1]
    return mask

def draw_box(img, locs):
    print("draw box locs:", locs)
    max_x = locs[0]
    max_y = locs[1]
    min_x = locs[2]
    min_y = locs[3]

    if max_x < 0 or min_x < 0 or max_y < 0 or min_y < 0:
        return

    img = cv2.rectangle(img, (max_x, max_y), (min_x, min_y), (0, 255, 0), 1)
    cv2.imshow("box", img)
#    cv2.waitKey(1)

def get_rectangle_locs(contour):
    h, w, l = contour.shape
    locs = contour.reshape((h, l))

    x_locs = locs[0:h, 0]
    y_locs = locs[0:h, 1]

    max_x = np.max(x_locs)
    max_y = np.max(y_locs)
    min_x = np.min(x_locs)
    min_y = np.min(y_locs)

    return np.array([[max_x, max_y], [min_x, min_y]])

def locs_distance(loc1, loc2):
    d = loc1 - loc2
    d = d * d
    d = math.sqrt(np.sum(d))
    return d

def locs_filter(mask, locs):
    h, w = mask.shape[:2]

    max_x = locs[0]
    max_y = locs[1]
    min_x = locs[2]
    min_y = locs[3]

    xd = locs[0] - locs[2]
    yd = locs[1] - locs[3]

#    print("height/3:", h/3, "weight/3:", h/3)
#    print("xd:", xd, "yd:", yd)
    if xd > h/3 or xd > w/3 or xd < 6 or yd < 6:
        return [-1, -1, -1, -1]

    ratio = 0.2    

    xd = max_x - min_x
    yd = max_y - min_y
    
    max_x = min(max_x + int(ratio*xd), h)
    if min_x - int(ratio*xd) > 0:
        min_x = min_x - int(ratio*xd)
    else:
        min_x = 0

    max_y = min(max_y + int(ratio*yd), w)
    if min_y - int(ratio*yd) > 0:
        min_y = min_y - int(ratio*yd)
    else:
        min_y = 0

    return locs

def detect(img, sen = 0):
    mask = _mask(img)
    binary, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    sum = 0
    if len(contours) < 1:
        return False, mask

    for i in range(len(contours)):
        sum += cv2.contourArea(contours[i])
    
    nums = np.sum(mask != 0) 
    #print(">>> ROI area:", sum)
    return sum >= ROI_THRESHOLD[sen], mask

def location(mask):
    h, w = mask.shape[:2]
    
    mask_locs = np.array([[0,0], [0,0]])
    mask_locs1 = np.array([[h,w],[h,w]])
    
    diagonal = locs_distance(mask_locs,mask_locs1)
    binary, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    num = len(contours)
    #print("len contours:", len(contours))

    if num == 0:
        return [-1, -1, -1, -1]
    elif num == 1:
        locs =  get_rectangle_locs(contours[0])
        return locs_filter(mask, [locs[0,0], locs[0,1], locs[1,0], locs[1,1]])
    
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i])) 

    area_copy = area[:]
    max_id = np.argmax(area_copy)

    locs0 = get_rectangle_locs(contours[max_id])

    dist = []
    for i in range(len(area)):
        locs = get_rectangle_locs(contours[i])
        dist.append(locs_distance(locs0, locs))
    
    dist_copy = dist[:]
    del dist_copy[max_id]
    d = min(dist_copy)

    if d > diagonal/8:
        return locs_filter(mask, [locs[0,0], locs[0,1], locs[1,0], locs[1,1]])

    locs1 = get_rectangle_locs(contours[dist.index(d)])
    locs = np.concatenate((locs0, locs1), axis=0)
    x_locs = locs[:, 0]
    y_locs = locs[:, 1]

    max_x = np.max(x_locs)
    max_y = np.max(y_locs)
    min_x = np.min(x_locs)
    min_y = np.min(y_locs)

    #print("upper point:", [max_x, max_y])
    #print("down point:", [min_x, min_y]) 
    return locs_filter(mask,[max_x, max_y, min_x, min_y])

def debug_draw_box(img):
    detected, mask = detect(img)
    print("contains sign ROI, need recognize?", detected)

    if not detected:
        return

    locs = location(mask)
    draw_box(img, locs)

if __name__ == '__main__':

    filename = sys.argv[1]
    img = cv2.imread(filename)
    cv2.imshow("original", img)

    detected, mask = detect(img)
    print("contains sign ROI, need recognize?", detected)

    if not detected:
        exit()

    locs = location(mask)
    draw_box(img, locs)
    cv2.waitKey(60000)
