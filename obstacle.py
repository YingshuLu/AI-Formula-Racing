import cv2
import numpy as np
import sys
import math

OBSTACLE_THRESHOLD=250
OBSTACLE_LOCATION=["NONE", "left", "middle", "right"]

def mask_override(mask_list):
    mask = mask_list[0]
    for i in range(1, len(mask_list)):
        mask = cv2.bitwise_and(mask, mask_list[i])
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

# -1: no wall
# 0: wall on left
# 1: wall on right
def wall_location(img):
    h, w = img.shape[:2]
    mask = track_umask(img)[int(h*0.65):,:]

    mask = cv2.blur(mask, (5,5))
    mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
    sum_wall = np.sum(mask > 0)
    if not sum_wall > 10:
        return -1

    base = int(w/2)
    value = []
    value.append(np.sum(mask[:, 0:base] > 0))
    value.append(np.sum(mask[:, base+1:] > 0))
    l = value.index(max(value))
    return l

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

def rgb_mask(img):
    cb, cg, cr = cv2.split(img)
    
    tbg = np.absolute(cb - cg)
    tbr = np.absolute(cb - cr)
    trg = np.absolute(cr - cg)

    color_low = np.array([50, 50, 50])
    color_high = np.array([55, 55, 55])

    cmask = cv2.inRange(img, color_low, color_high)
    mask = cv2.threshold(np.absolute(tbg - tbr), 8, 255, cv2.THRESH_BINARY)[1]
    mask1 = cv2.threshold(np.absolute(tbg - trg), 8, 255, cv2.THRESH_BINARY)[1]

    mask = cv2.bitwise_and(cmask, mask)
    mask = cv2.bitwise_and(mask1, mask)

    cb, cg, cr = cv2.split(img)
    min_c = (np.minimum(np.minimum(cb, cg),cr))
    max_c = (np.maximum(np.maximum(cb, cg),cr))
    mmask = cv2.threshold(max_c - min_c, 16, 255, cv2.THRESH_BINARY)[1]

    #mask = cv2.bitwise_and(mask, mmask)
    mask = cmask
    return mask

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
   

# -1: no obstacle
# 0: obstacle on left
# 1 : obstacle on middle
# 2 : obstacle on right
def detect(img):
    tmask = track_umask(img)
    #cmask = rgb_mask(im)
    hmask = hsv_mask(img)
    mask = cv2.bitwise_and(tmask, hmask)

    if np.sum(mask > 0) < OBSTACLE_THRESHOLD:
        return -1

    height, width = mask.shape[:2]
    mask = mask[int(math.floor(height*2/5 + 1)): height, :]
    base = int(width/3)
    value = []
    value.append(np.sum(mask[:, 0:base] > 0))
    value.append(np.sum(mask[:, base+1 : 2*base] > 0))
    value.append(np.sum(mask[:, 2*base+1 : -1] > 0))
    
    l = value.index(max(value))
    return l

def notice_obstacle(im):
    blur = cv2.GaussianBlur(im, (5,5), 0)
    blur = cv2.GaussianBlur(blur, (5,5), 0)
    blur = cv2.GaussianBlur(blur, (5,5), 0)

    hsv_low = np.array([85, 0, 50])
    hsv_high = np.array([140, 65, 255])
    height, width = im.shape[:2]


    bot = blur[int(math.floor(height*2/5))+1:-1, 0:width]
    hsv = cv2.cvtColor(bot, cv2.COLOR_RGB2HSV)
    hmask = cv2.inRange(hsv, hsv_low, hsv_high)

    cv2.imshow("hsv", hsv)
    color_low = np.array([50, 50, 50])
    color_high = np.array([120, 120, 120])

    cmask = cv2.inRange(bot, color_low, color_high)
    mask = cv2.bitwise_and(hmask, cmask)
#    mask = cv2.GaussianBlur(mask, (5,5), 0)
#    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if np.sum(mask > 0) < OBSTACLE_THRESHOLD:
        return -1

    cv2.imshow("cmask", cmask)
    cv2.imshow("hmask", hmask)
    cv2.imshow("mask", mask)

    base = int(width/3)
    value = []
    value.append(np.sum(mask[:, 0:base] > 0))
    value.append(np.sum(mask[:, base+1 : 2*base] > 0))
    value.append(np.sum(mask[:, 2*base+1 : -1] > 0))
    
    l = value.index(max(value))
    return l


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        exit(0)

    file_name = sys.argv[1]

    im = cv2.imread(file_name)
    cv2.imshow("image", im)

    print("obstacle: ", detect(im))
    wall_location(im)
    cv2.waitKey(6000)
