import sys
import cv2
import numpy as np
import math
import util
import rsign
import time
import os
import mark_svm as svm
import bird

def wall_mask(src_img):
    src_img = cv2.blur(src_img, (5,5))
    b,g,r = cv2.split(src_img)
    b[b < 13] = 0
    g[g < 13] = 0
    r[r < 13] = 0
    src_img = cv2.merge([b,g,r])

    bw_low = np.array([0,0,0])
    bw_high = np.array([7,7,7])
    wmask = cv2.inRange(src_img, bw_low, bw_high)
    return wmask

def match(cn1, cn2):
    return cn1[1] > 100 and cn2[1] > 100 and math.fabs(cn1[0] - cn2[0]) < 5 and math.fabs(cn1[1] - cn2[1]) < 25 

def right_angle(corners):
    invalid = np.array([[-1, -1], [-1, -1]])
    if len(corners) < 3:
        return invalid

    cn0 = corners[0][0].tolist()
    cn1 = corners[1][0].tolist()
    cn2 = corners[2][0].tolist()

    right =invalid
    r1 = []
    r2 = []
    if match(cn0,cn1):
        r1 = cn0
        r2 = cn1
    elif match(cn0,cn2):
        r1 = cn0
        r2 = cn2
    elif match(cn1,cn2):
        r1 = cn1
        r2 = cn2

    if len(r1) > 0 and len(r2) > 0:
        right = np.array([r1, r2])

    return right

def detect_corners(mask, corners = 3):
    mask = cv2.blur(mask, (5,5))
    cns = cv2.goodFeaturesToTrack(mask, corners, 0.1, 5)
    if not cns is None and len(cns) > 0:
        cns = np.int0(cns)
    else:
        cns = np.array([])
    return cns

def detect_cross(src_img):
    wmask = wall_mask(src_img)
    cns = detect_corners(wmask)
    cns = right_angle(cns)
    cns = cns.astype(np.int32)
    cn = (np.sum(cns, axis = 0) / 2)
    x = int(cn[0])
    y = int(cn[1])
    #if x > 0:
    #    cv2.circle(src_img, (x,y), 4, (0,0,255), -1)
    return [x, y]
    
def draw_corners(src_img):
    wmask = wall_mask(src_img)
   # cv2.imshow("wall mask", wmask)
    cns = detect_corners(wmask)
    cns = right_angle(cns)
    for cn in cns:
        x,y = cn.ravel()
        if x < 0:
            continue
        cv2.circle(src_img, (x,y), 4, (0,0,255), -1)
    

def lanes_split(img):

    h,w = img.shape[:2]
    (b,g,r) = cv2.split(img)

    white = (b > 200) & (g > 200) & (r > 200)
    red = (b < 200) & (g < 200) & (r > 200)

    b[white] = 0
    g[white] = 255
    r[white] = 0

    b[red] = 0
    g[red] = 0
    r[red] = 255 

    other = ~(white|red)
    b[other] = 0
    g[other] = 0
    r[other] = 0

    img = cv2.merge([b, g, r])
    #cv2.imshow("process", img)
    return img
    
def point_to_line_distance(line, point):
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[1][0]
    y2 = line[1][1]

    line_start_x = point[0]
    line_start_y = point[1]
    
    array_longi  = np.array([x2-x1, y2-y1])
    array_trans = np.array([x2-line_start_x, y2-line_start_y])
    array_temp = (float(array_trans.dot(array_longi)) / array_longi.dot(array_longi))
    array_temp = array_longi.dot(array_temp)
    distance   = np.sqrt((array_trans - array_temp).dot(array_trans - array_temp))

    return distance
    
def get_convex_point(cut_line, contour):
    distances = []
    for point in contour:
        dist = point_to_line_distance(cut_line, point[0]) 
        distances.append(dist)

    dist_array = np.asarray(distances)
    max_id = np.argmax(dist_array)
    idx = np.argsort(-dist_array)

    cnt = 5
    if cnt > len(contour):
        cnt = len(contour)

    if dist_array[max_id] < 5:
        return cut_line[1,:]
        
    points = np.asarray(contour[idx[0:cnt]]) 
    point = np.sum(points, axis= 0) / (points.shape[0])
    return point[0] 
    
def get_cut_points(contour):
    h, w, l = contour.shape
    locs = contour.reshape((h, l))

    row_locs = locs[0:h, 1]

    min_y = np.min(row_locs)
    max_y = np.max(row_locs)

    min_locs = locs[ locs[:, 1] == min_y, :]
    max_locs = locs[locs[:,1] == max_y, :] 
    min_loc = (np.sum(min_locs, axis=0) / min_locs.shape[0])
    max_loc = (np.sum(max_locs, axis=0) / max_locs.shape[0])

    return [min_loc, max_loc]

def draw_line(blank, line, color):
    cv2.line(blank, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), color, 2)

def mark_mask(src_img):
    h,w = src_img.shape[:2]
    img = src_img[int(h/2):, :]

    ety = np.zeros(img.shape[:2], np.uint8)
    rmask = util.flatten_red(img)
    #cv2.imshow("red mask", rmask)
    masks = lanes_split(img)
    (b, wmask, r) = cv2.split(img)

    wmask = cv2.threshold(wmask, 127, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow("white mask", wmask)

    binary, contours, hierarchy = cv2.findContours(rmask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i])) 

    if len(area) == 0:
        return [False, ety]

    idx = np.argmax(area)
    contour = contours[idx]
    cv2.drawContours(img, [contours[idx]], -1, (255,0,0), 3)

    blank = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(blank, pts =[contour], color=(255,255,255))
    #cv2.imshow("contour mask", blank)
    cmask = cv2.split(blank)[0]
 
    smask = cv2.bitwise_and(cmask, wmask)
    smask = cv2.blur(smask, (5,5))
    smask = cv2.threshold(smask, 127, 255, cv2.THRESH_BINARY)[1]

   # cv2.imshow("smask", smask)
    blur = cv2.blur(smask, (5,5))
    smask = cv2.addWeighted(smask, 1.5, blur, -0.5, 0)

    num = np.sum(smask > 0)
    #print("smask num:", num)

    if num < 200:
        return [False, ety]
    #cv2.imshow("sign mask", smask)

    return [True, smask]

def get_mark(smask):
    binary, contours, hierarchy = cv2.findContours(smask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i])) 

    if len(area) == 0:
        return -1
    idx = np.argmax(area)
    contour = contours[idx]
    locs = rsign.get_rectangle_locs(contour)

    max_x = locs[0][0]
    max_y = locs[0][1]
    min_x = locs[1][0]
    min_y = locs[1][1]

    mark = smask[min_y:max_y, min_x:max_x]
   # cv2.imshow("mark",mark)
    return mark

RED=1
WHITE=2
WHITE_LANE=WHITE
RED_LANE = 1
RED_LINE =0

class Contour:
    def __init__(self, lanes, cont):
        self.lanes = lanes
        self.cont  = cont
        self.cmask = lanes.contour_mask(cont)
        self.bird_cmask = None
        self.min_loc = None
        self.ctype = None

    def min_location(self):
        if self.min_loc is not None:
            return self.min_loc

        cmask = self.cmask
        
        bcmask = bird.view(cmask, 1)
        locs = np.where(bcmask > 0)
        idx = np.argmin(locs[1])
        min_x = locs[1][idx]
        min_y = locs[0][idx]
        self.min_loc = (min_x, min_y)
        return self.min_loc

    def min_lo(self):
        if self.min_loc is not None:
            return self.min_loc

        cmask = self.mask()
        locs = np.where(cmask > 0)

        idx = np.argmin(locs[1])
        min_x = locs[1][idx]
        min_y = locs[0][idx]

        self.min_loc = (min_x, min_y)
        return self.min_loc

    def mask(self):
        return self.cmask

    def bird_mask(self):
        if self.bird_cmask is None:
            cmask = self.mask()
            cmask = cv2.blur(cmask, (5,5))
            self.bird_cmask = bird.view(cmask, 1)
        return self.bird_cmask

    def bird_area(self):
        return np.sum(self.bird_mask() > 0)
    
    def area(self):
        return np.sum(self.mask() > 0)

    def type(self):
        if self.ctype is None:
            self.ctype = self.lanes.judge_red_contour(self.cont)
        return self.ctype

class Lanes:
    def __init__(self, img):
        self._src = img
        h,w = self._src.shape[:2]
        self._crop = img[int(h*0.5):h, :]
        self._bmask = None
        self._rmask = None
        self._wmask = None

        h,w = self._crop.shape[:2]
        self._camerax = int(w/2)
        self._cameray = int(h)
        self._wall_mask = wall_mask(self._src)
        self.obs_mask = None
        self._smask = None
        self.RED_LINE = RED_LINE
        self.RED_LANE = RED_LANE

    def src_img(self):
        return self._src

    def crop_img(self):
        return self._crop

    def whole_wall_mask(self):
        return self._wall_mask

    def wall_mask(self):
        if not self._bmask is None:
            return self._bmask
        self._bmask = wall_mask(self._crop)
        return self._bmask

    def white_mask(self):
        if not self._wmask is None:
            return self._wmask
        masks = lanes_split(self._crop)
        (b, wmask, r) = cv2.split(masks)
        self._wmask = cv2.threshold(wmask, 127, 255, cv2.THRESH_BINARY)[1]
        return self._wmask

    def red_mask(self):
        if not self._rmask is None:
            return self._rmask
        self._rmask = util.flatten_red(self._crop)
        return self._rmask

    def sign_mask(self):
        if not self._smask is None:
            return self._smask
        img = self._src[0:self._cameray, :]
        self._smask = rsign._mask(img)
        return self._smask
 
    def top_contours(self, mask):
        binary, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        area = []
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i])) 

        conts = []
        if len(area) == 0:
            return conts
        idx = np.argmax(area)
        conts.append(contours[idx])

        cumask = 255 - self.contour_mask(contours[idx])
        rmask = self.mask_add(cumask, mask)
        binary, contours, hierarchy = cv2.findContours(rmask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        area = []
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i])) 

        if len(area) == 0:
            return conts
        
        idx = np.argmax(area)
        sec_cont = contours[idx]
        cmask = self.contour_mask(sec_cont)
        locs = np.where(cmask > 0)

        loc_y = np.sum(locs[0])/len(locs[0])
        h,w = mask.shape[:2]
        if loc_y  < h*4/5:
            conts.append(contours[idx])
        else:
            area[idx] = -1
            if len(area) == 1:
                return conts
            idx = np.argmax(area)
            conts.append(contours[idx])
        
        string=''' 
        cmask = self.contour_mask()
        area = np.array(area)
        idx = np.argsort(-area)

        for i in idx:
            conts.append(contours[i])
        '''
        return conts

    def contour_mask(self, contour):
        blank = np.zeros(self._crop.shape, np.uint8)
        cv2.fillPoly(blank, pts =[contour], color=(255,255,255))
        mask = cv2.split(blank)[0]
        return mask

    @staticmethod
    def mask_add(mask1, mask2):
        return cv2.bitwise_and(mask1, mask2)

    #1: red lane, 0:red line
    def judge_red_contour(self, contour):
        rmask = self.contour_mask(contour)
        wmask = self.white_mask()
        smask = self.mask_add(rmask, wmask)
        snum = np.sum(smask > 0)
        rnum = np.sum(rmask > 0)
        if snum > rnum:
            return -1
        #print("red mask num:", rnum) 
        #print("sign mask num:", snum)

        if snum > 1:
            return RED_LANE
        if rnum > 8000:
            return RED_LANE
        return RED_LINE

    def contour_minx(self, contour):
        mask = self.contour_mask(contour)
        locs = np.where(mask > 0)
        idx = np.argmax(locs[0])
        min_x = locs[1][idx]
        min_y = locs[0][idx]
        return min_x, min_y

    def judge_main_lane(self):
        rmask = self.red_mask()
        wmask = self.white_mask()

        h,w = rmask.shape[:2]
        rs = np.sum(rmask[int(0.5*h):h, self._camerax] > 0) 
        ws = np.sum(wmask[int(0.5*h):h, self._camerax] > 0)

        if rs > ws:
            return RED
        return WHITE

    def locate(self):
        rmask = self.red_mask()
        wmask = self.white_mask()
        #bmask = self.wall_mask()

        #cv2.imshow("red mask", rmask)
        #cv2.imshow("white mask", wmask)
        #cv2.imshow("wall mask", bmask)

        main = self.judge_main_lane()
        #print("# main:", main)
        #print("# camerax:", self._camerax)
        r_cnts = self.top_contours(rmask)
        if not len(r_cnts) > 0:
            return -1

        max_red_shape = self.judge_red_contour(r_cnts[0])
        max_red_minx, max_red_miny = self.contour_minx(r_cnts[0])
        #print("# max red minx:", max_red_minx)
        #print("# max red shape:", max_red_shape)
        if max_red_shape == -1:
            return -1
 
        #cv2.drawContours(self._crop, [r_cnts[0]], -1, (255,0,0), 3)
        sec_red_shape = -1
        sec_red_minx = -1

        if len(r_cnts) > 1:
            sec_red_shape = self.judge_red_contour(r_cnts[1])
            sec_red_minx, sec_red_miny = self.contour_minx(r_cnts[1])
            #print("# sec red minx:", sec_red_minx)
            #print("# sec red shape:", sec_red_shape)
            #cv2.drawContours(self._crop, [r_cnts[1]], -1, (0,255,0), 3)
            if sec_red_miny > max_red_miny:
                max_red_shape, sec_red_shape = sec_red_shape, max_red_shape
                max_red_minx, sec_red_minx = sec_red_minx, max_red_minx
 

       #main is red
        if main == RED:
            if max_red_shape == RED_LINE:
                if sec_red_shape == -1:
                    return -1
                else:
                    return 2.5
            else:
                if sec_red_shape != -1:
                    if sec_red_minx < self._camerax:
                        return 4
                    else:
                        return 1
                else:
                    return -1
        #main is white
        else:
            if max_red_shape == RED_LINE:
                if sec_red_shape == -1:
                    return -1
                if max_red_minx < self._camerax:
                    return 3
                else:
                    return 2
            else:
               if max_red_minx < self._camerax:
                    if sec_red_shape != -1:
                        if sec_red_minx < self._camerax:
                            return 5
                        else:
                            return 2
                    else:
                        return -1
               else:
                    if sec_red_shape != -1:
                        if sec_red_minx < self._camerax:
                            return 3
                        else:
                            return 0
                    else:
                        return -1

    def detect_cross(self): 
        wmask = wall_mask(self._src)
        cns = detect_corners(wmask)
        cns = right_angle(cns)
        cns = cns.astype(np.int32)
        cn = (np.sum(cns, axis = 0) / 2)
        x = int(cn[0])
        y = int(cn[1])
        return [x, y]

    def cross_point(self, ratio=1.7):
        cross_point=self.detect_cross()
        invalid = [-1, -1]

        if cross_point[0] == -1:
            return invalid
       # cv2.circle(self._src, (cross_point[0], cross_point[1]), 4, (0,0,255), -1)
        wmask = wall_mask(self._src)
        wmask = self._wall_mask

        h,w = wmask.shape
        x = cross_point[0]
        y = cross_point[1]
        cross_line = wmask[:,x]
        left_x = 0
        right_x = w-1

        if x >= 5:
            left_x = x -5
            
        if w-x > 5:
            right_x = x + 5
    
        left_line = wmask[:,left_x]
        right_line = wmask[:,right_x]

        cnum = np.sum(cross_line > 0)
        lnum = np.sum(left_line > 0)
        rnum = np.sum(right_line > 0)

        direct = 1
        cross_line = left_line
        cnum = lnum
        if lnum < rnum:
            direct = -1
            cross_line = right_line
            cnum = rnum
            
        length = cnum * ratio
        cx = x
        cy = y

        for i in range(len(cross_line.tolist())-1):
            if cross_line[i] > 0 and cross_line[i+1] == 0:
                cy = i

        cx = cx + direct*length
        cp =  [int(cx), int(cy)] 
        return cp

    def cross_angle(self, MAX_STEERING=6.5):
        cp = self.cross_point()
        if cp == [-1, -1]:
            return None
        w = self._src.shape[1]
        x = cp[0]
        angle = (x - 160)/10
        return [angle, cp]

    def obstacle_mask(self):
        img = self._src[0:self._cameray, :]
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
        lower = np.array([0, 20, 20])
        upper = np.array([180, 150, 150])
                
        obstacle = cv2.inRange(img_hsv,lower,upper)
        _y, _x = np.where(obstacle == 255)
 
        size = 5
        for index in range(0, len(_x), 1):
            obstacle_mask = obstacle[max(0, _y[index] - size):min(img.shape[0], _y[index] + size), max(0, _x[index] - size):min(img.shape[1], _x[index] + size)]
            obstacle_mask_count = np.count_nonzero(obstacle_mask == 255)
            if obstacle_mask_count < 10:
                obstacle[_y[index]][_x[index]] = 0
 
        blur = cv2.blur(obstacle, (5,5))
        obstacle = cv2.threshold(obstacle, 160, 255, cv2.THRESH_BINARY)[1]

        _y, _x = np.where(obstacle == 255)
        size = 5
        for index in range(0, len(_x), 1):
            obstacle_mask = obstacle[max(0, _y[index] - size):min(img.shape[0], _y[index] + size), max(0, _x[index] - size):min(img.shape[1], _x[index] + size)]
            obstacle_mask.fill(255)
 
        return obstacle        

    def detect_obstacle(self, ignore = False):
        if self.obs_mask is not None:
            return self.obs_mask

        img = self._src[0:self._cameray, :]
        sign_mask = rsign._mask(img)
        sign_on = np.sum(sign_mask > 0) > 10

        if ignore and sign_on:
            return None

        smask = None
        if sign_on:
            slocs = np.where(sign_mask > 0)
            max_y = max(slocs[0].tolist())
            height = img.shape[0]
            if max_y + 4 > height - 1:
                return None
            max_y += 4 
            smask = np.zeros(img.shape, np.uint8)
            smask[max_y:img.shape[0], :] = (255, 255, 255)
            smask = cv2.split(smask)[0]
       
        obs_mask = self.obstacle_mask()
        if smask is not None:
            obs_mask = self.mask_add(smask, obs_mask)    

        blur = cv2.blur(obs_mask, (5,5))
        nmask = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[1]

        binary, conts, hierarchy = cv2.findContours(obs_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        
        cons = []
        for cont in conts:
            area = cv2.contourArea(cont) 
            if area < 300:
                continue

            mask = self.contour_mask(cont)
            locs = np.where(mask > 0)
            num = np.sum(locs[0] < 0.9 * self._cameray)
            if num < 20:
                continue
            cons.append(cont)

        if len(cons) == 0:
            return None

        obs_mask = self.contour_mask(cons[0])
        lcons = cons[1:-1]
        for cont in lcons:
            cmask = self.contour_mask(cont)
            obs_mask = self.mask_add(cmask, obs_mask)
        
        string = '''
        conts = self.top_contours(nmask)
        if len(conts) == 0:
            self.obs_mask = obs_mask
            return obs_mask

        cmask = self.contour_mask(conts[0])
        locs = np.where(cmask > 0)
        max_y = max(locs[0])
        smask = None
        if max_y >= 0.8 * self._cameray:
            smask = np.zeros(img.shape, np.uint8)
            smask[max_y:img.shape[0], :] = (255, 255, 255)
            smask = cv2.split(smask)[0]
        if smask is not None:
            obs_mask = self.mask_add(smask, obs_mask)
        '''
            
        self.obs_mask = obs_mask
        return obs_mask

    def locate_obstacle(self):
        obs_mask = self.obs_mask
        if self.obs_mask is None:
            obs_mask = self.detect_obstacle()
            obs_mask = self.obs_mask

        if obs_mask is None:
            return None
        self.obs_mask = obs_mask
        binary, contours, hierarchy = cv2.findContours(obs_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        areas = []
        for i in range(len(contours)):
            areas.append(cv2.contourArea(contours[i])) 

        conts = []
        if len(areas) == 0:
            return None
        idx = np.argmax(areas)
        
        cont = contours[idx]
        area = areas[idx]
        if area < 300:
            return None
        
        cmask = self.contour_mask(cont)
        #cv2.imshow("contour mask", cmask)
        locs = np.where(cmask > 0)

        if len(locs[1]) <= 2:
            return None
        min_id = np.argmin(locs[1])
        max_id = np.argmax(locs[1])

        loc =  [(locs[1][min_id], locs[0][min_id]), (locs[1][max_id], locs[0][max_id])]
        return loc, cmask
        
    def on_main_lanes(self):
        rmask = self.red_mask()

        binary, contours, hierarchy = cv2.findContours(rmask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        areas = []
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > 50:
                areas.append(area) 

        num = len(areas)
        #print("red contours", num)
        if num >= 3:
            return 1

        if num == 0:
            return -1
        return 0

class Mark:
    def __init__(self):
        self.clf=svm.Classify()
        self.clf.load_model()

    def predict(self, img):
        on = mark_mask(img)
        if not on[0]:
            return -1

        smask = on[1]
        binary, contours, hierarchy = cv2.findContours(smask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        area = []
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i])) 

        if len(area) == 0:
            return -1
        idx = np.argmax(area)
        contour = contours[idx]
        locs = rsign.get_rectangle_locs(contour)

        max_x = locs[0][0]
        max_y = locs[0][1]
        min_x = locs[1][0]
        min_y = locs[1][1]

        mark = smask[min_y:max_y, min_x:max_x]
        return self.clf.predict(mark)

