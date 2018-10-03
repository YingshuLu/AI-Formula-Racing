import cv2
import sys
import numpy as np
import math

RC,GC,BC=1,2,3
WALL=0
#lanes = [WALL, GC, BC, GC, RC, BC, RC, WALL]
lanes = [WALL, RC, BC, RC, GC, BC, GC, WALL]

def crop(img, ratio):
    bottom_half_ratios = (ratio, 1.0)
    bottom_half_slice  = slice(*(int(x * img.shape[0]) for x in bottom_half_ratios))
    bottom_half        = img[bottom_half_slice, :, :]
    return bottom_half

def flatten(img):
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

def _crop_image(img):
    return crop(img, 0.7)

def _flatten_rgb(img):
    return flatten(img)

def preprocess(img):
    img = _crop_image(img)
    img = _flatten_rgb(img)
    return img

def find_dominant_color(track_img):
    b, g, r = cv2.split(track_img)

    sums = []
    rs = np.sum(r[r == 255].size)
    gs = np.sum(g[g == 255].size)
    bs = np.sum(b[b == 255].size)
    #black_sum = np.sum(((r == 255) & (g == 255) & (b == 255)))
    black_sum = 0

    #print("black sum:", black_sum)
    #print("red sum:", rs)
    #print("green sum:", gs)
    #print("blue sum:", bs)
    sums.append(black_sum) #if black_sum > 0
    sums.append(rs) #if rs > 0
    sums.append(gs) #if gs > 0
    sums.append(bs) #if bs > 0
    sums_copy = sums[:]

    index = []
    while len(sums_copy) > 0:
        max_val = max(sums_copy)
        if max_val == 0:
            break
        index.append(sums.index(max_val))
        del sums_copy[sums_copy.index(max_val)]
    return index

def locate_on_main_lanes(track_img):
    height, width = track_img.shape[:2]
    base = int(width / 3)
    left_track = track_img[:, 0:base]
    #mid_track = track_img[:, base+1 : 2*base]
    right_track = track_img[:, 2*base + 1 : width]

    all_idx = find_dominant_color(track_img)
    
    left_idx = find_dominant_color(left_track)
    right_idx = find_dominant_color(right_track)

    if len(all_idx) == 0:
         return -1

    if all_idx[0] == WALL:
        if len(all_idx) > 0:
            if GC in all_idx:
                return 5
            elif RC in all_idx:
                return 0
            else:
                return -1
    elif all_idx[0] == RC:
         if WALL in left_idx or BC in right_idx:
            return 0
         elif BC in left_idx or GC in right_idx:
            return 2
         else:
            return -1

    elif all_idx[0] == GC:
        if WALL in right_idx or BC in left_idx:
            return 5
        elif BC in right_idx or RC in left_idx:
            return 3
        else:
            return -1
    elif all_idx[0] == BC:
        if RC in left_idx or RC in right_idx:
            return 1
        elif GC in left_idx or GC in right_idx:
            return 4
        else:
            return -1

    return -1
    
def locate(img):
    track_img = preprocess(img)
    return locate_on_main_lanes(track_img) 

def binary_track_lines(rgb):
    track_img = crop(rgb, 0.55)
    #track_img = flatten(img)
    grayed      = cv2.cvtColor(track_img, cv2.COLOR_BGR2GRAY)
    blurred     = cv2.GaussianBlur(grayed, (3, 3), 0)

    sobel_x     = cv2.Sobel(blurred, cv2.CV_16S, 1, 0)
    sobel_y     = cv2.Sobel(blurred, cv2.CV_16S, 0, 1)
    sobel_abs_x = cv2.convertScaleAbs(sobel_x)
    sobel_abs_y = cv2.convertScaleAbs(sobel_y)
    edged       = cv2.addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0)

    lines       = cv2.HoughLinesP(edged, 1, np.pi / 180, 10, 5, 5)
    height, width = track_img.shape[:2]
    blank = np.zeros((height, width, 3), np.uint8)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank, (x1, y1), (x2, y2), (255, 255, 255), 3)
        
        #cv2.imshow("track lines", blank)

    sobel_x     = cv2.Sobel(blank, cv2.CV_16S, 1, 0)
    sobel_y     = cv2.Sobel(blank, cv2.CV_16S, 0, 1)
    sobel_abs_x = cv2.convertScaleAbs(sobel_x)
    sobel_abs_y = cv2.convertScaleAbs(sobel_y)
    edged       = cv2.addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0)

    blank = cv2.threshold(edged, 100, 255, cv2.THRESH_BINARY)[1]
    return blank

def get_binary_track_lines(src_img):
    img = crop(src_img, 0.55)
    track_img = flatten(img)
    grayed      = cv2.cvtColor(track_img, cv2.COLOR_BGR2GRAY)
    blurred     = cv2.GaussianBlur(grayed, (3, 3), 0)

    sobel_x     = cv2.Sobel(blurred, cv2.CV_16S, 1, 0)
    sobel_y     = cv2.Sobel(blurred, cv2.CV_16S, 0, 1)
    sobel_abs_x = cv2.convertScaleAbs(sobel_x)
    sobel_abs_y = cv2.convertScaleAbs(sobel_y)
    edged       = cv2.addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0)

    lines       = cv2.HoughLinesP(edged, 1, np.pi / 180, 10, 5, 5)
    height, width = track_img.shape[:2]
    blank = np.zeros((height, width, 3), np.uint8)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank, (x1, y1), (x2, y2), (255, 255, 255), 3)
        
        #cv2.imshow("track lines", blank)

    sobel_x     = cv2.Sobel(blank, cv2.CV_16S, 1, 0)
    sobel_y     = cv2.Sobel(blank, cv2.CV_16S, 0, 1)
    sobel_abs_x = cv2.convertScaleAbs(sobel_x)
    sobel_abs_y = cv2.convertScaleAbs(sobel_y)
    edged       = cv2.addWeighted(sobel_abs_x, 0.5, sobel_abs_y, 0.5, 0)

    blank = cv2.threshold(edged, 100, 255, cv2.THRESH_BINARY)[1]
    return blank

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
    #rrturn contour[max_id][0]
    
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
    cv2.line(blank, (line[0][0], line[0][1]), (line[1][0], line[1][1]), color, 2)


def predict_form(src_img):

    height, width = src_img.shape[:2]
    track = get_binary_track_lines(src_img)
    track_mask, _, _ = cv2.split(track)
    binary, contours, hierarchy = cv2.findContours(track_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i])) 

    dist = -1
    cut_line = np.array([[-1, -1], [-1, -1]])
    if len(area) > 0:
        max_id = np.argmax(area)
        contour = contours[max_id]
        locsx = get_cut_points(contour)
        cut_line = np.asarray(locsx)
        point = get_convex_point(cut_line, contour)
        dist = point_to_line_distance(cut_line, point) 
        
    #print("## track line offset:", dist)
    form = -1
    if dist == -1:
        form = -1
    elif dist < 11:
        form = 0
    else: 
        form = 1

    return form, cut_line

def predict(rgb):
    track = binary_track_lines(rgb)
    track_mask, _, _ = cv2.split(track)
    binary, contours, hierarchy = cv2.findContours(track_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i])) 

    dist = -1
    cut_line = np.array([[-1, -1], [-1, -1]])
    if len(area) > 0:
        max_id = np.argmax(area)
        contour = contours[max_id]
        locsx = get_cut_points(contour)
        cut_line = np.asarray(locsx)
        point = get_convex_point(cut_line, contour)
        dist = point_to_line_distance(cut_line, point) 
        
    #print("## track line offset:", dist)
    form = -1
    if dist == -1:
        form = -1
    elif dist < 11:
        form = 0
    else: 
        form = 1

    return form, cut_line

LANE_THRESHOLD = 50
# detect if car is on main lanes:
#  1: on main lanes
# -1: on narrow lanes
#  0: unkown
def on_main_lanes(src_img):
    low_img = crop(src_img, 0.5)
    track_img = flatten(low_img)

    #cv2.imshow("track image", track_img)
    b, g, r = cv2.split(track_img)
    rs = np.sum(r[r == 255].size)
    gs = np.sum(g[g == 255].size)
    bs = np.sum(b[b == 255].size)
 
    all_idx = []
    if rs > LANE_THRESHOLD:
        all_idx.append(RC)

    if bs > LANE_THRESHOLD:
        all_idx.append(BC)
    
    if gs > LANE_THRESHOLD:
        all_idx.append(GC)
    

    on = 0
    main = (sum(all_idx) >= 6)
    
    nmask = cv2.inRange(low_img, np.array([0, 170, 171]), np.array([0, 170, 171]), cv2.THRESH_BINARY)[1]
    
    #wall = np.sum(nmask)
    wall = np.sum(nmask[nmask == 255].size)
    nwall = (wall > 10)

    #print("lane ids:", all_idx)
    #print("rgb:", rs, gs, bs)
    #print("y:", wall)
    if main and nwall:
        if wall > RC or wall > GC or wall > BC:
            on = -1
        else:
            on = 1
    elif main:
        on = 1
    elif nwall:
        on = -1
    else:
        on = 0
    return on

def driving_on_main_lanes(src, rgb):
    low_img = crop(src, 0.5)
    track_img = crop(rgb, 0.5)
    #track_img = flatten(low_img)

    #cv2.imshow("track image", track_img)
    b, g, r = cv2.split(track_img)
    rs = np.sum(r[r == 255].size)
    gs = np.sum(g[g == 255].size)
    bs = np.sum(b[b == 255].size)
 
    all_idx = []
    if rs > LANE_THRESHOLD:
        all_idx.append(RC)

    if bs > LANE_THRESHOLD:
        all_idx.append(BC)
    
    if gs > LANE_THRESHOLD:
        all_idx.append(GC)
    

    on = 0
    main = (sum(all_idx) >= 6)
    
    nmask = cv2.inRange(low_img, np.array([0, 170, 171]), np.array([0, 170, 171]), cv2.THRESH_BINARY)[1]
    
    #wall = np.sum(nmask)
    wall = np.sum(nmask[nmask == 255].size)
    nwall = (wall > 10)

    print("lane ids:", all_idx)
    print("rgb:", rs, gs, bs)
    print("y:", wall)
    if main and nwall:
        if wall > RC or wall > GC or wall > BC:
            on = -1
        else:
            on = 1
    elif main:
        on = 1
    elif nwall:
        on = -1
    else:
        on = 0
    return on   

class Lane:
    def __init__(self, src):
        self._rgb = flatten(src)
        self._src = src
       
    def locate(self):
        img = _crop_image(self._rgb)
        return locate_on_main_lanes(img)

    def predict_form(self):
        return predict(self._rgb)

    def on_main_lanes(self):
        return driving_on_main_lanes(self._src, self._rgb)

           
        
         

if __name__ == '__main__':

    if len(sys.argv) < 2:
        exit()

    file_name = sys.argv[1]
    img = cv2.imread(file_name)

    form, cut_line = lane.predict_form(src_img)
    print("### lane form is:", form)

    if form != -1:
        lane.draw_line(src_img, cut_line, (0, 255, 255))

    print("location of main lanes:", (locate(img)))
    cv2.waitKey(6000)

    
