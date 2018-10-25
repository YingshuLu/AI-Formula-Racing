# coding=utf-8
import cv2, numpy as np
from skimage import measure
import Globals
import logger
import copy
import Settings
import TrafficSignClassify
import time
from ImageProcessor import ImageProcessor
import traceback

#min_r_pixel_count = 150
max_r_pixel_count = 800
min_r_width = 14
min_r_height = 5
max_r_width = 50
max_r_height = 30

logger = logger.get_logger(__name__)
def logit(msg):
    logger.info("%s" % msg)

#尝试确定离车最近的墙是左边还是右边，用来确定撞墙时，是从左边撞的还是右边撞的
def check_wall_direction(ori_img, wall_factor):
    img_width = ori_img.shape[1]
    img_height = ori_img.shape[0]
    thresh_hold = wall_factor*img_height*img_width/4
    part_to_check =  ori_img[int(0.20*img_height):int(0.65*img_height), :, :]
    b,g,r = cv2.split(part_to_check)
    blackmask = (b<30) & (r<30) & (g<30)
    blackwall_nonzero_index = blackmask.nonzero()[1]
    black_pixel_count = len(blackwall_nonzero_index)
    rgmask = (r<175) & (r>165) & (g<175) & (g>165) & (b<30)
    rgwall_nonzero_index = rgmask.nonzero()[1]
    rg_pixel_count = len(rgwall_nonzero_index)

    #print black_pixel_count, rg_pixel_count
    average_x = 0
    if black_pixel_count>rg_pixel_count and black_pixel_count>thresh_hold:
        average_x = blackwall_nonzero_index.mean()
    if rg_pixel_count>=black_pixel_count and rg_pixel_count>thresh_hold:
        average_x = rgwall_nonzero_index.mean()
    #print black_pixel_count, rg_pixel_count, average_x
    return average_x/img_width

#当前方有躲避障碍物标志时，变道一直到看到黑色的墙
def wall_is_seen(ori_img, direction, factor):
    img_width = ori_img.shape[1]
    img_height = ori_img.shape[0]
    thresh_hold = factor*img_width*img_height/4
    part_to_check = None
    if direction=='right':
        part_to_check =  ori_img[int(0.20*img_height):int(0.65*img_height), -img_width/2:, :]
    else:
        part_to_check = ori_img[int(0.20*img_height):int(0.65*img_height), 0:img_width/2, :]

    b,g,r = cv2.split(part_to_check)
    blackmask = (b<30) & (r<30) & (g<30)
    blackwall_nonzero_index = blackmask.nonzero()[1]
    black_pixel_count = len(blackwall_nonzero_index)
    #print black_pixel_count

    if black_pixel_count>thresh_hold:
        return True 
    else:
        return False

    
def check_sharp_turn_ahead(ori_img, factor):
    #ori_img中间部分长条状的墙（blackwall or rgwall), ori_img: 320*240
    medium_part_to_check = ori_img[75:155, :, :]
    b,g,r = cv2.split(medium_part_to_check)
    blackmask = (b<30) & (r<30) & (g<30)
    blackwall_nonzero_index = blackmask.nonzero()[0]
    black_pixel_count = len(blackwall_nonzero_index)
    rgmask = (r<175) & (r>165) & (g<175) & (g>165) & (b<30)
    rgwall_nonzero_index = rgmask.nonzero()[0]
    rg_pixel_count = len(rgwall_nonzero_index)

    #print black_pixel_count+rg_pixel_count
    if black_pixel_count>100 and rg_pixel_count>100 and black_pixel_count+rg_pixel_count>= 70*240*factor:
        return True
    else:
        return False


def check_next_to_wall(ori_img, wall_factor):     
    # 判断左右是否靠近墙,取中间部分，然后再判断左右两边是否有blackwall (超过wall_factor比率是墙可以认为靠近墙)    
    img_width = ori_img.shape[1]
    img_height = ori_img.shape[0]
    thresh_hold = wall_factor*img_height*img_width/4
    up_left_part =  ori_img[int(0.20*img_height):int(0.65*img_height), 0:img_width/2, :]
    up_right_part = ori_img[int(0.20*img_height):int(0.65*img_height), -img_width/2:, :]
    b,g,r = cv2.split(up_left_part)
    blackmask = (b<30) & (r<30) & (g<30)
    blackwall_nonzero_index = blackmask.nonzero()[0]
    left_black_pixel_count = len(blackwall_nonzero_index)


    b,g,r = cv2.split(up_right_part)
    blackmask = (b<30) & (r<30) & (g<30)
    blackwall_nonzero_index = blackmask.nonzero()[0]
    right_black_pixel_count = len(blackwall_nonzero_index)

    #print left_black_pixel_count,right_black_pixel_count,left_rg_pixel_count,right_rg_pixel_count

    if left_black_pixel_count>thresh_hold and right_black_pixel_count>thresh_hold:
        return Globals.BlackWallBoth
    if right_black_pixel_count>thresh_hold:
        return Globals.OnlyBlackWallRight
    if left_black_pixel_count>thresh_hold:
        return Globals.OnlyBlackWallLeft
    
    return None   



def white_ratio(img):
    b,g,r = cv2.split(img)
    redmask = (r>200) &  (b>200) &  (g>200)
    red_nonzero_index = redmask.nonzero()[0]
    red_pixel_count = len(red_nonzero_index)
    return red_pixel_count*1.0/(img.shape[0]*img.shape[1])


# return isWrongWay, arrowImg
def detect_wrongway(ori_img):
    try:
        up_part        = ori_img[160:240, :, :]
        b,g,r = cv2.split(up_part)

        redmask = (r>200) & ((r-100)>b) & ((r-100)>g)
        red_nonzero_index = redmask.nonzero()[0]
        red_pixel_count = len(red_nonzero_index)

        if red_pixel_count<320*50:
            return None, None

        whitemask = (r>200) & (g>200) & (b>200)
        r[whitemask]=255
        r[np.invert(whitemask)] = 0
        r   = cv2.dilate(r, None, iterations=4)
        #cv2.imwrite('r.jpg', r)

        #连通域标记
        # perform a connected component analysis on the thresholded
        # image, then initialize a mask to store only the "large"
        # components
        labels = measure.label(r, neighbors=4, background=0)

        # first find the region that has at least 120 red pixels
        # choose the one in the medium if there are three, otherwise, choose the one has small pixels
        sign_img_to_check = None
        
        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue
        
            # otherwise, construct the label mask and count the
            # number of pixels 
            labelMask = np.zeros(r.shape, dtype="uint8")
            labelMask[labels == label] = 255
            non_zero_index = labelMask.nonzero()
            white_pixels_count = len(non_zero_index[0])
            
            if white_pixels_count > 500:
                non_zero_index_x = non_zero_index[1]
                non_zero_index_y = non_zero_index[0]
                min_x = non_zero_index_x.min()
                max_x = non_zero_index_x.max()
                min_y = non_zero_index_y.min()
                max_y = non_zero_index_y.max()
                mean_x = non_zero_index_x.mean()
                mean_y = non_zero_index_y.mean()
                #print mean_x, mean_y, (max_x-min_x), (max_y-min_y)
                if mean_x>80 and mean_x<240 and mean_y>15 and mean_y<70 and (max_x-min_x)>60 and (max_y-min_y)>20:
                    sign_img_to_check = up_part[min_y:max_y, min_x:max_x, :]
                    whiteratio = white_ratio(sign_img_to_check)
                    if whiteratio>0.2 and whiteratio<0.35:
                        arrow_type = TrafficSignClassify.get_arrow_type(sign_img_to_check)
                        #print 'arrow_type='+str(arrow_type)
                        #ImageProcessor.force_save_image_to_log_folder(sign_img_to_check, prefix = "arrow", suffix=str(arrow_type))
                        if arrow_type == 10:
                            return True, sign_img_to_check
                    else:
                        sign_img_to_check = None

        if sign_img_to_check is not None:
            return False, sign_img_to_check

        return None, None
    except Exception as exception:
        print(exception)
        traceback.print_exc()
        return None, None
    

def detect_obstacle(ori_img, obstacle_pixel_count_threshold):
    try:
        # -1 means wall on the left side, 1 means wall on the right side, 0 means the track is just on the front, turn right or left is ok
        on_side = 0
        img_width = ori_img.shape[1]
        img_height = ori_img.shape[0]
        part_y1 = int(0.45*img_height)
        part_y2 = int(0.75*img_height)
        part_x1 = int(0.30*img_width)
        part_x2 = int(0.70*img_width)

        medium_part = ori_img[part_y1:part_y2, part_x1:part_x2, :]
        b,g,r = cv2.split(medium_part)

        obstaclemask = (r>15) & (g>15) & (b>15) & (r<90) & (b<90) & (g<90) 
        r[obstaclemask]=255
        r[np.invert(obstaclemask)] = 0            
        r = cv2.dilate(r, None, iterations=4)
        #cv2.imwrite('obstacle.jpg', r)

        blackmask = (r<5) & (g<5) & (b<5)
        black_nonzero_index_x = blackmask.nonzero()[1]
        black_nonzero_index_pixel_count = len(black_nonzero_index_x)

        obstacle_nonzero_index_x = obstaclemask.nonzero()[1]
        obstacle_nonzero_index_pixel_count = len(obstacle_nonzero_index_x)
        #print obstacle_nonzero_index_pixel_count
        # if obstacle_nonzero_index_pixel_count>obstacle_pixel_count_threshold:
        #     #print obstacle_nonzero_index_pixel_count
        #     x_min = obstacle_nonzero_index_x.min()
        #     x_max = obstacle_nonzero_index_x.max()
        #     #print x_min, x_max
        #     mean_x = (x_min+x_max)/2
        #     if mean_x>0.52*(part_x2-part_x1):
        #         on_side = 1
        #     elif mean_x<0.48*(part_x2-part_x1):
        #         on_side = -1
        #     else:
        #         on_side = 0
        #     return on_side                            
        #print obstacle_nonzero_index_pixel_count 

        if obstacle_nonzero_index_pixel_count>obstacle_pixel_count_threshold and black_nonzero_index_pixel_count+obstacle_nonzero_index_pixel_count<2000:
            wall_part = ori_img[int(0.35*img_height):int(0.65*img_height),:, :]        
            wb,wg,wr = cv2.split(wall_part)
            wallmask = (wr<5) & (wg<5) & (wb<5)
            wr[wallmask] = 255
            wr[np.invert(wallmask)] = 0
            #wr = cv2.dilate(wr, None, iterations=4)
            wr_average_x = wr.nonzero()[1].mean()
            #cv2.imwrite('wall.jpg', wr)
            if wr_average_x>0.51*img_width:
                on_side = 1
            elif wr_average_x<0.49*img_width:
                on_side = -1
            else:
                on_side = 0
            #ImageProcessor.force_save_bmp_to_log_folder(ori_img, str(on_side))    
            #print wr_average_x, on_side           
            return on_side
            #cv2.imwrite('obstaclemask.jpg', r)
        return None

    except Exception as exception:
        print(exception)
        traceback.print_exc()
        return None


def check_wall_obstacle(ori_img, wall_factor):
    # 判断前方是否撞墙或者障碍物
    # 墙分两种情况，一种是黑色的最外围的，为纯黑色，另外一种是分叉赛道的，RGB为(170,170,0)
    part_y1 = int(ori_img.shape[0]/2)
    part_y2 =ori_img.shape[0]
    img_width = ori_img.shape[1]
    img_height = ori_img.shape[0]
    down_part = ori_img[part_y1:part_y2, :, :]
    b,g,r = cv2.split(down_part)
    blackmask = (b<30) & (r<30) & (g<30)
    blackwall_nonzero_index = blackmask.nonzero()[0]
    if len(blackwall_nonzero_index)>img_width*img_height/3:
        #print 'StuckBlackWall len(nonzero_index)=%d' % (len(blackwall_nonzero_index))
        return Globals.StuckBlackWall
    

    # 障碍物一般为小车的车轮
    up_part        = ori_img[0:part_y1, :, :]
    b,g,r = cv2.split(up_part)
    obstaclemask = (r>5) & (g>5) & (b>5) & (r<90) & (b<90) & (g<90) 
    bot_nonzero_index = obstaclemask.nonzero()[0]
    if len(bot_nonzero_index)>img_width*img_height/8:
        return Globals.StuckObstacle

    return None

def may_have_traffic_sign(ori_img, traffic_sign_count_threshold):
    part_y1 = 0
    part_y2 = int(0.45*ori_img.shape[0])
    img_width = ori_img.shape[1]
    img_height = ori_img.shape[0]
    up_part        = ori_img[part_y1:part_y2, :, :]
    b,g,r = cv2.split(up_part)

    redmask = (r>90) & (r-b>40) & (r-g>40)
    red_nonzero_index = redmask.nonzero()[0]
    red_pixel_count = len(red_nonzero_index)
    if red_pixel_count>traffic_sign_count_threshold:
        return True  
    
def detect_traffic_sign_type(ori_img, traffic_sign_count_threshold):
    traffic_sign = None
    try:
        part_y1 = 0
        part_y2 = int(0.45*ori_img.shape[0])
        up_part        = ori_img[part_y1:part_y2, :, :]
        b,g,r = cv2.split(up_part)

        redmask = (r>100) & ((r-70)>b) & ((r-70)>g)
        red_nonzero_index = redmask.nonzero()[0]
        red_pixel_count = len(red_nonzero_index)

        # R像素数量少于traffic_sign_count_threshold，可能不存在或不完整,或者在顶端，忽略
        if red_pixel_count<traffic_sign_count_threshold or red_pixel_count>3000:
            if red_pixel_count<traffic_sign_count_threshold and red_pixel_count>traffic_sign_count_threshold*0.5:
                traffic_sign = Globals.MayHaveTrafficSign                 
            return None, None

        traffic_sign = 0
        r[redmask]=255
        r[np.invert(redmask)] = 0

        r   = cv2.dilate(r, None, iterations=4)
        #cv2.imwrite('r.jpg', r)

        #连通域标记
        # perform a connected component analysis on the thresholded
        # image, then initialize a mask to store only the "large"
        # components
        labels = measure.label(r, neighbors=4, background=0)

        # first find the region that has at least 120 red pixels
        # choose the one in the medium if there are three, otherwise, choose the one has largest pixels
        traffic_sign_img_list = []
        largest_pixels_count = 3000
        largest_pixels_sign_img = None
        sign_img_to_check = None
        
        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue
        
            # otherwise, construct the label mask and count the
            # number of pixels 
            labelMask = np.zeros(r.shape, dtype="uint8")
            labelMask[labels == label] = 255
            non_zero_index = labelMask.nonzero()
            red_pixels_count = len(non_zero_index[0])
        
            if red_pixels_count > traffic_sign_count_threshold:
                non_zero_index_x = non_zero_index[1]
                non_zero_index_y = non_zero_index[0]
                min_x = non_zero_index_x.min()
                max_x = non_zero_index_x.max()
                min_y = non_zero_index_y.min()
                max_y = non_zero_index_y.max()
                sign_img_width = max_x-min_x
                sign_img_height = max_y-min_y
                if sign_img_width*sign_img_height>20*15:
                    sign_img = ori_img[min_y:max_y, min_x:max_x, :]
                    #cv2.imwrite(str(red_pixels_count)+'.jpg', sign_img) 
                    traffic_sign_img_list.append((red_pixels_count, sign_img))
                    if red_pixels_count>largest_pixels_count:
                        largest_pixels_count = red_pixels_count
                        largest_pixels_sign_img = sign_img

        if len(traffic_sign_img_list)==3:            
            sign_img_to_check = traffic_sign_img_list[1][1]
        else:
            sign_img_to_check = largest_pixels_sign_img

        #cv2.imwrite('sign.jpg', sign_img_to_check)   

        if sign_img_to_check is not None:
            sign_type = TrafficSignClassify.get_traffic_sign_type(sign_img_to_check)
            # if sign_type in range(1,9):
            #     ImageProcessor.force_save_image_to_log_folder(sign_box_ori, prefix = "ori", suffix=str(sign_type))
            if sign_type is not None:
                traffic_sign = sign_type
            return traffic_sign, sign_img_to_check

        return None, None
    except Exception as exception:
        logger.error(exception)
        traceback.print_exc()
        return (None, None)


if __name__ == '__main__': 
    import argparse    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help = "path to the image file")
    args = vars(ap.parse_args())
    img = cv2.imread(args["image"])

    time_begin = time.time()
    sign_type, ori_sign_box = detect_traffic_sign_type(img, Settings.TRAFFIC_SIGN_PIXEL_COUNT_THRESHOLD)
    time_cost = time.time()-time_begin

    if ori_sign_box is not None:
        print('find traffic sign {}'.format(sign_type))
        print (time_cost)
        cv2.imwrite('sign_box.jpg', ori_sign_box)
    else:
        print ('no traffic sign found')

    on_side = detect_obstacle(img, 95)
    if on_side is not None:
        print('found obstacle on ' + str(on_side))
        #cv2.imwrite('obstaclemask.jpg', obstacle_img)  

    # print 'time_cost={}'.format(time_cost)
    # wall_condition = check_stuck_wall_obstacle(img)
    # if wall_condition is not None:
    #     print 'wall_condition=%d' % (wall_condition)
    # import numpy as np
    # np.set_printoptions(threshold=np.inf)
    # wrongway = check_wrong_way(img)
    # if wrongway is not None:
    #     print 'wrongway=%d' % (wrongway)

    # wall_is_seen = wall_is_seen(img, 'left', 0.25)
    # print 'wall_is_seen {}'.format(wall_is_seen)

    # next_to_wall = check_next_to_wall(img, 0.15)
    # print 'check_next_to_wall {}'.format(wall_is_seen)
    
    # sharp_turn_ahead = check_sharp_turn_ahead(img, Settings.SharpTurnWallFactor)
    # print sharp_turn_ahead

    # image_mark_color = ImageProcessor.preprocess_ex(img, 0.55)
    # cv2.imwrite('image_mark_color.jpg', image_mark_color)