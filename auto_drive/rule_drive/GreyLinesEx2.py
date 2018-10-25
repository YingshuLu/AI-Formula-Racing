import cv2 
import math
import numpy as np
from ImageProcessor import ImageProcessor,ImageLines
import Settings
import logger
import time
import copy
from keras.models import load_model
from skimage import measure
from skimage import morphology
from skimage.color import rgb2grey
import imutils
from TrafficSignType import detect_obstacle
from skimage import morphology
logger = logger.get_logger(__name__)

#lostLine_model = load_model('lostLine_v4.h5')
lostLine_model = load_model('lostLine_v5.h5')

def logit(msg):
    pass
    #if Settings.DEBUG:
    #    logger.info("%s" % msg)
    #    print(msg)

class GreyLines(object):
    prev_obs_detect_time = -1
    prev_obs_turn_start_pos = -1
    @staticmethod
    def drawline(greyimage):
        cv2.line(greyimage, (0, 0), (100, 100), (255, 0, 0), 2)
    
    @staticmethod
    def PrintLines(lines, name, img):
        if lines is not None: 
            for line in lines:
                if line is not None:
                    for x1, y1, x2, y2 in line:
                        #if(y1 == y2 and (abs(x1-x2) > 5)):
                        #    continue
                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

        ImageProcessor.show_image(img, name)
   
    @staticmethod 
    def DrawPoints(nparray, name):
        img = np.zeros((132,320,1), np.uint8)
        for xx in nparray:
            x = xx[1]
            y = xx[0]
            cv2.circle(img,(x, y),1,(55,255,155),1)
        ImageProcessor.show_image(img, name)

    @staticmethod
    def ArraySum(array):
        arrsum = np.sum(array, 0)
        #print "DstArray size: %d, arrsum: %s" % (array.size, arrsum)

        if array.size >0 and array.shape[0] != 0:
            lanepoint = arrsum/array.shape[0]
        else:
            lanepoint = [0, 0]
        return [lanepoint[1], lanepoint[0]]

                
    @staticmethod
    def GetCoordinate(edge):
        #np.set_printoptions(threshold='nan')
        ans = np.zeros((edge.shape[1], edge.shape[0]), dtype=int)
        for y in range(0, edge.shape[0]):
            for x in range(0, edge.shape[1]):
                if edge[y, x] != 0:
                    ans[x, y] = 1
                    #print "(%d, %d)" % (x, y)

        print("numpy shape: %d %d"% (ans.shape[0], ans.shape[1]))
        #print "ans: %s" % ans
        return ans

    @staticmethod
    def GetZone(edgecoor, part = [0, 0, 0, 0]):
        #np.set_printoptions(threshold='nan')
        #print "part: %s, %s, %s, %s" % (part[0], part[1], part[2], part[3])
        ybeg = edgecoor.shape[0]/part[0] if part[0] > 0 else 0
        yend = edgecoor.shape[0]/part[1] if part[1] > 0 else edgecoor.shape[0]

        xbeg = edgecoor.shape[1]/part[2] if part[2] > 0 else 0
        xend = edgecoor.shape[1]/part[3] if part[3] > 0 else edgecoor.shape[1]
        
        #print "ybeg: %d, yend: %d, xbeg: %d, xend: %d"% (ybeg, yend, xbeg, xend)
        targetzone = edgecoor[ybeg:yend,xbeg:xend]
        #print "targetzone: %s" % targetzone
        return targetzone
        

    @staticmethod
    def EdgeMergeByZone(srcedge, objedge, zone = [0, 0, 0, 0]):
        srcedgeslice = GreyLines.GetZone(srcedge, zone)
        objectedgeslice = GreyLines.GetZone(objedge, zone)
        rstedge = np.logical_xor(srcedgeslice, objectedgeslice)
        #print "rstedge: %s " % rstedge
        indexs = np.transpose(rstedge.nonzero())
        #print "index of nonzero: %s" % indexs
        return indexs


    @staticmethod
    def EdgeMergeByValidPixel(srcedge, objedge):
        ##np.set_printoptions(threshold='nan')
        rstedge = np.logical_xor(srcedge, objedge)
        #print "rstedge: %s " % rstedge
        indexs = np.transpose(rstedge.nonzero())
        #print "index of nonzero: %s" % indexs
        return indexs[:indexs.shape[0], :]

    @staticmethod
    def Drawline(img, point):
        image_height = img.shape[0]
        image_width  = img.shape[1]
        #print "image: heigh: %d, width: %d" % (image_height, image_width)
        cv2.line(img,(image_width/2, image_height),(point[0], point[1]),(255,255,0),3)

    @staticmethod
    def GetAngle(carx, cary, expectx, expecty):
        #print carx, cary, expectx, expecty
        myradians = math.atan2(expectx-carx, cary - expecty)
        return math.degrees(myradians)
    
    @staticmethod
    def GetleftWallZoneCoor(rawarray, blackwallimg):
        rows = np.where(rawarray[:, 0] > blackwallimg.shape[0] * Settings.eagle_wall_ylevel)
        indexs1 = rawarray[rows]

        rows = np.where(indexs1[:,1] < blackwallimg.shape[1] * Settings.eagle_wall_xlevel)
        indexs = indexs1[rows]
        firstclomn = indexs[:,1]
        validcolumn = list(set(firstclomn))
        yorglen, yselen = len(firstclomn), len(validcolumn)
        
        wallpoint = [0, 0]
        rstangle = 0
        if yorglen < Settings.eagle_wall_throttle:
            return wallpoint, rstangle
        elif yorglen > Settings.eagle_wall_throttle:
            #logit('GetleftWallZoneCoor  xdensity (%s %s) ydensity(%s  %s)' % (xorglen, xselen, yorglen, yselen))
            if yselen > 0 and yorglen/yselen > Settings.eagle_wall_density:
                return wallpoint, rstangle
        else:
            if Settings.DEBUG:
                logit('GetleftWallZoneCoor indexs size: %s' % (indexs.size)) 
                logit('ydensity(%s  %s)' % (yorglen, yselen))
                #logit('GetleftWallZoneCoor indexs :%s' % (indexs)) 
       
        wallpoint = GreyLines.ArraySum(indexs) 
        rstangle = GreyLines.GetAngle(blackwallimg.shape[1]/2, blackwallimg.shape[0], wallpoint[0], wallpoint[1])
        
        return wallpoint, rstangle

    @staticmethod
    def GetRightWallZoneCoor(rawarray, blackwallimg):
        rows = np.where(rawarray[:, 0] > blackwallimg.shape[0] * Settings.eagle_wall_ylevel)
        indexs1 = rawarray[rows]

        rows = np.where( indexs1[:,1] > blackwallimg.shape[1] * (1-Settings.eagle_wall_xlevel))
        indexs = indexs1[rows]
        firstclomn = indexs[:,1]
        validcolumn = list(set(firstclomn))
        yorglen, yselen = len(firstclomn), len(validcolumn)

        wallpoint = [0, 0]
        rstangle = 0
        if yorglen < Settings.eagle_wall_throttle:
            return wallpoint, rstangle
        elif yorglen > Settings.eagle_wall_throttle:
            if yselen > 0 and yorglen/yselen > Settings.eagle_wall_density:
                return wallpoint, rstangle
        else:
            if Settings.DEBUG:
                logit('GetRightWallZoneCoor indexs size:%s' % (indexs.size))
                logit('ydensity(%s  %s)' % (yorglen, yselen))
                #logit('GetleftWallZoneCoor indexs :%s' % (indexs))
        wallpoint = GreyLines.ArraySum(indexs) 
        rstangle = GreyLines.GetAngle(blackwallimg.shape[1]/2, blackwallimg.shape[0], wallpoint[0], wallpoint[1])
        
        return wallpoint, rstangle

    @staticmethod
    def GetWallAngle(Walls, blackwallimg):
        ##np.set_printoptions(threshold='nan')

        indices = np.where( Walls != [0])
        indexs = np.array (zip(indices[0], indices[1])) 
        
        leftlanepoint, leftangle = [0, 0], 0
        rightlanepoint, rightangle = [0, 0], 0
        
        if indexs.size == 0:
            return leftlanepoint, leftangle, rightlanepoint, rightangle

        leftlanepoint, leftangle = GreyLines.GetleftWallZoneCoor(indexs, blackwallimg)
        #print "left lanepoint: %s, angle: %s" % (leftlanepoint, leftangle)

        rightlanepoint, rightangle = GreyLines.GetRightWallZoneCoor(indexs, blackwallimg)
        #print "right lanepoint: %s, angle: %s" % (rightlanepoint, rightangle)
        
        #For to turn angle, wall on left, go to righ and vise versa
        # if leftangle != 0 and rightangle != 0:
        #    angletotrun = lanesangle if lanesangle !=0 else last_lanes_angle  
        #    wallpoint = [(rightlanepoint[0] + leftlanepoint[0])/2, (rightlanepoint[1] + leftlanepoint[1])/2]
        # elif leftangle != 0 or rightangle != 0: # compare y coordinate
        #     if rightangle == 0 or (leftangle != 0 and leftlanepoint[1] < rightlanepoint[1]):
        #         angletotrun = 90 + leftangle
        #         wallpoint = leftlanepoint               
        #     else:
        #         angletotrun = rightangle - 90
        #         wallpoint = rightlanepoint
    
        if Settings.DEBUG_IMG: 
            if leftangle != 0:
                 GreyLines.Drawline(blackwallimg, leftlanepoint)
            if rightangle != 0:     
                 GreyLines.Drawline(blackwallimg, rightlanepoint)
            #ImageProcessor.show_image(LanesCoor, "Lanes")
            ImageProcessor.show_image(Walls, "Walls")

        return leftlanepoint, leftangle, rightlanepoint, rightangle

    #parts cooradinate ratio: [ybeg, yend, xbeg, xend]
    @staticmethod
    def GetLanesAngle(WallAndLanes, Walls, blackwallimg, direction = 0):
        targetxy = GreyLines.EdgeMergeByValidPixel(WallAndLanes, Walls) # Remove wall edges
        #targetzone = GreyLines.EdgeMergeByZone(WallAndLanes, Walls, Settings.birdviewpart)
        rstangle, lanepoint = 0, [0, 0]
        if targetxy.size == 0:
            return rstangle, lanepoint
        slicedarray = targetxy[:int(targetxy.shape[0]/3), :]

        firstclomn = slicedarray[:,1]
        validcolumn = list(set(firstclomn))
        orglen, selen = len(firstclomn), len(validcolumn)
        if orglen < Settings.eagle_lanes_throttle:
            return rstangle, lanepoint
        #print 'GetLanesAngle indexs size:%s,selen %s' % (orglen, selen)
        
        validarray = slicedarray
        sliceratio = 0
        if direction == -1: #turn left
            sliceratio = blackwallimg.shape[1] * (1-Settings.eagle_traffice_slice_ratio)
            rows = np.where(slicedarray[:, 1] < sliceratio)
            validarray = slicedarray[rows]
        if direction == 1: #turn right
            sliceratio = blackwallimg.shape[1] * Settings.eagle_traffice_slice_ratio
            rows = np.where(slicedarray[:, 1] > sliceratio)
            validarray = slicedarray[rows]
        
        lanepoint = GreyLines.ArraySum(validarray) # Pickup specified zone to calc the center points

        if Settings.DEBUG_IMG: 
            GreyLines.Drawline(blackwallimg,lanepoint)
            #ImageProcessor.show_image(blackwallimg, "blackwallimg")
            #ImageProcessor.show_image(LanesCoor, "Lanes")
            #ImageProcessor.show_image(Walls, "Walls")

        if lanepoint[0] == 0 and lanepoint[1] == 0:
            if Settings.DEBUG:
                logit("No lanes can be found")
                logit("slice ratio: %s, turn %s" % (sliceratio, direction))
        else:
            rstangle = GreyLines.GetAngle(blackwallimg.shape[1]/2, blackwallimg.shape[0], lanepoint[0], lanepoint[1])
        
        return rstangle, lanepoint, 

    @staticmethod
    def GetEdgeImages(srcimg):
        blackwallimg = ImageProcessor.preprocess(srcimg, 0.5) # Get cropped image and convert image wall to black

        blackwallimg = cv2.medianBlur(blackwallimg, 3)

        whilelanes = GreyLines.ChangeLaneToWhite(blackwallimg) # change all lanes to white
        #ImageProcessor.show_image(whilelanes, "whilelanes")
        Walls = GreyLines.TractGrey(whilelanes) # Get wall edges
        WallAndLanes = GreyLines.TractGrey(blackwallimg) # Get lanes and wall edges

        #cv2.imwrite("TestImage/WallAndLanes_"+str(time.time())+".jpg", WallAndLanes)          
        

        return WallAndLanes, Walls, blackwallimg
        #GreyLines.DrawPoints(LanesCoor, "LanesCoorpoint")
        #GreyLines.DrawPoints(WallAndLanesCoor, "WallAndLanesCoorpoint")
    
    @staticmethod
    def CalRoadPosition(srcimg):
        
        road_image = GreyLines.GetRoadImages(srcimg)
        road_image_temp = road_image.copy()

        middle_position = road_image_temp.shape[1]/2

        #print "middle_position", middle_position
        row_position = []
        has_check_road = False
        for row in road_image_temp[::-1]:
            if len(row[row>100]) > (len(row) -2):
                row[row>100] = 50
            else:  
                position = []
                if has_check_road and len(row[row<10]) ==  len(row) and len(position)>0:
                    if row_position[-1] > middle_position:
                        row_position.append(len(row))
                    else:
                        row_position.append(0)
                    continue
                has_check_road = True 
                for col_n, pixel in enumerate(row):
                    if pixel > 100:
                       position.append(col_n)
                if len(position) > 0:
                    row_position.append(np.array(position).mean())
        ImageProcessor.show_image(road_image_temp, "road_image_temp")

        car_pos = np.array(row_position).mean()

        
            
        print("row_position",car_pos)        
        if len(row_position) == 0:
            return 160
        return car_pos
    
    @staticmethod
    def CalRoadNumber(srcimg):
        blackwallimg = ImageProcessor.preprocess(srcimg, 0.5) # Get cropped image and convert image wall to black

        blackwallimg = cv2.medianBlur(blackwallimg, 5)

        ImageProcessor.show_image(blackwallimg, "blackwallimg")
        line_number = 0
        line_color = []
        for row_pixel in blackwallimg[::-1]:
            if row_pixel[0][0]<10 and row_pixel[0][1]<10 and row_pixel[0][2]<10 and\
                row_pixel[1][0]<10 and row_pixel[1][1]<10 and row_pixel[1][2]<10 and\
                row_pixel[-1][0]<10 and row_pixel[-1][1]<10 and row_pixel[-1][2]<10 and\
                row_pixel[-2][0]<10 and row_pixel[-2][1]<10 and row_pixel[-2][2]<10:
                #Both side is black
                for pixel in row_pixel[2:(len(row_pixel)-2)]:
                    r = pixel[0]
                    g = pixel[1]
                    b = pixel[2]
                    if r< 10 and g<10 and b<10:
                        continue
                    if len(line_color) == 0:
                        if r > 10 and g<10 and b< 10:
                            line_color.append("red")
                        elif r < 10 and g>10 and b< 10:
                            line_color.append("green")
                        elif r < 10 and g<10 and b> 10:
                            line_color.append("blue")
                        else:
                            print("---------------Color error  in road line")
                    elif r > 10 and g<10 and b< 10 and line_color[-1] != "red":
                        #red line
                        line_color.append("red")
                    elif r < 10 and g>10 and b< 10 and line_color[-1] != "green":
                        #green line
                        line_color.append("green")
                    elif r < 10 and g<10 and b>10 and line_color[-1] != "blue":
                        line_color.append("blue")
                break
        print(line_color)

        if len(line_color) <2:
            line_number = 0
        elif len(line_color)<4:
            line_number = 3
        else:
            line_number = 6

        print("line number ", len(line_color))
        return len(line_color)
    @staticmethod
    def has_two_line(line_data):
        last_pos = line_data[0]
        for line_pos in line_data:
            if (line_pos - last_pos) >2:
                return True
            last_pos = line_pos      
        return False
    @staticmethod
    def middle_line_pos(line_data):
        last_pos = line_data[0]
        for line_pos in line_data:
            if (line_pos - last_pos) >2:
                return last_pos,(line_pos+last_pos)/2,line_pos
            last_pos = line_pos      
        return -1


    @staticmethod
    def GetTwoLinePos(line_image, line_number):
        row_line = line_image[line_number]
        line_pos = np.where(row_line==255)
        print("STart to get two lone pos",time.time())
        double_line_pos = -1
        line_list = []

        if len(line_pos[0]) == 0:
            #No line
            left_point = -1
            right_point = -1
            for lower_number in range(100):
                if line_number+lower_number+1 >= 119:
                    break
                lower_row_line = line_image[line_number+lower_number+1]
                if lower_row_line[0] == 255 and left_point == -1:
                    left_point = line_number+lower_number+1
                if lower_row_line[319] == 255 and right_point == -1:
                    right_point = line_number+lower_number+1

                if left_point != -1 and right_point!= -1:
                    print("left_point,right_point",left_point,right_point)
                    if left_point > right_point:
                        double_line_pos = 320
                    else:
                        double_line_pos = 0
                    break

            if double_line_pos == -1:
                for lower_number in range(50):
                    #print "line_number+lower_number",line_number+lower_number+1
                    if line_number+lower_number >= 119:
                        break
                    lower_row_line = line_image[line_number+lower_number+1]
                    lower_line_pos = np.where(lower_row_line==255)
                    if len(lower_line_pos[0]) > 2:
                        
                        below_lower_row_line = line_image[line_number+lower_number+2]
                        below_lower_line_pos = np.where(below_lower_row_line==255)

                        if lower_line_pos[0].mean() > below_lower_line_pos[0].mean():
                            #turn Right
                            double_line_pos = line_image.shape[1]
                        else:
                            double_line_pos = 0
                        break

        elif len(line_pos[0]) != 0 and GreyLines.has_two_line(line_pos[0]):
            #Has two line
            print("Has two line")
            left_pos, double_line_pos, right_pos = GreyLines.middle_line_pos(line_pos[0])              
        else:
            double_line_row = -1
            need_to_check_upper = True
            need_to_check_lower = True
            for i in range(120):
                upper_line = line_number-i-1
                lower_line = line_number+i+1
                if upper_line >= 0 and need_to_check_upper:
                    upper_row_line = line_image[upper_line]
                    upper_line_pos = np.where(upper_row_line==255)
                    if len(upper_line_pos[0]) == 0:
                        need_to_check_upper = False
                    elif GreyLines.has_two_line(upper_line_pos[0]):
                        print("Find two line row,upper_line",upper_line)
                        double_line_row = upper_line
                        last_left_pos, last_middle_pos, last_right_pos = GreyLines.middle_line_pos(upper_line_pos[0])
                        line_list.append(last_middle_pos)
                        for j in range(line_number - upper_line):
                            current_line = line_image[upper_line+j+1]
                            current_line_pos = np.where(current_line==255)
                            if len(current_line_pos[0]) == 0:
                                current_left_pos, current_middle_pos, current_right_pos = last_left_pos, last_middle_pos, last_right_pos
                                line_list.append(current_middle_pos)
                            elif GreyLines.has_two_line(current_line_pos[0]):
                                current_left_pos, current_middle_pos, current_right_pos = GreyLines.middle_line_pos(current_line_pos[0])
                                line_list.append(current_middle_pos)
                            else:
                                #Has one line, make offset by last middle line
                                if abs(current_line_pos[0][0] - last_left_pos) <5:
                                    #The only line is left line
                                    current_left_pos = current_line_pos[0][0]
                                    current_middle_pos = last_middle_pos + (current_left_pos - last_left_pos)
                                    line_list.append(current_middle_pos)
                                else:
                                    #The only line is right line
                                    current_right_pos = current_line_pos[-1][0]
                                    current_middle_pos = last_middle_pos + (current_right_pos - last_right_pos)
                                    line_list.append(current_middle_pos)

                        break
                if lower_line < 120 and need_to_check_lower:
                    lower_row_line = line_image[lower_line]
                    lower_line_pos = np.where(lower_row_line==255)
                    if len(lower_line_pos[0]) == 0:
                        need_to_check_lower = False
                    elif GreyLines.has_two_line(lower_line_pos[0]):
                        print("Find two line row,lower_line",lower_line)
                        double_line_row = lower_line
                        last_left_pos, last_middle_pos, last_right_pos = GreyLines.middle_line_pos(lower_line_pos[0])
                        line_list.append(last_middle_pos)
                        for j in range(lower_line - line_number):
                            print("CurrentLineNumber",lower_line-j-1)
                            current_line = line_image[lower_line-j-1]
                            current_line_pos = np.where(current_line==255)
                            if len(current_line_pos[0]) == 0:
                                print("Not found any line")
                                current_left_pos, current_middle_pos, current_right_pos = last_left_pos, last_middle_pos, last_right_pos
                                line_list.append(current_middle_pos)
                            elif GreyLines.has_two_line(current_line_pos[0]):
                                print("Still has two line")
                                current_left_pos, current_middle_pos, current_right_pos = GreyLines.middle_line_pos(current_line_pos[0])
                                line_list.append(current_middle_pos)
                            else:
                                print("Has one line",current_line_pos[0][0],last_left_pos,last_right_pos)
                                #Has one line, make offset by last middle line

                                if abs(current_line_pos[0][0] - last_left_pos) - abs(current_line_pos[0][0] - last_right_pos) <0:
                                    
                                    current_right_pos = line_image.shape[1]
                                    current_left_pos = current_line_pos[0][-1]
                                    current_middle_pos = last_middle_pos + (current_left_pos - last_left_pos)
                                    line_list.append(current_middle_pos)
                                else:
                                    current_left_pos = 0
                                    current_right_pos = current_line_pos[-1][0]
                                    current_middle_pos = last_middle_pos + (current_right_pos - last_right_pos)
                                    line_list.append(current_middle_pos)
                                    #print("current_left_pos, current_middle_pos, current_right_pos",current_left_pos, current_middle_pos, current_right_pos)
                            if current_middle_pos < 0:
                                current_middle_pos = 0
                                break
                            elif current_middle_pos >= line_image.shape[1]:
                                current_middle_pos = line_image.shape[1]-1
                                break
                            last_left_pos, last_middle_pos, last_right_pos = current_left_pos, current_middle_pos, current_right_pos

            
                        break 
            if double_line_row != -1:
                double_line_pos = line_list[-1]
            else:
                #not find two line in whole image
                #double_line_pos = -1
                mean_pos = line_pos[0].mean()
                lower_line = line_number+1
                upper_line = line_number-1
                lower_row_line = line_image[lower_line]
                upper_row_line = line_image[upper_line]

                lower_line_pos = np.where(lower_row_line==255)
                upper_line_pos = np.where(upper_row_line==255)
                if len(lower_line_pos[0])!=0:
                    lower_mean_pos = lower_line_pos[0].mean()
                    if lower_mean_pos>mean_pos:
                        #turn  left
                        double_line_pos = line_pos[0][0] - 160
                    else:
                        double_line_pos = line_pos[0][0] + 160
                elif len(upper_line_pos[0])!=0:
                    upper_mean_pos = upper_line_pos[0].mean()
                    if upper_mean_pos>mean_pos:
                        #turn  right
                        double_line_pos = line_pos[0][0] + 160
                    else:
                        double_line_pos = line_pos[0][0] - 160
                else:
                    double_line_pos = -1
        return double_line_pos

    @staticmethod
    def SeeSixRoad(srcimg):
        blackwallimg = ImageProcessor.preprocess(srcimg, 0.5) # Get cropped image and convert image wall to black

        blackwallimg = cv2.medianBlur(blackwallimg, 5)
        b_image, g_image, r_image = cv2.split(blackwallimg)
        kernel = np.ones((3,3),np.uint8)

        r_image_dilate = cv2.dilate(r_image, kernel, iterations =1)
        g_image_dilate = cv2.dilate(g_image, kernel, iterations =1)

        line_image = cv2.bitwise_and(r_image_dilate,g_image_dilate)

        row_line = line_image[15]

        line_pos = np.where(row_line==255)

        if len(line_pos[0])>10:
            return True
        else:
            return False

    @staticmethod
    def CheckWrongDirectionByColor(srcimg, last_pos_list):
        #return True means wrong direction
        last_pos = np.array(last_pos_list).mean()
        if abs(last_pos-160) < 10:
            #Stright road, check if wrong way
            blackwallimg = ImageProcessor.preprocess(srcimg, 0.5) 
            blackwallimg = cv2.medianBlur(blackwallimg, 5)
            b_image, g_image, r_image = cv2.split(blackwallimg)

            r_image_pos = np.where(r_image==255)
            g_image_pos = np.where(g_image==255)

            if len(r_image_pos[0]) > 50 and len(g_image_pos[0])>50:
                if r_image_pos[1].mean() > g_image_pos[1].mean():
                    return True
                else:
                    return False

        return False
    @staticmethod
    def GetPosByRowNumber(blackwallimg, row_number,last_pos):
        b,g,r = cv2.split(blackwallimg)
        kernel = np.ones((1,5),np.uint8)
        r_dilated = cv2.dilate(r, kernel,iterations =1)
        r_close = cv2.erode(r_dilated,kernel,iterations =1)

        r_close[0:row_number] = np.zeros((row_number,320))

        label_r = measure.label(r_close)

        area_limit = 2
        label_value_list = []
        bigger_limit = 8000

        if last_pos < 1:
            last_pos = 180
        elif last_pos > 319:
            last_pos = 140

        label_props = measure.regionprops(label_r)
        for label_prop in label_props:
            print("Label",label_prop.label,"Area",label_prop.area)
            if label_prop.area < area_limit:         
                label_r[label_r == label_prop.label] = 0
            elif label_prop.area > bigger_limit:         
                label_r[label_r == label_prop.label] = 0
            else:
                label_value_list.append(label_prop.label)
        # for l in range(100):

        #     label_pos = np.where(label_r == l+1) 
        #     print("l+1",l+1,"len",len(label_pos[0]))
        #     if len(label_pos[0]) == 0:
        #         break

        #     if len(label_pos[0]) < area_limit:         
        #         label_r[label_r == l+1] = 0
        #     elif len(label_pos[0]) > bigger_limit:
        #         label_r[label_r == l+1] = 0
        #     else:
        #         label_value_list.append(l+1)

        row_image = label_r[row_number]
        conditate_area = []

        canditate_pos = []
        for l in label_value_list:
            l_pos = np.where(row_image == l)
            print(l, len(l_pos[0]),"max width",20+row_number*3)
            
            
            if len(l_pos[0]) > 2 and  len(l_pos[0]) < 20+row_number*3:
                if l_pos[0][0] == 0 or l_pos[0][-1] == 319:
                    continue
                
                label_pos = np.where( label_r == l )
                label_area = len(label_pos[0])
                if label_area > 100:
                    isMiddleLine = GreyLines.CheckLineWidth(label_r,l)
                    if isMiddleLine:
                        row_pos = l_pos[0].mean()
                        canditate_pos.append(row_pos)
                        
                else:            
                    row_pos = l_pos[0].mean()
                    canditate_pos.append(row_pos)
        print("canditate_pos",canditate_pos)
        if len(canditate_pos) == 0:
            return -1
        elif len(canditate_pos) == 1:
            return canditate_pos[0]
        else:
            # dist_last_pos = []
            # for c_pos in canditate_pos:
            #     dist_last_pos.append(abs(c_pos-last_pos))
            # candidate_idx = np.argmin(dist_last_pos)
            # return canditate_pos[candidate_idx]
            return np.array(canditate_pos).mean()
        
    
    @staticmethod
    def CheckLineWidth(label_image,label):
        label_pos = np.where( label_image == label )
        print("Cal label",label,time.time())
        if len(label_pos[0]) == 0:
            return 0
        label_area = len(label_pos[0])
        labelMask = np.zeros(label_image.shape, dtype="uint8")
        labelMask[label_image == label] = 255
        
        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        candidatecount = len(cnts[0])
        width = label_area / candidatecount
        label_mean_y = label_pos[0].mean()
        print("label_width",width,label_mean_y,time.time())
        
        if label_mean_y < 10:
            if width < 2:
                return True
            else:
                return False
        elif label_mean_y < 11:
            if width < 4:
                return True
            else:
                return False
        elif label_mean_y < 13:
            if width < 7:
                return True
            else:
                return False
        elif label_mean_y < 15:
            if width < 8:
                return True
            else:
                return False
        elif label_mean_y < 20:
            if width < 11:
                return True
            else:
                return False
        elif label_mean_y < 25:
            if width < 15:
                return True
            else:
                return False
        elif label_mean_y < 30:
            if width < 19:
                return True
            else:
                return False
        elif label_mean_y < 40:
            if width < 23:
                return True
            else:
                return False
        elif label_mean_y < 50:
            if width < 30:
                return True
            else:
                return False
        elif label_mean_y < 60:
            if width < 50:
                return True
            else:
                return False
        elif label_mean_y < 70:
            if width < 80 and width > 4:
                return True
            else:
                return False
        elif label_mean_y < 85:
            if width < 160 and width > 5:
                return True
            else:
                return False
        else:
            return False

    

    @staticmethod
    def findLineEnd(a, start_pos=0, noise_num=2):
        for i in range(start_pos, len(a) - 1):
            #print("{} delta:{}".format(i, a[i+1]-a[i]))
            if (a[i+1] - a[i]) <= noise_num:
                continue
            else:
                return i
        return len(a) - 1

    @staticmethod
    def findLine(a, start_pos=0, min_line_pixels=3):
        bFound = False
        end_pos = GreyLines.findLineEnd(a, start_pos)
        #print("findlineend:{}".format(end_pos))
        while (end_pos - start_pos + 1 <= min_line_pixels and end_pos < len(a) - 1):
            start_pos = end_pos + 1
            end_pos = GreyLines.findLineEnd(a, start_pos)
                
            if (end_pos - start_pos + 1 <= min_line_pixels):
                continue
        
        if end_pos - start_pos + 1 > min_line_pixels:
            bFound = True
        
        return (start_pos, end_pos, bFound)      


    @staticmethod
    def checkObstacle(raw_img, start_col=5, end_col=315, obs_min_width=10, obs_max_width=150):
        gray_image = rgb2grey(raw_img)

        #print(gray_image)
        #ShowGrayImage(gray_image)

        #print(gray_image)
        black_img = gray_image < 0.05
        #print(b)
        #ShowGrayImage(black_img)   

        #label_b = measure.label(black_img)
        revised_black_img = morphology.remove_small_holes(black_img, 16)
        #ShowGrayImage(revised_black_img)

        #skeleton = morphology.skeletonize(black_img)
        #ShowGrayImage(skeleton)
            

        check_img = revised_black_img


        #revised_label_b = morphology.remove_small_holes(label_b, 9)

        #ShowGrayImage(revised_label_b)

        check_start_row = 100
        check_end_row = 150

        obs_lefts = []

        if (raw_img.shape[0] < 150):
            check_start_row = 10
            check_end_row = 50

        for x in range(check_start_row, check_end_row): 
            
            black_arrays = np.where(check_img[x,start_col:end_col] == 0)
            
            curr_gray_row = np.asarray(gray_image[x,start_col:end_col])
            
            # red pixels: 0.2 ~ 0.3
            # blue pixels: 0.5 ~ 0.8
            # black pixels: < 0.05
            
            blue_pixels = ( (curr_gray_row > 0.5) & (curr_gray_row < 0.8) ).sum()
            
            
            #print("col:{}, black_arrays:{}, blue_arrays:{}".format(x, len(black_arrays), blue_pixels ) )
            
            if len(black_arrays[0]) > 10 and blue_pixels < 5:
                
                
                check_lines = black_arrays[0]
                obs_left = -1
                obs_right = -1
                
            # print("rows:{}".format(x))
            # print(check_lines)
                
                start_pos = 0
                first_start_pos, first_end_pos, bFound = GreyLines.findLine(check_lines, start_pos)
                #print("first_start:{}, first_end:{}".format(first_start_pos, first_end_pos))
                if bFound:
                
                    second_start_pos, second_end_pos, bFound = GreyLines.findLine(check_lines, first_end_pos + 1)
                    #print("second_start:{}, second_end:{}".format(second_start_pos, second_end_pos))
                    if bFound:
                        third_start_pos, third_end_pos, bFound = GreyLines.findLine(check_lines, second_end_pos + 1)
                        #print("third_start:{}, third_end:{}".format(third_start_pos, third_end_pos))
                        if bFound:
                            obs_left = check_lines[second_start_pos]
                            obs_right = check_lines[second_end_pos]
                            if (obs_right - obs_left >= obs_min_width  and obs_right - obs_left <= obs_max_width):
                                #print("col: {}, left:{}, right:{}".format(x, obs_left, obs_right))
                                obs_lefts.append(obs_left)
                                #print(gray_image[x,obs_left:obs_right])
                                #print(check_lines)
                                
                                #print(gray_image[x,:])


        #print("obs:{}".format(len(obs_lefts)))
        if len(obs_lefts) >= 3:
            return np.median(obs_lefts), True
        else:
            return -1, False


    @staticmethod
    def black_rgb(img):
        b,g,r = cv2.split(img)
        black_filter = ((r < 60) & (g <60) & (b < 60))
        r[black_filter], r[np.invert(black_filter)], g[black_filter],g[np.invert(black_filter)], b[black_filter],b[np.invert(black_filter)] = 255,0,255,0,255,0
        flattened = cv2.merge((r, g, b))
        
        flattened_gray = cv2.cvtColor(flattened,cv2.COLOR_BGR2GRAY)
            
        return flattened_gray
    
    @staticmethod
    def CheckSide(black_gray):
        black_sum = black_gray.sum(axis=0).astype(int)
        black_sum_diff = abs(np.diff(black_sum))
        black_sum_diff[black_sum_diff<1000] = 0
        
        black_sum_pos = np.where(black_sum_diff > 0)

        

        #print(len(black_sum_diff),black_sum_diff) 
        # middle_pos = -1
        # if len(black_sum_pos[0]) == 0:
        #     return middle_pos

        left_sum = black_sum[0:160].sum()
        right_sum = black_sum[160:320].sum()

        diff = abs(left_sum - right_sum)
        
        print("diff:".format(diff))

        if left_sum > right_sum:
            return 300, diff
        else:
            return 40, diff
    @staticmethod
    def GetTwoLinePosEx(lineImage,traceRow):
        lineRow = lineImage[traceRow]
        lineRowPos = np.where(lineRow == 255)
        if len(lineRowPos[0]) == 0:
            return -1
        else:
            return lineRowPos[0].mean()
    @staticmethod
    def GetOptimizedPos(srcimg, pre_pos,middle_number,trace_road):
        black_src = GreyLines.black_rgb(srcimg)
        print("Optimize pre_pos",pre_pos)
        op_pos = pre_pos
        # black_temp = None
        # if pre_pos < 160:
        #     black_temp = black_src[int(2*middle_number):,:160]
        # else:
        #     black_temp = black_src[int(2*middle_number):,160:]
        if trace_road == 5:
            offset_flag = 2
            black_temp = black_src[int(2*middle_number):]
        else:
            offset_flag = 2
            black_temp = black_src[int(3*middle_number):]
        black_temp = black_src[int(2*middle_number):]
        black_temp_sum = black_temp.sum(axis=0).astype(int)
        #print("black_temp_sum",black_temp_sum)
        black_sum = black_temp_sum[black_temp_sum>500]
        #cv2.imwrite("../TestImage5/srcimg_"+str(time.time())+"_black_src_"+".bmp", black_src) 
        #cv2.imwrite("../TestImage5/srcimg_"+str(time.time())+"_black_tmp_"+".bmp", black_temp) 

        # if trace_road == 5:
        #     offset_flag = 2
        # else:
        #     offset_flag = 1

        if len(black_sum) > 0:
            

            black_temp_sum_pos = np.where(black_temp_sum>500)
            black_mean = black_temp_sum_pos[0].mean()#2*black_sum.mean()/255
            print("black_mean",black_mean)
            if black_mean < 160:
                op_pos =  pre_pos + offset_flag*black_mean
                if op_pos < (320 + offset_flag*black_mean)/2:
                    op_pos = (320 + offset_flag*black_mean)/2
            else:
                op_pos =  pre_pos - offset_flag*(320-black_mean)
                if op_pos > (320 - offset_flag*(320-black_mean))/2:
                    op_pos = (320 - offset_flag*(320-black_mean))/2
        if op_pos < 1 :
            op_pos = 1
        elif op_pos > 319:
            op_pos = 319
        
        return op_pos

    @staticmethod
    def GetWhiteRoadPos(srcimg):
        blackimg = GreyLines.black_rgb(srcimg)
        # for row in blackimg[:10]:
        #     black_len = len(row[row==255])
        #     if len(row) == black_len:
        #         return -1
        blackSum = blackimg.sum(axis=0).astype(int)
        #print(len(blackSum),blackSum)
        blackSum[blackSum > 600] = 0
        blackSum[blackSum>0] = 255
        #blackSum = 255 - blackSum
        #print("====",len(blackSum),blackSum)
        

        #blackimg = blackimg[0]
        kernel = np.ones((1,5),np.uint8)
        #blackimg_dilated = cv2.dilate(blackSum, kernel,iterations =1)
        blackimg_close = blackSum#cv2.erode(blackimg_dilated,kernel,iterations =1)
        # blackimg_close[blackimg_close>50] = 255
        # blackimg_close[blackimg_close<=50] = 0
        # blackimg_close = 255 - blackimg_close


        #

        blackimg_label = measure.label(blackimg_close)

        label_list = []
        label_area_list = []
        label_pos_list = []
        for l in range(100):
            l_pos = np.where(blackimg_label == l+1)
            if len(l_pos[0]) == 0:
                break
            if l_pos[0][0] == 0 or l_pos[0][-1] == 319:
                continue
            label_list.append(l+1)
            label_area_list.append(len(l_pos[0]))
            label_pos_list.append(l_pos[0].mean())
        if len(label_list) == 0:
            return -1
        elif len(label_list) == 1:
            return label_pos_list[0]
        else:
            l_idx = np.argmax(label_area_list)
            return label_pos_list[l_idx]

        # label_props = measure.regionprops(blackimg_label)

        # if len(label_props) == 0:
        #     return -1
        # elif len(label_props) == 1:
        #     return label_props[0].centroid[1]
        # else:
        #     max_area = 0
        #     max_area_label = -1
        #     for label_prop in label_props:
        #         if label_prop.area > max_area:
        #             max_area = label_prop.area
        #             max_area_label = label_prop.label
        #     return label_props[max_area_label].centroid[1]


    @staticmethod
    def GetCarPos(srcimg, last_pos_list, trace_road, current_speed, last_idx, isChangingRoad,changelane_direction):
        #cv2.imwrite("TestImage/srcimg_"+str(time.time())+".bmp", srcimg[120:]) 
        blackwallimg = ImageProcessor._flatten_rgb_old(srcimg[120:])
        #cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+".bmp", blackwallimg) 
        middle_position =160

        last_pos = np.array(last_pos_list).mean()

      
        if trace_road == 5:
            
            

            car_left_pos, car_right_pos, car_top_pos, car_bottom_pos, bFoundCar = GreyLines.checkObstacle_v2(srcimg)
            
            if bFoundCar:
                #print("last_pos:{}".format(last_pos_list[last_idx]))
                prev_turn_pos = last_pos_list[last_idx]
                car_mid_pos = car_left_pos + (car_right_pos - car_left_pos)/2

                flattened_gray = GreyLines.black_rgb(srcimg[120:])
                pos, diff = GreyLines.CheckSide(flattened_gray)
                
                curr_time = time.time()
                time_str = str(curr_time)
               
                if GreyLines.prev_obs_detect_time > 0:
                    time_diff = (curr_time - GreyLines.prev_obs_detect_time) * 1000
                    turn_time = 400
                    if time_diff < turn_time: #500 ms to turn around the car
                        curr_pos = round( GreyLines.prev_obs_turn_start_pos + (160 - GreyLines.prev_obs_turn_start_pos) * time_diff / turn_time, 1)
        
                        #print("car_pos:{}-{}, prev_turn:{}, curr_turn_pos:{}, diff:{}".format(car_left_pos, car_right_pos, last_pos_list[last_idx], curr_pos, diff))
                        logger.info("car_pos:{}-{}, prev_turn:{}, curr_turn_pos:{}, diff:{}".format(car_left_pos, car_right_pos, prev_turn_pos, curr_pos, diff))
            
                        #cv2.imwrite("/log/TestImage/Car/srcimg_"+time_str+"_"+str(car_left_pos)+"-"+str(car_right_pos) +"_"+ str(last_pos_list[last_idx]) + "_" + str(curr_pos)+"_"+ str(diff) + ".bmp", srcimg)
                        return curr_pos, bFoundCar,False,0

                
                GreyLines.prev_obs_detect_time = curr_time
                
                obs_width = car_right_pos - car_left_pos
                if obs_width in range (100, 150):
                    gear_gap = 40

                
               
                new_turn_pos = prev_turn_pos

                if prev_turn_pos < car_mid_pos:
                    pos_diff = abs(car_mid_pos - prev_turn_pos)
                    if pos_diff >= 140:
                        new_turn_pos = prev_turn_pos # no change
                    else:
                        if car_mid_pos < 120 + 40:
                            new_turn_pos = 320
                        else:
                            if pos_diff < 40:
                                new_turn_pos = prev_turn_pos - 100
                                if new_turn_pos<160 :
                                    new_turn_pos=new_turn_pos-40
                            else:
                                new_turn_pos = prev_turn_pos - 40
                                if new_turn_pos<160 :
                                    new_turn_pos=new_turn_pos-20
                            if new_turn_pos < 0:
                                new_turn_pos = 0
                else:
                    pos_diff = abs(prev_turn_pos - car_mid_pos)
                    if pos_diff >= 120:
                        new_turn_pos = prev_turn_pos # no change
                    else:
                        if car_mid_pos > 200 + 40:
                            new_turn_pos = 0
                        else:
                            if pos_diff < 40:
                                new_turn_pos = prev_turn_pos + 100
                                if new_turn_pos>=160 :
                                    new_turn_pos=new_turn_pos+20
                            else:
                                new_turn_pos = prev_turn_pos + 40
                                if new_turn_pos>=160 :
                                    new_turn_pos=new_turn_pos+10
                            if new_turn_pos > 320:
                                new_turn_pos = 320

                #cv2.imwrite("/log/TestImage/Car/srcimg_"+time_str+"_"+str(car_left_pos)+"-"+str(car_right_pos) +"_"+ str(prev_turn_pos) + "_" + str(new_turn_pos)+"_"+ str(diff) + ".bmp", srcimg)
                #print("car_pos:{}-{}, prev_turn:{}, turn_pos:{}, diff:{}".format(car_left_pos, car_right_pos, prev_turn_pos, new_turn_pos, diff))
                logger.info("=> car_pos:{}-{}, prev_turn:{} turn_pos:{}, diff:{}".format(car_left_pos, car_right_pos,prev_turn_pos , new_turn_pos, diff))
            
                GreyLines.prev_obs_turn_start_pos = new_turn_pos
                return new_turn_pos, bFoundCar,False,0


            middle_trace_number = 9
            if current_speed <0.5:
                middle_trace_number = middle_trace_number+10
            elif current_speed <1.0:
                middle_trace_number = middle_trace_number+5
            elif current_speed <1.4:
                middle_trace_number = middle_trace_number+4
            elif current_speed <1.5:
                middle_trace_number = middle_trace_number+2
            elif current_speed <1.6:
                middle_trace_number = middle_trace_number+1
            elif current_speed <1.8:
                middle_trace_number = middle_trace_number
            elif current_speed <1.9:
                middle_trace_number = middle_trace_number
            elif current_speed <1.98:
                middle_trace_number = middle_trace_number
            pos = GreyLines.GetPosByRowNumber(blackwallimg,middle_trace_number,last_pos_list[last_idx]) 
            if pos != -1:
                pos = GreyLines.GetOptimizedPos(srcimg[120:],pos,middle_trace_number,trace_road)
                print("Optimized pos",pos)
            if pos != -1:
                #srcimg[120:][middle_trace_number] = srcimg[120:][middle_trace_number] + 20
                #cv2.imwrite("TestImage1/srcimg_"+str(time.time())+"_"+str(pos)+".bmp", srcimg[120:])
                # if pos < 60:
                #     pos = GreyLines.GetPosByRowNumber(blackwallimg,middle_trace_number+8,last_pos_list[last_idx]) 
                #if current_speed > 0.1:
                    #cv2.imwrite("/Volumes/jeffrey/TestImage5/srcimg_"+str(time.time())+"_"+str(int(pos))+".bmp", srcimg) 
                    #print time.time(), "Store Image"
                    #cv2.line(srcimg,(0,130),(319,130),(255,0,0))
                    #cv2.imwrite("../TestImage5/srcimg_"+str(time.time())+"_"+str(int(pos))+".bmp", srcimg) 

                return pos, False,False,0
            else:
                # if last_pos > 160:
                #     cv2.imwrite("TestImage2/2/srcimg_"+str(time.time())+"_"+str(320)+".bmp", srcimg[120:])
                #     return 320, False
                # else: 
                #     cv2.imwrite("TestImage2/1/srcimg_"+str(time.time())+"_"+str(0)+".bmp", srcimg[120:])
                #     return 0, False
                # whitePos = GreyLines.GetWhiteRoadPos(srcimg[120:])
                # if whitePos != -1:
                #     cv2.imwrite("../TestImage6/srcimg_"+str(time.time())+"_"+str(whitePos)+".bmp", srcimg[120:])
                #     whitePos = GreyLines.GetOptimizedPos(srcimg[120:],whitePos,9,trace_road)

                #     return whitePos, False,False,0

                input_image = cv2.resize(srcimg[120:],(80,30),interpolation=cv2.INTER_CUBIC)
                input_data = input_image[np.newaxis,:,:,:]

                direction_prob = lostLine_model.predict(input_data)
                direction = np.argmax(direction_prob)
                logit("direction_prob"+str(direction_prob))
                if direction == 0:
                    #cv2.imwrite("lostLine/1/blackwallimg_"+str(time.time())+"_"+str(-1)+"_"+str(0)+".bmp", blackwallimg)          
                    # if current_speed >0.5:
                    #     cv2.imwrite("../TestImage4/1/srcimg_"+str(time.time())+".bmp", srcimg[120:])
                    # pos = GreyLines.GetPosByRowNumber(blackwallimg,middle_trace_number+8,last_pos_list[last_idx]) 

                    # if pos == -1:
                    #     return 0,False
                    # else:
                    #     return pos, False
                    if current_speed > 0.5:
                        # cv2.imwrite("/Volumes/jeffrey/TestImage5/srcimg_"+str(time.time())+"_"+str(int(pos))+".bmp", srcimg) 
                        # print time.time(), "Store Image model 0"
                        # cv2.line(srcimg,(0,130),(319,130),(255,0,0))
                        cv2.imwrite("../TestImage44/1/srcimg_"+str(time.time())+"_"+str(int(0))+".bmp", srcimg[120:]) 

                    return 0,False,False,0
                else:
                    #cv2.imwrite("lostLine/2/blackwallimg_"+str(time.time())+"_"+str(-1)+"_"+str(320)+".bmp", blackwallimg)          
                    # if current_speed >0.5:
                    #     cv2.imwrite("../TestImage4/2/srcimg_"+str(time.time())+".bmp", srcimg[120:]) 
                    if current_speed > 0.5:
                        # cv2.imwrite("/Volumes/jeffrey/TestImage5/srcimg_"+str(time.time())+"_"+str(int(pos))+".bmp", srcimg) 
                        # print time.time(), "Store Image model 320"
                        # cv2.line(srcimg,(0,130),(319,130),(255,0,0))
                        # cv2.imwrite("../TestImage5/srcimg_"+str(time.time())+"_"+str(int(320))+".bmp", srcimg) 
                        cv2.imwrite("../TestImage44/2/srcimg_"+str(time.time())+"_"+str(int(0))+".bmp", srcimg[120:]) 

                    return 320,False,False,0
        else:
            trace_row = 14
            black_row = 18
            if current_speed <0.5:
                trace_row = 60
            elif current_speed <1.0:
                trace_row = trace_row+15
            elif current_speed <1.4:
                trace_row = trace_row+5
            elif current_speed <1.5:
                trace_row = trace_row+4
            elif current_speed <1.6:
                trace_row = trace_row+3
            elif current_speed <1.8:
                trace_row = trace_row+2
            elif current_speed <1.9:
                trace_row = trace_row+1
            elif current_speed <1.98:
                trace_row = trace_row
            # if current_speed <0.5:
            #     trace_row = 60
            # elif current_speed <1.0:
            #     trace_row = 30
            # elif current_speed <1.2:
            #     trace_row = 15
            # elif current_speed <1.4:
            #     trace_row = 14
            # elif current_speed <1.6:
            #     trace_row = 14
            # elif current_speed <1.8:
            #     trace_row = 14
            # elif current_speed <1.9:
            #     trace_row = 15
            # elif current_speed <1.98:
            #     trace_row = 14
            
            blackwallimg = ImageProcessor.preprocess(srcimg[120:], 0.5) # Get cropped image and convert image wall to black

            #blackwallimg = cv2.medianBlur(blackwallimg, 5)

            b_image, g_image, r_image = cv2.split(blackwallimg)

            line_image,isSeeSixRoad,label_number = GreyLines.GetTwoLineImage(srcimg[120:],isChangingRoad,changelane_direction)
            


            row_line = line_image[trace_row]

            row_line_pos = np.where(row_line==255)
            black_b = np.where(b_image[black_row]==255)
            black_g = np.where(g_image[black_row]==255)
            black_r = np.where(r_image[black_row]==255)
            isNearWall = False
            if len(black_b[0]) == 0 and\
                len(black_g[0]) == 0 and\
                len(black_r[0]) == 0 and len(row_line_pos[0]) == 0:
                isNearWall = True

            car_pos = GreyLines.GetTwoLinePosEx(line_image, trace_row)

            # car_pos = GreyLines.GetTwoLinePos(line_image, trace_row)
            # print("car_pos",car_pos)
            print("car_pos14",car_pos)
            if car_pos!= -1 and car_pos < 150 and trace_row == 14:
                car_pos = GreyLines.GetTwoLinePosEx(line_image, trace_row+1)
                print("car_pos15",car_pos)
            # if car_pos != -1:
            #     car_pos = GreyLines.GetOptimizedPos(srcimg[120:],car_pos,trace_row, trace_road)
            #     print("Optimized pos",car_pos)
            
            # cv2.imwrite("../TestImage5/srcimg_"+str(time.time())+"_"+str(int(car_pos))+".bmp", srcimg) 
            # cv2.imwrite("../TestImage5/srcimg_"+str(time.time())+"_Line_"+str(int(car_pos))+".bmp", line_image)
            #print("two line car_pos is ",car_pos)
            #cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+".bmp", line_image)
            if car_pos == -1:
                input_image = cv2.resize(srcimg[120:],(80,30),interpolation=cv2.INTER_CUBIC)
                input_data = input_image[np.newaxis,:,:,:]

                direction_prob = lostLine_model.predict(input_data)
                direction = np.argmax(direction_prob)
                logit("direction_prob"+str(direction_prob))
                if direction == 0:
                    # if current_speed > 0.5:
                    #     cv2.imwrite("../TestImage7/srcimg_"+str(time.time())+"_"+str(int(0))+".bmp", srcimg[120:]) 
                    return 0,isNearWall,isSeeSixRoad,label_number
                else:
                    # if current_speed > 0.5:
                    #     cv2.imwrite("../TestImage7/srcimg_"+str(time.time())+"_"+str(int(320))+".bmp", srcimg[120:]) 
                    return 320,isNearWall,isSeeSixRoad,label_number
            if car_pos>320:
                car_pos = 320
            elif car_pos <0:
                car_pos = 0
            # if current_speed > 0.5:
            #     cv2.imwrite("../TestImage7/srcimg_"+str(time.time())+"_"+str(int(car_pos))+".bmp", srcimg[120:]) 
                    
            if car_pos != -1:
                pre_car_pos = car_pos
                car_pos = GreyLines.GetOptimizedPos(srcimg[120:],car_pos,trace_row, trace_road)
                
            return car_pos,isNearWall,isSeeSixRoad,label_number
        # if pos != -1 and xy != [0,0]:
        #     #srcimg[120:][middle_trace_number] = srcimg[120:][middle_trace_number] + 20
        #     cv2.imwrite("TestImage/srcimg_"+str(time.time())+"_"+str(xy[0])+".bmp", srcimg[120:])
        #     return xy[0], False
        # else:
        #     if last_pos > 160:
        #         cv2.imwrite("TestImage/srcimg_"+str(time.time())+"_"+str(320)+".bmp", srcimg[120:])
        #         return 320, False
        #     else: 
        #         cv2.imwrite("TestImage/srcimg_"+str(time.time())+"_"+str(0)+".bmp", srcimg[120:])
        #         return 0, False
            # input_image = cv2.resize(srcimg[120:],(80,30),interpolation=cv2.INTER_CUBIC)
            # input_data = input_image[np.newaxis,:,:,:]

            # direction_prob = lostLine_model.predict(input_data)
            # direction = np.argmax(direction_prob)
            # logit("direction_prob"+str(direction_prob))
            # if direction == 0:
            #     #cv2.imwrite("lostLine/1/blackwallimg_"+str(time.time())+"_"+str(-1)+"_"+str(0)+".bmp", blackwallimg)          
            #     cv2.imwrite("TestImage/1/srcimg_"+str(time.time())+".bmp", srcimg[120:])
            #     return 0,False
            # else:
            #     #cv2.imwrite("lostLine/2/blackwallimg_"+str(time.time())+"_"+str(-1)+"_"+str(320)+".bmp", blackwallimg)          
            #     cv2.imwrite("TestImage/2/srcimg_"+str(time.time())+".bmp", srcimg[120:]) 
            #     return 320,False
    @staticmethod
    def GetMiddleLinePos(mask):
        # blackwallimg = ImageProcessor._flatten_rgb_old(src_image)
        # b,g,r = cv2.split(blackwallimg)
        # r[0:15] = np.zeros((15,320))
        
        # label_r = measure.label(r)
        # mask = morphology.remove_small_objects(label_r, 50)
        label_props = measure.regionprops(mask)
        for label_prop in label_props:
            print("label_area",label_prop.area,"label",label_prop.label)
            if label_prop.area > 8000:
                continue
            isMiddle = GreyLines.CheckLineWidth(mask,label_prop.label)
            if isMiddle:
                print("label_prop.centroid",label_prop.centroid)
                return label_prop.centroid[1],label_prop.label
        return -1,-1





    @staticmethod
    def GetTwoLineImage(src_image,isChangingRoad,changelane_direction):
        flatten_image = ImageProcessor._flatten_rgb_old(src_image)
        
        kernel = np.ones((5,11),np.uint8)
        
        
        b,g,r = cv2.split(flatten_image)

        black_image = GreyLines.black_rgb(src_image)       
        black_image = cv2.dilate(black_image, kernel,iterations =1)
        black_image = 255 - black_image
        

        r = cv2.bitwise_and(r,black_image)
        # # black_image[black_image>250] = 255
        # # black_image[black_image<251] = 0

        # r = cv2.dilate(r,np.ones((1,3),np.uint8), kernel,iterations =1)

        # cv2.imwrite("../TestImage5/srcimg_"+str(time.time())+"_black_image_"+".bmp", black_image)
        # cv2.imwrite("../TestImage5/srcimg_"+str(time.time())+"_r_"+".bmp", r)
        # cv2.imwrite("../TestImage5/srcimg_"+str(time.time())+"_src_"+".bmp", src_image)
        # print("r",r[0])
        # print("black_image",black_image[0])
        # print("r",r[20])
        # print("black_image",black_image.astype(int)[20])
        # print("r.shape",r.shape,"black_image.shape",black_image.shape)
        # r = cv2.bitwise_and(r,black_image)

        
        label_r = measure.label(r)

        l_pos_list = []
        l_list = []
        l_area_list = []
        isSeeSixRoad = False

        label_number = 0
        
        
        

            # if len(l_pos_list)>0:
            #     if changelane_direction == 1:
            #         l_idx = np.argmax(l_pos_list)
            #         label_r[label_r!=l_idx] = 0
            #     else:
            #         l_idx = np.argmin(l_pos_list)
            #         label_r[label_r!=l_idx] = 0]
        mask = morphology.remove_small_objects(label_r, 40)
        # if isChangingRoad:
        #     mask = morphology.remove_small_objects(label_r, 40)
        # else:
        #     mask = morphology.remove_small_objects(label_r, 450)
        #     mask_middle = morphology.remove_small_objects(label_r, 40)
        if isChangingRoad:
            middle_pos,middle_label = GreyLines.GetMiddleLinePos(mask)
            print("middle_pos",middle_pos,middle_label)
            label_props = measure.regionprops(mask)
            label_number = len(label_props)
            if middle_pos != -1:
                #return np.zeros(r.shape),False,label_number
            
                mask[mask == middle_label] = 0
            
            for label_prop in label_props:
                if label_prop.label == middle_label:
                    continue
                # if changelane_direction == 1 and label_prop.centroid[1]<middle_pos:
                #     mask[mask==label_prop.label] = 0
                #     continue
                # if changelane_direction == -1 and label_prop.centroid[1]>middle_pos:
                #     mask[mask==label_prop.label] = 0
                #     continue
                l_pos_list.append(label_prop.centroid[1])
                l_list.append(label_prop.label)
                l_area_list.append(label_prop.area)
            # for l in range(100):
            #     if l+1 == middle_label:
            #         continue
            #     l_pos = np.where(mask == l+1)
            #     if len(l_pos[1]) > 0:
            #         l_pos_list.append(l_pos[1].mean())
            #         l_list.append(l+1)
            #         # l_pos_mean = l_pos[1].mean()
            #         # if l_pos_mean > middle_pos:
            #         #     l_list.append(l+1)
            #         #     l_pos_list.append(l_pos_mean)
            #     else:
            #         break

            print("l_pos_list",l_pos_list)
            if len(l_pos_list) == 0:
                return np.zeros(r.shape),False, 0
            else:
                if changelane_direction == 1:
                    l_idx = np.argmax(l_pos_list)
                    mask[mask!=l_list[l_idx]] = 0
                    if l_area_list[l_idx] > 10000:
                        label_number = 1
                else:
                    l_idx = np.argmin(l_pos_list)
                    mask[mask!=l_list[l_idx]] = 0
                    if l_area_list[l_idx] > 10000:
                        label_number = 1

        else:
            middle_pos,middle_label = GreyLines.GetMiddleLinePos(mask)
            label_num = measure.regionprops(mask)
            label_number = len(label_num)
            print("---------------middle_pos",middle_pos,middle_label,len(label_num))
            if middle_pos != -1 and len(label_num) > 1:
                #cv2.imwrite("../TestImage6/srcimg_"+str(time.time())+"_Middle_"+str(int(len(label_num)))+".bmp", src_image)
                isSeeSixRoad = True
        area_list = []
        area_list_label = []
        # for l in range(10):
        #     l_pos = np.where(mask==l+1)
        #     if len(l_pos[0]) == 0:
        #         break
        #     area_list.append(len(l_pos[0]))
        label_props = measure.regionprops(mask)
        for label_prop in label_props:
            area_list.append(label_prop.area)
            area_list_label.append(label_prop.label)
        print("area_list",area_list,area_list_label,"label_num",label_number)
        if len(area_list) != 0:
            max_area_label = np.argmax(area_list)
            mask[mask!=area_list_label[max_area_label]] =0

        label_bw = mask.copy()
        label_bw[label_bw != 0] = 255

        
                

        # kernel = np.ones((9,9),np.uint8)
        # r_dilated = cv2.dilate(label_bw.astype(float), kernel,iterations =1)
        # r_close = cv2.erode(r_dilated, kernel,iterations =1)

        kernel = np.ones((3,3),np.uint8)
        r_dilated = cv2.dilate(label_bw.astype(float), kernel,iterations =1)
        r_close = cv2.erode(r_dilated, kernel,iterations =1)
        

        r_edge = cv2.bitwise_xor(r_dilated, r_close)

        r_edge[0:9] = np.zeros((9,320))

        #cv2.imwrite("../TestImage7/srcimg_"+str(time.time())+"_label_bw_"+".bmp", label_bw)
        return label_bw,isSeeSixRoad,label_number
    @staticmethod
    def checkObstacle_v2(ori_img, start_row=100, end_row=150, start_col=0, end_col=320, obs_min_width=30, obs_max_width=150, obs_lines_threshold = 4):
        bFound = False
        left_pos = -1
        right_pos = -1
        top_pos = -1
        bottom_pos = -1
        

        check_img = ori_img[start_row:end_row, start_col:end_col]
        #ShowGrayImage(check_img)
        #print("check size:{}".format(check_img.shape))
        
        b,g,r = cv2.split(check_img)

        obstaclemask = (r>15) & (g>15) & (b>15) & (r<90) & (b<90) & (g<90) 
        
        
        obstaclemask = morphology.remove_small_objects(obstaclemask, 4)
        
        
        line_left = []
        line_right = []
        line_row = []
        
        for x in range(start_row, end_row):
            car_index = np.where(obstaclemask[(x-start_row), :] == 1)[0]
            start_pos = 0
            end_pos = len(car_index)
    #         for i in range(0, len(car_index) - 1):
    #             if car_index[i+1] - car_index[i] >= 10:
    #                 start_pos = i + 1
    #             else:
    #                 break
    #         for i in range(len(car_index) - 1, 0, -1):
    #             if car_index[i] - car_index[i-1] >= 10:
    #                 end_pos = i
    #             else:
    #                 break
            
            reversed_car_index = car_index[start_pos:end_pos]
                
            if len(reversed_car_index) >= 4:   # at least 4 pixels
                #print("row:{} width:{} - {}".format(x, car_index[-1], car_index[0]))
                #print("row:{} - {}".format(x, reversed_car_index))
                width = reversed_car_index[-1] - reversed_car_index[0] + 1
                if width >= obs_min_width and width <= obs_max_width:
                    line_left.append(reversed_car_index[0])
                    line_right.append(reversed_car_index[-1])
                    line_row.append(x)
        
        if len(line_left) >= obs_lines_threshold:
            right_pos = round(np.average(line_right),1) + start_col #np.average(np.array(line_right)[np.argpartition(line_right, -3)[-3:]]) + start_col
            left_pos = round(np.average(line_left),1) + start_col #np.average(np.array(line_left)[np.argpartition(line_left, 3)[:3]]) + start_col
            width = right_pos - left_pos + 1
            top_pos = line_row[0]
            bottom_pos = line_row[-1]
            if width >= obs_min_width and width <= obs_max_width:
                bFound = True
        
            
        return left_pos, right_pos, top_pos, bottom_pos, bFound
   
    @staticmethod
    def GetCarPos_old(srcimg, last_pos_list, trace_road, current_speed):
        #Road color is[Red, Blue,  Red, Green, Blue, Green]
        #trace_road is[ 0  1  2  3  4  5  6   7  8  9  10 ]
        last_pos = np.array(last_pos_list).mean()
        blackwallimg = ImageProcessor.preprocess(srcimg, 0.5) # Get cropped image and convert image wall to black

        blackwallimg = cv2.medianBlur(blackwallimg, 5)

        #cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+".bmp", blackwallimg) 
        b_image, g_image, r_image = cv2.split(blackwallimg)

        line_image = np.zeros(b_image.shape)
        middle_position = b_image.shape[1]/2
        isNearWall = False
        middle_trace_number = 10
        if current_speed > 1.4:
            middle_trace_number = middle_trace_number
        elif current_speed > 1.0:
            middle_trace_number = middle_trace_number+2
        elif current_speed > 0.5:
            middle_trace_number = middle_trace_number+4
        else:
            middle_trace_number = middle_trace_number+6
            
        if trace_road == 5:
            #print("trace_road 5")
            
            kernel = np.ones((5,5),np.uint8)
            r_image_dilate = cv2.dilate(r_image, kernel, iterations =1)
            g_image_dilate = cv2.dilate(g_image, kernel, iterations =1)

            line_image = cv2.bitwise_and(r_image_dilate,g_image_dilate)

            row_line = line_image[middle_trace_number]
        elif trace_road == 2:
            trace_row = 13
            black_row = 18
            if current_speed <0.5:
                trace_row = 60
            elif current_speed <1.0:
                trace_row = 30
            elif current_speed <1.4:
                trace_row = 20
            elif current_speed <1.5:
                trace_row = 18
            elif current_speed <1.6:
                trace_row = 17
            elif current_speed <1.8:
                trace_row = 16
            elif current_speed <1.9:
                trace_row = 15
            elif current_speed <1.98:
                trace_row = 14
            #print("trace_road 2")
            kernel = np.ones((3,3),np.uint8)
            r_image_dilate = cv2.dilate(r_image, kernel, iterations =1)
            b_image_dilate = cv2.dilate(b_image, kernel, iterations =1)
            line_image = cv2.bitwise_and(r_image_dilate,b_image_dilate)
            row_line = line_image[trace_row]

            row_line_pos = np.where(row_line==255)
            black_b = np.where(b_image[black_row]==255)
            black_g = np.where(g_image[black_row]==255)
            black_r = np.where(r_image[black_row]==255)
            
            if len(black_b[0]) == 0 and\
                len(black_g[0]) == 0 and\
                len(black_r[0]) == 0 and len(row_line_pos[0]) == 0:
                isNearWall = True

            car_pos = GreyLines.GetTwoLinePos(line_image, trace_row)
            #print("two line car_pos is ",car_pos)
            #cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+".bmp", line_image)
            if car_pos == -1:
                if last_pos < middle_position:
                    return 0,isNearWall
                else:
                    return line_image.shape[1],isNearWall
            return car_pos,isNearWall
        elif trace_road == 8:
            trace_row = 13
            black_row = 18
            if current_speed <0.5:
                trace_row = 60
            elif current_speed <1.0:
                trace_row = 30
            elif current_speed <1.3:
                trace_row = 20
            elif current_speed <1.5:
                trace_row = 18
            elif current_speed <1.6:
                trace_row = 17
            elif current_speed <1.8:
                trace_row = 16
            elif current_speed <1.9:
                trace_row = 15
            elif current_speed <1.98:
                trace_row = 14
            #print("trace_road 8")
            kernel = np.ones((3,3),np.uint8)
            g_image_dilate = cv2.dilate(g_image, kernel, iterations =1)
            b_image_dilate = cv2.dilate(b_image, kernel, iterations =1)
            line_image = cv2.bitwise_and(g_image_dilate,b_image_dilate)
            row_line = line_image[trace_row]

            row_line_pos = np.where(row_line==255)
            black_b = np.where(b_image[black_row]==255)
            black_g = np.where(g_image[black_row]==255)
            black_r = np.where(r_image[black_row]==255)
            
            if len(black_b[0]) == 0 and\
                len(black_g[0]) == 0 and\
                len(black_r[0]) == 0 and len(row_line_pos[0]) == 0:
                isNearWall = True

            car_pos = GreyLines.GetTwoLinePos(line_image, trace_row)
            #cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+".bmp", line_image)
            #print("two line car_pos is ",car_pos)
            #cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+str(car_pos)+".bmp", blackwallimg)
            
            if car_pos == -1:
                if last_pos < middle_position:
                    return 0,isNearWall
                else:
                    return line_image.shape[1],isNearWall
            
            return car_pos,isNearWall
            
            # print("len(row_line_pos[0])",len(row_line_pos[0]))
            # if len(row_line_pos[0])>80:
            #     if row_line_pos[0].mean() < middle_position:
            #         return 0,isNearWall
            #     else:
            #         return line_image.shape[1],isNearWall

            # if (len(row_line_pos[0])!=0 and GreyLines.has_two_line(row_line_pos[0]) == False):
            #     print("One line")
            #     isRight = False
            #     for g_row in g_image[::-1]:
            #         g_row_idx = np.where(g_row==255)

            #         print(g_row_idx)
            #         if len(g_row_idx[0]) >2:
            #             if g_row_idx[0][0] < 160:
            #                 isRight = True
            #             else:
            #                 isRight = False
            #             break

            #     if isRight:
            #         return line_image.shape[1],isNearWall
            #     else:
            #         return 0,isNearWall
                
        else:
            #print("trace_road default")
            kernel = np.ones((3,3),np.uint8)
            r_image_dilate = cv2.dilate(r_image, kernel, iterations =1)
            g_image_dilate = cv2.dilate(g_image, kernel, iterations =1)
            line_image = cv2.bitwise_and(r_image_dilate,g_image_dilate)
            row_line = line_image[10]


        
        line_pos = np.where(row_line==255)


        if len(line_pos[0])>0:   
            #cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+str(int(line_pos[0].mean()))+".bmp", blackwallimg)          
            return line_pos[0].mean(),isNearWall
        # if len(line_pos[0])>=60:
        #     if line_pos[0].mean() > 165:
        #         cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+str(319)+".bmp", blackwallimg)          
            
        #         return 319,isNearWall
        #     elif line_pos[0].mean() < 155: 
        #         cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+str(1)+".bmp", blackwallimg) 
        #         return 1,isNearWall
        # for middle_line in xrange(90):
        #     if middle_trace_number+middle_line+1 > 119:
        #         break
        #     lower_line = line_image[middle_trace_number+middle_line+1]
        #     lower_line_pos = np.where(lower_line==255)
        #     if len(lower_line_pos[0])>60:
        #         break
        #     if len(lower_line_pos[0])>1:
        #         below_lower_line = line_image[middle_trace_number+middle_line+10]
        #         below_lower_line_pos = np.where(below_lower_line==255)
        #         if len(below_lower_line_pos[0]) >1:
        #             high_line_pos = lower_line_pos[0].mean()
        #             low_line_pos = below_lower_line_pos[0].mean()

        #             if abs(high_line_pos - low_line_pos)<10:
        #                 break
        #             if high_line_pos > low_line_pos:
        #                 #Turn right
        #                 cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+str(320)+".bmp", blackwallimg)          
            
        #                 return 320,isNearWall
        #             else:
        #                 cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+str(0)+".bmp", blackwallimg)          
            
        #                 return 0, isNearWall
        #         break
        # r_image_dilate_half =  r_image_dilate[0:30] 
        # g_image_dilate_half =  g_image_dilate[0:30] 

        # r_pos_half = np.where(r_image_dilate_half==255)
        # g_pos_half = np.where(g_image_dilate_half==255)

        # if len(r_pos_half[0]) > 10 and len(g_pos_half[0]) > 10:
        #     r_row_mean = r_pos_half[0].mean()
        #     g_row_mean = g_pos_half[0].mean()
        #     if r_row_mean > g_row_mean:
        #         cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+str(1)+".bmp", blackwallimg) 
        #         return 1,isNearWall
        #     else:
        #         cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+str(319)+".bmp", blackwallimg) 
        #         return 319,isNearWall
            
            
        # for middle_line in xrange(90):
        #     if middle_trace_number+middle_line+1 > 119:
        #         break
        #     lower_line = line_image[middle_trace_number+middle_line+1]
        #     lower_line_pos = np.where(lower_line==255)
        #     if len(lower_line_pos[0])>60:
        #         break
        #     if len(lower_line_pos[0])>1:
        #         below_lower_line = line_image[middle_trace_number+middle_line+10]
        #         below_lower_line_pos = np.where(below_lower_line==255)
        #         if len(below_lower_line_pos[0]) >1:
        #             high_line_pos = lower_line_pos[0].mean()
        #             low_line_pos = below_lower_line_pos[0].mean()

        #             if abs(high_line_pos - low_line_pos)<10:
        #                 break
        #             if high_line_pos > low_line_pos:
        #                 #Turn right
        #                 cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+str(320)+".bmp", blackwallimg)          
            
        #                 return 320,isNearWall
        #             else:
        #                 cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+str(0)+".bmp", blackwallimg)          
            
        #                 return 0, isNearWall
        #         break
        # print "Not found any line"
        # r_image_dilate = cv2.dilate(r_image, kernel, iterations =1)
        # g_image_dilate = cv2.dilate(g_image, kernel, iterations =1)
        # b_image_dilate = cv2.dilate(b_image, kernel, iterations =1)
        # gb_line_image = cv2.bitwise_and(b_image_dilate,g_image_dilate)
        # rb_line_image = cv2.bitwise_and(b_image_dilate,r_image_dilate)

        # temp_line_image = None
        # gb_line_image_pos = np.where(gb_line_image==255)
        # rb_line_image_pos = np.where(rb_line_image==255)
        # gr_line_image_pos = np.where(line_image==255)

        # if len(gb_line_image_pos[0]) > 0 or len(rb_line_image_pos[0]) > 0 or len(gr_line_image_pos[0]) > 0:
        #     gb_mean_pos = 0
        #     gr_mean_pos = 0
        #     rb_mean_pos = 0

        #     if len(gb_line_image_pos[0]) > 0:
        #         gb_mean_pos = gb_line_image_pos[0].mean()
        #     else:
        #         gb_mean_pos = 0
            
        #     if len(gr_line_image_pos[0]) > 0:
        #         gr_mean_pos = gr_line_image_pos[0].mean()
        #     else:
        #         gr_mean_pos = 0

        #     if len(rb_line_image_pos[0]) > 0:
        #         rb_mean_pos = rb_line_image_pos[0].mean()
        #     else:
        #         rb_mean_pos = 0

        #     line_mean_pos_list = [gb_mean_pos,gr_mean_pos, rb_mean_pos]

        #     lower_line_idx = np.argmax(line_mean_pos_list)
        #     if lower_line_idx == 0:
            #     temp_line_image = gb_line_image
            #     print "Use gb line",time.time()
            # elif lower_line_idx == 1:
            #     temp_line_image = line_image
            #     print "Use gr line",time.time()
            # else:
            #     temp_line_image = rb_line_image
            #     print "Use rb line",time.time()

            # cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+str("qqq")+".bmp", temp_line_image)    
            # for middle_line in xrange(80):
            #     lower_line = temp_line_image[middle_line+1]
            #     lower_line_pos = np.where(lower_line==255)
            #     if len(lower_line_pos[0])>1:
            #         below_lower_line = line_image[middle_line+20]
            #         below_lower_line_pos = np.where(below_lower_line==255)
            #         if len(below_lower_line_pos[0]) >1:
            #             high_line_pos = lower_line_pos[0].mean()
            #             low_line_pos = below_lower_line_pos[0].mean()

            #             if abs(high_line_pos - low_line_pos)<10:
            #                 break
            #             if high_line_pos > low_line_pos:
            #                 #Turn right
                                  
            #                 cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+str(-1)+"_"+str(320)+".bmp", temp_line_image)          
  
            #                 return 320,isNearWall
            #             else:
            #                 cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+str(-1)+"_"+str(0)+".bmp", temp_line_image)  
            #                 return 0, isNearWall
            #         break


        if Settings.DEBUG:
            logit("last_pos_list"+str(last_pos_list))
        # print "Not find any line"
        # print 
        input_image = cv2.resize(blackwallimg,(80,30),interpolation=cv2.INTER_CUBIC)
        input_data = input_image[np.newaxis,:,:,:]

        direction_prob = lostLine_model.predict(input_data)
        direction = np.argmax(direction_prob)
        logit("direction_prob"+str(direction_prob))
        if direction == 0:
            #cv2.imwrite("lostLine/1/blackwallimg_"+str(time.time())+"_"+str(-1)+"_"+str(0)+".bmp", blackwallimg)          
            
            return 0,isNearWall
        else:
            #cv2.imwrite("lostLine/2/blackwallimg_"+str(time.time())+"_"+str(-1)+"_"+str(320)+".bmp", blackwallimg)          
            
            return 320,isNearWall

        if last_pos < middle_position:
            #cv2.imwrite("lostLine/1/blackwallimg_"+str(time.time())+"_"+str(-1)+"_"+str(0)+".bmp", blackwallimg)          
            if Settings.DEBUG:
                logit("return 0")
            return 0,isNearWall
        else:
            #cv2.imwrite("lostLine/2/blackwallimg_"+str(time.time())+"_"+str(-1)+"_"+str(320)+".bmp", blackwallimg) 
            if Settings.DEBUG:
                logit("return 320")
            return line_image.shape[1],isNearWall
            
        # WallAndLanes = GreyLines.TractGrey(blackwallimg)
        # print "WallAndLanes.shape",WallAndLanes.shape
        # for pixel in WallAndLanes[::-1]


        #cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+".bmp", blackwallimg)          
        


        # blackwallimg_middle_line = copy.deepcopy(blackwallimg)
        # blackwallimg_middle_line = cv2.resize(blackwallimg_middle_line, (blackwallimg_middle_line.shape[1]/4,blackwallimg_middle_line.shape[0]/5))
        

        # search_pos = last_pos
        # dist_color_thredshold = 20

        # print blackwallimg[-1][search_pos]
        # if np.linalg.norm(blackwallimg[-1][search_pos]-[0,0,255])>dist_color_thredshold and\
        #    np.linalg.norm(blackwallimg[-1][search_pos]-[0,255,0])>dist_color_thredshold and\
        #    np.linalg.norm(blackwallimg[-1][search_pos]-[255,0,0])>dist_color_thredshold:
        #     search_pos = search_pos+3
        
        # return GreyLines.GetColorMidLine(blackwallimg_middle_line, blackwallimg_middle_line[-1][last_pos], 35, last_pos)
        
        # ImageProcessor.show_image(blackwallimg, "blackwallimg")

        # whilelanes = GreyLines.ChangeLaneToWhite(blackwallimg) # change all lanes to white

        # whilelanes_gray = cv2.cvtColor(whilelanes, cv2.COLOR_BGR2GRAY)

        # whilelanes_gray = cv2.threshold(whilelanes_gray, 127, 255, cv2.THRESH_BINARY)
        

        #return blackwallimg

    @staticmethod
    def GetColorMidLine(blackwallimg_middle_line, bgr_thredshold, row_number, last_pos):
        
        # print blackwallimg_middle_line.shape
        # r_thredshold = bgr_thredshold[2]
        # g_thredshold = bgr_thredshold[1]
        # b_thredshold = bgr_thredshold[0]

        # if color == "red":
        #     bgr_thredshold = [0, 0, 255]
        # elif color == "green":
        #     bgr_thredshold = [0, 255, 0]
        # elif color == "blue":
        #     bgr_thredshold = [255, 0, 0]
        print("last_pos",last_pos)
        start_time = time.time()
        search_pos = last_pos#int(blackwallimg_middle_line.shape[1]/2)
        left_edge =0
        mide_line = blackwallimg_middle_line.shape[1]/2
        right_edge = blackwallimg_middle_line.shape[1]

        last_left_edge = left_edge
        last_right_edge = right_edge
        last_middle = mide_line

        print("blackwallimg_middle_line.shape[1]",blackwallimg_middle_line.shape[1])

        dist_color_thredshold = 20

        pos_list = []


        
        for row_n, row_image in enumerate(blackwallimg_middle_line[::-1]):
            if row_n>row_number:
                break
            dist_color_middle = np.linalg.norm(row_image[search_pos] - bgr_thredshold)

            if dist_color_middle > dist_color_thredshold:
                break

            left_line = row_image[0:search_pos]
            right_line = row_image[search_pos:]

            for idx,pixel in enumerate(left_line[::-1]):
                dist_color_left = np.linalg.norm(pixel - bgr_thredshold)
                if dist_color_left > dist_color_thredshold:
                    left_edge = search_pos - idx
                    break
            for idx,pixel in enumerate(right_line):
                dist_color_right = np.linalg.norm(pixel - bgr_thredshold)
                if dist_color_right > dist_color_thredshold:
                    right_edge = search_pos + idx
                    break
            # if left_edge !=0 and right_edge != blackwallimg_middle_line.shape[1]:
            #     mide_line = int((right_edge+left_edge) /2)
            # elif left_edge ==0:
            #     mide_line = (right_edge - last_right_edge) + last_middle
            # elif right_edge == blackwallimg_middle_line.shape[1]:
            #     mide_line = (left_edge - last_right_edge) + last_middle
            # else:
            #     mide_line = int((right_edge+left_edge) /2)

            mide_line = int((right_edge+left_edge) /2)
            last_left_edge = left_edge
            last_right_edge = right_edge
            last_middle = mide_line 
                
            search_pos = mide_line    

            pos_list.append(search_pos)       

            row_image[mide_line][0] = 255
            row_image[mide_line][1] = 255
            row_image[mide_line][2] = 255

            if mide_line == 0 or mide_line == blackwallimg_middle_line.shape[1]:
                break
        end_time = time.time()
        print("Process time", end_time-start_time)
        angle = 0
        if len(pos_list)<15:
            angle = math.degrees(math.atan2(pos_list[-1]-pos_list[0], len(pos_list)))
            print("y", pos_list[-1]-pos_list[0])
            print("x", len(pos_list))
        else:
            angle = math.degrees(math.atan2(pos_list[-1]-pos_list[-15], 15))
            print("y", pos_list[-1]-pos_list[-15])
            print("x", 15)
        print("Angle", angle)
        pos = np.array(pos_list[-15:]).mean()
        #cv2.imwrite("TestImage/blackwallimg_"+str(time.time())+"_"+str(int(angle))+"_"+str(int(pos))+".jpg", blackwallimg_middle_line)          
        return pos,angle

    @staticmethod
    def ChangeLaneToWhite(srcimg):
        #print "change lan to white"
        # Convert all non-black to white
        r, g, b = cv2.split(srcimg)
        # r_filter = (r == 255) & (g == 0) & (b == 0)
        # g_filter = (r == 0) & (g == 255) & (b == 0)
        # b_filter = (r == 0) & (g == 0) & (b == 255)

        r_filter = (r != 0) | (g != 0) | (b != 0)
        g_filter = (r != 0) | (g != 0) | (b != 0)
        b_filter = (r != 0) | (g != 0) | (b != 0)

        #filter = (r>20 or g>20 or b>20)
        #r[filter], g[filter], b[filter] = 255, 255, 255
        r[b_filter], g[b_filter] = 255, 255
        b[r_filter], g[r_filter] = 255, 255
        r[g_filter], b[g_filter] = 255, 255
        
        redlane = cv2.merge((r, g, b))

        redlane = cv2.medianBlur(redlane, 5)
        return redlane
        

    @staticmethod
    def SourceGrey(src_image):
        #print 'PrinGrey'
        r, g, b = cv2.split(src_image)
        r_filter = r == np.maximum(np.maximum(r, g), b)
        g_filter = g == np.maximum(np.maximum(r, g), b)
        b_filter = b == np.maximum(np.maximum(r, g), b)
        r[r_filter],  r[np.invert(r_filter)] = 255, 0
        g[g_filter],  g[np.invert(g_filter)] = 255, 0
        b[b_filter],  b[np.invert(b_filter)] = 255, 0
        mono_tone = cv2.merge([r, g, b])
        gray_scale_img = cv2.cvtColor(mono_tone, cv2.COLOR_RGB2GRAY)
        edge_img = cv2.Canny(gray_scale_img, 50, 150)
        #ImageProcessor.show_image(gray_scale_img, "gray")
        #ImageProcessor.drawline(edge_img);
        #lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 10, 5, 5)
        lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180 , 1, 1, 1)
        #ImageProcessor.show_image(edge_img, "Orginal")
        return lines

    @staticmethod
    def TractGrey(tract_image):
        gray_scale_img = cv2.cvtColor(tract_image, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(gray_scale_img, 50, 150)
        #ImageProcessor.show_image(gray_scale_img, "gray")
        #GreyLines.drawline(edge_img);
        #ImageProcessor.show_image(edge_img, "Tract")
        #return cv2.HoughLinesP(edge_img, 1, np.pi / 180, 10, 5, 5) 
      
    @staticmethod
    def auto_canny(image, sigma=0.33):
    	# compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

    @staticmethod
    def pic_canny(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
        # apply Canny edge detection using a wide threshold, tight
        # threshold, and automatically determined threshold
        wide = cv2.Canny(blurred, 10, 200)
        tight = cv2.Canny(blurred, 225, 250)
        auto = GreyLines.auto_canny(blurred)
    
        # show the images
        ImageProcessor.show_image(image, "original")
        ImageProcessor.show_image(np.hstack([wide, tight, auto]), "Edges")
        #cv2.imshow("Original", image)
        #cv2.imshow("Edges", np.hstack([wide, tight, auto]))
        #cv2.waitKey(0)
