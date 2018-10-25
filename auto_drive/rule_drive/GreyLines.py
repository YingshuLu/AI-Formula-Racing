import cv2 
import math
import numpy as np
from ImageProcessor import ImageProcessor
import Settings
import logger
from enum import Enum
from ImageProcessor import ImageLines

from keras.models import load_model
lostLine_model = load_model('lostLine_v5.h5')

logger = logger.get_logger(__name__)
def logit(msg):
    pass
    #if Settings.DEBUG:
    #   logger.info("%s" % msg)
class CarAction(Enum):
    forward = 0
    toleft = 1
    toright = 2
    backward = 3
    turnahead = 4

class PureWall(Enum):
    nowall = 0 
    onleft = 1
    onRight = 2
    onbothside = 3
  

class TheQuaters:
    def __init__(self):
        self._leftpoint = [0, 0]
        self._leftangle = 0
        self._leftmidyorglen = 0
        self._leftmidyselen = 0

        self._leftmidpoint = [0, 0]
        self._leftmidangle = 0
        self._leftmidyorglen = 0
        self._leftmidyselen = 0

        self._rightmidpoint = [0, 0]
        self._rightmidangle = 0
        self._rightmidyorglen = 0
        self._rightmidyselen = 0

        self._rightpoint = [0, 0]
        self._rightangle = 0
        self._rightyorglen = 0
        self._rightyselen = 0

        self._lefthalfwallangle = 0
        self._lefthalfwallpoint = [0, 0]

        self._righthalfwallangle = 0
        self._righthalfwallpoint = [0, 0]

        self._leftmidtobottom = 0
        self._rightmidtobottom = 0
        self._lefttobottom = 0
        self._righttobottom = 0
        self._lefthalftobottom = 0
        self._righthalftobottom = 0

        self._turntopointbottom = 0

        self._imagesize = [0, 0]

        self._turnaction = CarAction.forward #0: donothing, forward, 1: to left, 2: toright, 3:backward 4:wall ahead,
                                             #turnahead
        self._purewall = PureWall.nowall
        self._angletoturn = 0
        self._pointtoturn = [0,0]
        self._singlelane = False

    def AnalyzeSingleLane(self, lanepoint):
        if self._leftangle != 0 and self._rightangle != 0:
            if self._leftmidangle == 0 or self._rightmidangle == 0:
                self._singlelane = True
        
        # if self._singlelane == True:
        #     if self._leftmidangle !=0 and abs(self._lefttobottom -
        #     self._leftmidtobottom) < Settings.eagle_wall_vertical_linex:
        #         self._singlelane = False

        #     if self._rightmidangle != 0 and abs(self._righttobottom -
        #     self._rightmidtobottom) < Settings.eagle_wall_vertical_linex:
        #         self._singlelane = False
     
        if self._singlelane == True:
            pass
            #print "single lane"
            #logit("single lane, interval (%s, %s)" % (abs(self._lefttobottom -
            #self._leftmidtobottom), abs(self._righttobottom -
            #self._rightmidtobottom) ) )
        return self._singlelane
    
    def RunSummary(self, corppedimage):
        self._imagesize = corppedimage
        corppedimageY = corppedimage[1]

        self._lefttobottom = corppedimageY - self._leftpoint[1] if self._leftangle != 0 else 0
        self._leftmidtobottom = corppedimageY - self._leftmidpoint[1] if self._leftmidangle != 0 else 0
        self._rightmidtobottom = corppedimageY - self._rightmidpoint[1] if self._rightmidangle != 0 else 0
        self._righttobottom = corppedimageY - self._rightpoint[1] if self._rightangle != 0 else 0
        
        if self._leftangle == 0 and self._leftmidangle == 0 and self._rightmidangle == 0 and self._rightangle == 0:
            return False

        # Remove false positive middle wall
        if self._leftmidangle != 0 and self._leftmidtobottom < Settings.eagle_far_wall_distanceY:
            # 1.  no wall on left, 2.  wall on left and density > 1.2, 3.  not
            # a vertical line
            #logit("remove all left: (%s, %s), (%s, %s)" % (self._leftpoint,
            #self._leftangle, self._leftyorglen, self._leftyselen) )
            if (self._leftangle == 0 and self._leftyorglen != 0) or (self._leftangle != 0 and self._leftyorglen / float(self._leftyselen) > 1.2) or (self._leftangle != 0 and (abs(self._leftpoint[1] - self._leftmidpoint[1]) < Settings.eagle_wall_horizon_line_interval)): #besides wall or
                #logit('Remove leftmid positive wall (%s, %s, %s, %s)'%
                self._leftmidpoint, self._leftmidangle = [0, 0], 0
                
        if self._rightmidangle != 0 and self._rightmidtobottom < Settings.eagle_far_wall_distanceY:
            if (self._rightangle == 0 and self._rightyorglen != 0) or (self._rightangle != 0 and self._rightyorglen / float(self._rightyselen) > 1.2) or (self._rightangle != 0 and (abs(self._rightpoint[1] - self._rightmidpoint[1]) < Settings.eagle_wall_horizon_line_interval)): #besides wall or
                #logit('Remove right mid positive wall (%s, %s, %s, %s)'%
                self._rightmidpoint, self._rightmidangle = [0,0], 0
                
        self.AnalyzeQuarters()

        self._lefthalftobottom = corppedimageY - self._lefthalfwallpoint[1] if self._lefthalfwallangle != 0 else 0
        self._righthalftobottom = corppedimageY - self._righthalfwallpoint[1] if self._righthalfwallangle != 0 else 0
        self._turntopointbottom = corppedimageY - self._pointtoturn[1] if self._angletoturn != 0 else 0
        
    def AnaylyzeLeft(self):
        if self._leftmidangle != 0 and self._leftangle != 0:
            self._lefthalfwallpoint = [(self._leftmidpoint[0] + self._leftpoint[0]) / 2, (self._leftmidpoint[1] + self._leftpoint[1]) / 2]
            self._lefthalfwallangle = GreyLines.GetAngle(self._imagesize[0] / 2, self._imagesize[1], self._lefthalfwallpoint[0], self._lefthalfwallpoint[1])
            
            #logit("AnalyzeRight before: (%s, %s), imagesize (%s, %s)" %
            #(self._lefthalfwallpoint, self._lefthalfwallangle,
            #self._imagesize[0], self._imagesize[1]) )

            if self._leftmidtobottom > self._lefttobottom: #to right
                self._lefthalfwallangle += 90
            elif self._leftmidtobottom < self._lefttobottom: #to left
                self._lefthalfwallangle = -45
            else:
                logit("exception: leftmidtobottom == lefttobottom and angle is not null")
            
            print ("left wall should go to: %s" % self._righthalfwallangle )

        elif self._leftangle != 0:
            self._lefthalfwallangle = self._leftangle + 90
            self._lefthalfwallpoint = self._leftpoint
        elif self._leftmidangle != 0:
            self._lefthalfwallangle = self._leftmidangle + 90
            self._lefthalfwallpoint = self._leftmidpoint
        else:
            self._lefthalfwallangle = 0
            self._lefthalfwallpoint = [0, 0]

        if (self._leftangle == 0 or self._leftmidangle == 0) and self._leftyorglen < Settings.eagel_pure_wall_points:
            self._purewall = PureWall.onleft
            #print "pure wall on left"
            logit("pure wall on left")
        
        #logit("AnaylyzeLeft: (%s, %s)" % (self._lefthalfwallpoint,
        #self._lefthalfwallangle) )
    
    def AnalyzeRight(self):
        if self._rightmidangle != 0 and self._rightangle != 0:
            self._righthalfwallpoint = [(self._rightmidpoint[0] + self._rightpoint[0]) / 2, (self._rightmidpoint[1] + self._rightpoint[1]) / 2]
            self._righthalfwallangle = GreyLines.GetAngle(self._imagesize[0] / 2, self._imagesize[1], self._righthalfwallpoint[0], self._righthalfwallpoint[1])
            #logit("AnalyzeRight before: (%s, %s), imagesize (%s, %s)" %
            #(self._righthalfwallpoint, self._righthalfwallangle,
            #self._imagesize[0], self._imagesize[1]) )
            
            if self._rightmidtobottom > self._righttobottom: #to left
                self._righthalfwallangle = self._righthalfwallangle - 90
            elif self._rightmidtobottom < self._righttobottom: #to right
                self._righthalfwallangle = 45
            else:
                logit("exception: rightmidtobottom == righttobottom")
            print ("right wall should go to: %s" % self._righthalfwallangle )
        elif self._rightangle != 0:
            self._righthalfwallangle = self._rightangle - 90
            self._righthalfwallpoint = self._rightpoint
        elif self._rightmidangle != 0:
            self._righthalfwallangle = self._rightmidangle - 90
            self._righthalfwallpoint = self._rightmidpoint
        else:
            self._righthalfwallangle = 0
            self._righthalfwallpoint = [0, 0]

        if (self._rightangle == 0 or self._rightmidangle == 0) and self._rightyorglen < Settings.eagel_pure_wall_points:
            self._purewall = PureWall.onRight if self._purewall != None else PureWall.onbothside
            #print "pure wall on right"
            logit("pure wall on right")
        
        #logit("AnalyzeRight: (%s, %s)" % (self._righthalfwallpoint,
        #self._righthalfwallangle) )
        
    def AnalyzeSharpTurn(self):
        pass
        
    def AnalyzeQuarters(self):
        # Adjust far away angle
        self.AnaylyzeLeft()
        self.AnalyzeRight()

        if self._lefthalfwallangle != 0 and self._righthalfwallangle != 0: #wall on both sides
            leftside = min(self._leftangle, self._leftmidangle)
            rightside = min(self._rightangle, self._rightangle)

            if leftside != 0 and rightside != 0 and abs(self._leftpoint[1] - self._rightpoint[1]) < Settings.eagle_wall_horizon_line_interval:
                #logit("debuginfo: Forward to wall, might be backward: left
                #half: %s, righthalf: %s" % (self._leftpoint, self._rightpoint)
                #)
                self._turnaction = CarAction.backward                    
            elif self._lefthalfwallpoint[1] > self._righthalfwallpoint[1]: #choose the wall of nearby
                self._turnaction = CarAction.toright if self._lefthalfwallangle > 0 else CarAction.toleft
                self._angletoturn = self._lefthalfwallangle
                self._pointtoturn = self._lefthalfwallpoint
            else:
                self._turnaction = CarAction.toright if self._righthalfwallangle > 0 else CarAction.toleft
                self._angletoturn = self._righthalfwallangle
                self._pointtoturn = self._righthalfwallpoint
            
            if self._leftmidangle != 0 and self._rightmidangle != 0: #and self._turnaction == CarAction.forward: #car is forwarding the wall,
            #logit("debuginfo: wall in the middle, turn sharp ahead")
                self._turnaction = CarAction.turnahead
        elif self._lefthalfwallangle != 0:
            self._turnaction = CarAction.toright if self._lefthalfwallangle > 0 else CarAction.toleft
            self._angletoturn = self._lefthalfwallangle
            self._pointtoturn = self._lefthalfwallpoint
        elif self._righthalfwallangle != 0:
            self._turnaction = CarAction.toright if self._righthalfwallangle > 0 else CarAction.toleft
            self._angletoturn = self._righthalfwallangle
            self._pointtoturn = self._righthalfwallpoint
        else:
            logit("debuginfo:: Both right and left angle are 0")

        #logit("turnaction %s %s, lefthalf wallinfo %s %s , righthalf wallinfo
        #%s %s, self._purewall %s"%
        #    (self._turnaction, self._angletoturn, self._lefthalfwallpoint,
        #    self._lefthalfwallangle, self._righthalfwallpoint,
        #    self._righthalfwallangle, self._purewall))
class GreyLines(object):
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
        #arrsum = np.sum(array, 0)
        #print "DstArray size: %d, arrsum: %s" % (array.size, arrsum)

        if array.size > 0 and array.shape[0] != 0:
            lanepoint = np.mean(array, axis = 0) #arrsum/array.shape[0]
        else:
            lanepoint = [0, 0]
        return [int(lanepoint[1]), int(lanepoint[0])]

                
    @staticmethod
    def GetCoordinate(edge):
        #np.set_printoptions(threshold='nan')
        ans = np.zeros((edge.shape[1], edge.shape[0]), dtype=int)
        for y in range(0, edge.shape[0]):
            for x in range(0, edge.shape[1]):
                if edge[y, x] != 0:
                    ans[x, y] = 1
                    #print "(%d, %d)" % (x, y)

        print("numpy shape: %d %d" % (ans.shape[0], ans.shape[1]))
        #print "ans: %s" % ans
        return ans

    @staticmethod
    def GetZone(edgecoor, part=[0, 0, 0, 0]):
        #np.set_printoptions(threshold='nan')
        #print "part: %s, %s, %s, %s" % (part[0], part[1], part[2], part[3])
        ybeg = edgecoor.shape[0] / part[0] if part[0] > 0 else 0
        yend = edgecoor.shape[0] / part[1] if part[1] > 0 else edgecoor.shape[0]

        xbeg = edgecoor.shape[1] / part[2] if part[2] > 0 else 0
        xend = edgecoor.shape[1] / part[3] if part[3] > 0 else edgecoor.shape[1]
        
        #print "ybeg: %d, yend: %d, xbeg: %d, xend: %d"% (ybeg, yend, xbeg,
        #xend)
        targetzone = edgecoor[int(ybeg):int(yend),int(xbeg):int(xend)]
        #print "targetzone: %s" % targetzone
        return targetzone
        

    @staticmethod
    def EdgeMergeByZone(srcedge, objedge, zone=[0, 0, 0, 0]):
        srcedgeslice = GreyLines.GetZone(srcedge, zone)
        objectedgeslice = GreyLines.GetZone(objedge, zone)
        rstedge = np.logical_xor(srcedgeslice, objectedgeslice)
        #print "rstedge: %s " % rstedge
        indexs = np.transpose(rstedge.nonzero())
        #print "count of nonzero: %s" % indexs.size
        return indexs
    
    @staticmethod
    def GetSharpLinesUturnOnly(WallAndLanes, Walls, blackwallimg):
        leftbase = [120, 20]
        rightbase = [120, 20]
        targetzone = GreyLines.EdgeMergeByZone(WallAndLanes, Walls, [6, 3, 3, 1.6])
        #targetrightzone = GreyLines.EdgeMergeByZone(WallAndLanes, Walls, [4,
        #2, 2, 1.4])
        
        if targetzone.size > 10:
            targetxy = GreyLines.ArraySum(targetzone)
            print("targetxy %s" % targetxy)
            
            rows = np.where(targetzone[:,1] < targetxy[0])
            leftxy = GreyLines.ArraySum(targetzone[rows])
            print("leftxy %s" % leftxy)

            rows = np.where(targetzone[:,1] > targetxy[0])
            rightxy = GreyLines.ArraySum(targetzone[rows])
            print("rightxy %s" % rightxy)

            leftajxy = [leftxy[0] + leftbase[0], leftxy[1] + leftbase[1]]
            rightajxy = [rightxy[0] + rightbase[0], rightxy[1] + rightbase[1]]
            yinterval = rightajxy[1] - leftajxy[1]
            lineAngle = GreyLines.GetAngle(rightajxy[0], rightajxy[1], leftajxy[0], leftajxy[1]) if yinterval > 0 else GreyLines.GetAngle(leftajxy[0], leftajxy[1], rightajxy[0], rightajxy[1])
            print("lineAngle: %s, leftadxy: %s, rightadxy: %s, " % (lineAngle, leftajxy, rightajxy))
            if Settings.DEBUG_IMG:
                GreyLines.Drawline(blackwallimg, leftajxy, (255,0,255))
                GreyLines.Drawline(blackwallimg, rightajxy, (0,255,255))
        
        else:
            return 0, None, None
        return lineAngle, leftajxy, rightajxy    



    @staticmethod
    def GetSharpLines(WallAndLanes, Walls, blackwallimg):
        leftbase = [80, 60]
        rightbase = [160, 60]
        targetleftzone = GreyLines.EdgeMergeByZone(WallAndLanes, Walls, [2, 0, 4, 2])
        targetrightzone = GreyLines.EdgeMergeByZone(WallAndLanes, Walls, [2, 0, 2, 1.4])
        
        if targetleftzone.size > 5 and targetrightzone.size > 5:
            leftxy = GreyLines.ArraySum(targetleftzone)
            rightxy = GreyLines.ArraySum(targetrightzone)
            
            leftajxy = [leftxy[0] + leftbase[0], leftxy[1] + leftbase[1]]
            rightajxy = [rightxy[0] + rightbase[0], rightxy[1] + rightbase[1]]
            yinterval = rightajxy[1] - leftajxy[1]
            lineAngle = GreyLines.GetAngle(rightajxy[0], rightajxy[1], leftajxy[0], leftajxy[1]) if yinterval > 0 else GreyLines.GetAngle(leftajxy[0], leftajxy[1], rightajxy[0], rightajxy[1])
            #print "lineAngle: %s, leftadxy: %s, rightadxy: %s, " % (lineAngle,
            #leftajxy, rightajxy, )
            if Settings.DEBUG_IMG:
                GreyLines.Drawline(blackwallimg, leftajxy, (255,0,255))
                GreyLines.Drawline(blackwallimg, rightajxy, (0,255,255))
        
        else:
            return 0, None, None
        return lineAngle, leftajxy, rightajxy    


    @staticmethod
    def EdgeMergeByValidPixel(srcedge, objedge):
        #np.set_printoptions(threshold='nan')
        rstedge = np.logical_xor(srcedge, objedge)
        #print "rstedge: %s " % rstedge
        indexs = np.transpose(rstedge.nonzero())
        #print "index of nonzero: %s" % indexs
        return indexs[:indexs.shape[0], :]

    @staticmethod
    def Drawline(img, point, color=(255,255,0)):
        image_height = img.shape[0]
        image_width = img.shape[1]
        #print "image: heigh: %d, width: %d" % (image_height, image_width)
        cv2.line(img,(int(image_width / 2), int(image_height)),(point[0], point[1]), color, 2)

    @staticmethod
    def GetAngle(carx, cary, expectx, expecty):
        #print carx, cary, expectx, expecty
        myradians = math.atan2(expectx - carx, cary - expecty)
        return math.degrees(myradians)

    @staticmethod
    def ExtractValidZone(indexs, blackwallimg, zone): #
        firstclomn = indexs[:,1]
        validcolumn = list(set(firstclomn))
        yorglen, yselen = len(firstclomn), len(validcolumn)
        
        density = yorglen / float(yselen) if yselen != 0 else 0.0
        
        #logit('%s wallpoint, wallinfo (%s, %s, %s, %s)' % (zone, yorglen,
        #yselen, density, Settings.eagle_wall_density))
        
        wallpoint = [0, 0]
        rstangle = 0
        if yorglen < Settings.eagle_wall_throttle:
            return wallpoint, rstangle, yorglen, yselen
        else:
            if yselen > 0 and density > Settings.eagle_wall_density:
                return wallpoint, rstangle, yorglen, yselen
       
        wallpoint = GreyLines.ArraySum(indexs) 
        rstangle = GreyLines.GetAngle(blackwallimg.shape[1] / 2, blackwallimg.shape[0], wallpoint[0], wallpoint[1])
        
        #if Settings.DEBUG:
        #    logit('%s wallpoint, rstangle(%s %s), wallvalidinfo (%s, %s, %s)'
        #    % (zone, wallpoint, rstangle, yorglen, yselen, density))

        return wallpoint, rstangle, yorglen, yselen

  
    @staticmethod
    def GetleftWallZoneCoor(rawarray, blackwallimg, ylevel=Settings.eagle_wall_ylevel):
        #print (rawarray)
        rows = np.where(rawarray[:, 0] > blackwallimg.shape[0] * ylevel)
        indexs1 = rawarray[rows]

        rows = np.where(indexs1[:,1] < blackwallimg.shape[1] * 0.25)
        indexs = indexs1[rows]
        
        return GreyLines.ExtractValidZone(indexs, blackwallimg, "left")
    
    @staticmethod
    def GetleftMidlleWallZoneCoor(rawarray, blackwallimg, ylevel=Settings.eagle_wall_ylevel):
        rows = np.where(rawarray[:, 0] > blackwallimg.shape[0] * ylevel)
        indexs1 = rawarray[rows]

        rows = np.where(indexs1[:,1] > blackwallimg.shape[1] * 0.25)
        indexs2 = indexs1[rows]

        rows = np.where(indexs2[:,1] < blackwallimg.shape[1] * 0.5)
        indexs = indexs2[rows]
        
        return GreyLines.ExtractValidZone(indexs, blackwallimg, "leftmiddle")

    @staticmethod
    def GetrightMidlleWallZoneCoor(rawarray, blackwallimg, ylevel=Settings.eagle_wall_ylevel):
        rows = np.where(rawarray[:, 0] > blackwallimg.shape[0] * ylevel)
        indexs1 = rawarray[rows]

        rows = np.where(indexs1[:,1] > blackwallimg.shape[1] * 0.5)
        indexs2 = indexs1[rows]

        rows = np.where(indexs2[:,1] < blackwallimg.shape[1] * 0.75)
        indexs = indexs2[rows]
        
        return GreyLines.ExtractValidZone(indexs, blackwallimg, "righmiddle")

    @staticmethod
    def GetRightWallZoneCoor(rawarray, blackwallimg, ylevel=Settings.eagle_wall_ylevel):      
        rows = np.where(rawarray[:, 0] > blackwallimg.shape[0] * ylevel)
        indexs1 = rawarray[rows]

        rows = np.where(indexs1[:,1] > blackwallimg.shape[1] * 0.75)
        indexs = indexs1[rows]

        return GreyLines.ExtractValidZone(indexs, blackwallimg, "right")

    @staticmethod
    def GetWallAngle(Walls, blackwallimg):
        #np.set_printoptions(threshold='nan')

        #indices = np.where( Walls != [0])
        #indexs = np.array (zip(indices[0], indices[1]))
        indexs = np.transpose(Walls.nonzero())
        #print (indexs.size)

        rq = TheQuaters()
                
        if indexs.size == 0:
            return rq

        rq._leftpoint, rq._leftangle, rq._leftyorglen, rq._leftyselen = GreyLines.GetleftWallZoneCoor(indexs, blackwallimg)
        
        rq._leftmidpoint, rq._leftmidangle, rq._leftmidyorglen, rq._leftmidyselen = GreyLines.GetleftMidlleWallZoneCoor(indexs, blackwallimg)
        
        rq._rightmidpoint, rq._rightmidangle, rq._rightmidyorglen, rq._rightmidyselen = GreyLines.GetrightMidlleWallZoneCoor(indexs, blackwallimg)

        rq._rightpoint, rq._rightangle, rq._rightyorglen, rq._rightyselen = GreyLines.GetRightWallZoneCoor(indexs, blackwallimg)

        rst = rq.RunSummary([blackwallimg.shape[1], blackwallimg.shape[0]])
       
        if Settings.DEBUG_IMG and Settings.UIMG == 1: 
            # if rq._lefthalfwallangle != 0:
            #     print(rq._righthalfwallpoint)
            #     GreyLines.Drawline(blackwallimg, rq._lefthalfwallpoint,
            #     (255,255,255))
            
            # if rq._righthalfwallangle != 0:
            #     print(rq._righthalfwallpoint)
            #     GreyLines.Drawline(blackwallimg, rq._righthalfwallpoint,
            #     (255,255,255))
            
            if rq._leftangle != 0:
                 GreyLines.Drawline(blackwallimg, rq._leftpoint, (0,0,0))
            if rq._rightangle != 0:     
                 GreyLines.Drawline(blackwallimg, rq._rightpoint, (0,0,0))
            if rq._leftmidangle != 0:     
                 GreyLines.Drawline(blackwallimg, rq._leftmidpoint, (0,0,0)) 
            if rq._rightmidangle != 0:     
                 GreyLines.Drawline(blackwallimg, rq._rightmidpoint, (0,0,0)) 
        #     #ImageProcessor.show_image(LanesCoor, "Lanes")
            #ImageProcessor.show_image(Walls, "Walls")
        
        return rq

    @staticmethod
    def GetWallAngleForUtrun(Walls, blackwallimg):
        #np.set_printoptions(threshold='nan')

        indices = np.where(Walls != [0])
        indexs = np.array(zip(indices[0], indices[1])) 
        
        rq = TheQuaters()
                
        if indexs.size == 0:
            return rq

        rq._leftpoint, rq._leftangle, rq._leftyorglen, rq._leftyselen = GreyLines.GetleftWallZoneCoor(indexs, blackwallimg, Settings.eagle_wall_ylevel_uturn)
        
        rq._leftmidpoint, rq._leftmidangle, rq._leftmidyorglen, rq._leftmidyselen = GreyLines.GetleftMidlleWallZoneCoor(indexs, blackwallimg, Settings.eagle_wall_ylevel_uturn)
        
        rq._rightmidpoint, rq._rightmidangle, rq._rightmidyorglen, rq._rightmidyselen = GreyLines.GetrightMidlleWallZoneCoor(indexs, blackwallimg, Settings.eagle_wall_ylevel_uturn)

        rq._rightpoint, rq._rightangle, rq._rightyorglen, rq._rightyselen = GreyLines.GetRightWallZoneCoor(indexs, blackwallimg, Settings.eagle_wall_ylevel_uturn)

        rst = rq.RunSummary([blackwallimg.shape[1], blackwallimg.shape[0]])
        
        if Settings.DEBUG_IMG and Settings.UIMG == 2: 
            if rq._lefthalfwallangle != 0:
                GreyLines.Drawline(blackwallimg, rq._lefthalfwallpoint, (255,255,255))
            
            if rq._righthalfwallangle != 0:
                GreyLines.Drawline(blackwallimg, rq._righthalfwallpoint, (255,255,255))
            
            if rq._leftangle != 0:
                 GreyLines.Drawline(blackwallimg, rq._leftpoint, (0,0,0))
            if rq._rightangle != 0:     
                 GreyLines.Drawline(blackwallimg, rq._rightpoint, (0,0,0))
            if rq._leftmidangle != 0:     
                 GreyLines.Drawline(blackwallimg, rq._leftmidpoint, (0,0,0)) 
            if rq._rightmidangle != 0:     
                 GreyLines.Drawline(blackwallimg, rq._rightmidpoint, (0,0,0)) 
            #ImageProcessor.show_image(LanesCoor, "Lanes")
            #ImageProcessor.show_image(Walls, "Walls")
        
        return rq

    @staticmethod
    def GetWallAngleForBirdview(Walls, blackwallimg):
        #np.set_printoptions(threshold='nan')

        indices = np.where(Walls != [0])
        indexs = np.array(zip(indices[0], indices[1])) 
        
        rq = TheQuaters()
                
        if indexs.size == 0:
            return rq

        rq._leftpoint, rq._leftangle, rq._leftyorglen, rq._leftyselen = GreyLines.GetleftWallZoneCoor(indexs, blackwallimg, Settings.eagle_wall_ylevel_birdview)
        
        rq._leftmidpoint, rq._leftmidangle, rq._leftmidyorglen, rq._leftmidyselen = GreyLines.GetleftMidlleWallZoneCoor(indexs, blackwallimg, Settings.eagle_wall_ylevel_birdview)
        
        rq._rightmidpoint, rq._rightmidangle, rq._rightmidyorglen, rq._rightmidyselen = GreyLines.GetrightMidlleWallZoneCoor(indexs, blackwallimg, Settings.eagle_wall_ylevel_birdview)

        rq._rightpoint, rq._rightangle, rq._rightyorglen, rq._rightyselen = GreyLines.GetRightWallZoneCoor(indexs, blackwallimg, Settings.eagle_wall_ylevel_birdview)

        rst = rq.RunSummary([blackwallimg.shape[1], blackwallimg.shape[0]])
        
        if Settings.DEBUG_IMG and Settings.UIMG == 3: 
            if rq._lefthalfwallangle != 0:
                GreyLines.Drawline(blackwallimg, rq._lefthalfwallpoint, (255,255,255))
            
            if rq._righthalfwallangle != 0:
                GreyLines.Drawline(blackwallimg, rq._righthalfwallpoint, (255,255,255))
            
            if rq._leftangle != 0:
                 GreyLines.Drawline(blackwallimg, rq._leftpoint, (0,0,0))
            if rq._rightangle != 0:     
                 GreyLines.Drawline(blackwallimg, rq._rightpoint, (0,0,0))
            if rq._leftmidangle != 0:     
                 GreyLines.Drawline(blackwallimg, rq._leftmidpoint, (0,0,0)) 
            if rq._rightmidangle != 0:     
                 GreyLines.Drawline(blackwallimg, rq._rightmidpoint, (0,0,0)) 
            #ImageProcessor.show_image(LanesCoor, "Lanes")
            #ImageProcessor.show_image(Walls, "Walls")
        
        return rq

    @staticmethod
    def GetLanesAngleV1(WallAndLanes, Walls, blackwallimg, direction = 0):
        targetxy = GreyLines.EdgeMergeByValidPixel(WallAndLanes, Walls) # Remove wall edges
        #targetzone = GreyLines.EdgeMergeByZone(WallAndLanes, Walls, Settings.birdviewpart)
        rstangle, lanepoint = 0, [0, 0]
        if targetxy.size == 0:
            return rstangle, lanepoint
        
        #print (targetxy)
        rows = np.where(targetxy[:, 0] < Settings.eagle_high_alllane_yvalue)
        slicedarray = targetxy[rows]
        GreyLines.Drawline(blackwallimg,GreyLines.ArraySum(slicedarray))

        #print (targetxy)    
        if len(slicedarray) == 0 and targetxy.size > 0:
            print ("no value where y< %s "%Settings.eagle_high_alllane_yvalue)
            ratio = int(targetxy.size / 20) + 1
            slicedarray = targetxy[:ratio, :] 
            GreyLines.Drawline(blackwallimg,GreyLines.ArraySum(slicedarray))


        # firstclomn = slicedarray[:,1]
        # validcolumn = list(set(firstclomn))
        # orglen, selen = len(firstclomn), len(validcolumn)
        # if orglen < Settings.eagle_lanes_throttle:
        #     return rstangle, lanepoint
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
            #GreyLines.DrawPoints(targetxy, "targetxy")
            #ImageProcessor.show_image(Walls, "Walls")

        if lanepoint[0] == 0 and lanepoint[1] == 0:
            if Settings.DEBUG:
                logit("No lanes can be found")
                logit("slice ratio: %s, turn %s" % (sliceratio, direction))
        else:
            rstangle = GreyLines.GetAngle(blackwallimg.shape[1]/2, blackwallimg.shape[0], lanepoint[0], lanepoint[1])
        
        return rstangle, lanepoint

    #parts cooradinate ratio: [ybeg, yend, xbeg, xend]
    @staticmethod
    def GetLanesAngle(srcimg, direction=0):
        #targetxy = GreyLines.EdgeMergeByValidPixel(WallAndLanes, Walls) #
        #Remove wall edges
        #targetzone = GreyLines.EdgeMergeByZone(WallAndLanes, Walls,
        #Settings.birdviewpart)
        il = ImageLines(srcimg)
        il.ImageBasicProcess()
        blackwallimg = il._image
        targetxy = il.GetAllEdgeLine()
        #print "targetxy: %s" % targetxy
        cdpos, cdcnt, cdxy, mapcount = il.GetMidlleLine(Settings.eagle_high_middlelane_yvalue)
        rstangle, airstangle, lanepoint = 0, 0, [0, 0] 
        
        slicedarray = targetxy
        
        sliceratio = 0
        if direction == -1: #turn left
            sliceratio = blackwallimg.shape[1] * (1 - Settings.eagle_traffice_slice_ratio)
            rows = np.where(slicedarray[:, 1] < sliceratio)
            targetxy = slicedarray[rows]
        if direction == 1: #turn right
            sliceratio = blackwallimg.shape[1] * Settings.eagle_traffice_slice_ratio
            rows = np.where(slicedarray[:, 1] > sliceratio)
            targetxy = slicedarray[rows]

        rows = np.where(targetxy[:, 0] < Settings.eagle_high_alllane_yvalue)
        slicedarray = targetxy[rows]
        
        #Get middle area for single lane
        middleareaangle = 0
        midareapoint = [0, 0]
        if targetxy.size > 0:
            rows = np.where(targetxy[:, 0] > 10)
            middlearea = targetxy[rows]

            rows = np.where(middlearea[:, 0] < 30)
            middlearea = middlearea[rows]

            midareapoint = GreyLines.ArraySum(middlearea) 

            middleareaangle = GreyLines.GetAngle(blackwallimg.shape[1] / 2, blackwallimg.shape[0], midareapoint[0], midareapoint[1])

        if len(slicedarray) == 0 and targetxy.size > 0:
            ratio = int(targetxy.size / 20) + 1
            slicedarray = targetxy[:ratio, :] 

            logit("lines not found %s " % Settings.eagle_high_alllane_yvalue)
            
            
        #print "slicedarray: %s " % slicedarray

        # firstclomn = slicedarray[:,1]
        # validcolumn = list(set(firstclomn))
        # orglen, selen = len(firstclomn), len(validcolumn)
        # if orglen < Settings.eagle_lanes_throttle:
        #     return rstangle, lanepoint
        #print 'GetLanesAngle indexs size:%s,selen %s' % (orglen, selen)
        # if len(slicedarray) == 0:
        #     rows = np.where(targetxy[:, 0] <
        #     Settings.eagle_high_alllane_yvalue + 5)
        #     slicedarray = targetxy[rows]
        
        validarray = slicedarray
        
        lanepoint = GreyLines.ArraySum(validarray) # Pickup specified zone to calc the center points
        # if cdxy[0] == 0 and cdxy[1] == 0:
        #         #blackwallimg = ImageProcessor.preprocess(srcimg, 0.5) # Get
        #         #cropped image and convert image wall to black

        #         blackwallimg = cv2.medianBlur(blackwallimg, 5)
        #         input_image = cv2.resize(blackwallimg,(80,30),interpolation=cv2.INTER_CUBIC)
        #         input_data = input_image[np.newaxis,:,:,:]

        #         direction_prob = lostLine_model.predict(input_data)
        #         direction = np.argmax(direction_prob)
        #         #logit("direction_prob"+str(direction_prob))
        #         if direction == 0:
        #             #cv2.imwrite("lostLine/1/blackwallimg_"+str(time.time())+"_"+str(-1)+"_"+str(0)+".bmp",
        #             #blackwallimg)
        #             #cv2.imwrite("TestImage3/1/srcimg_"+str(time.time())+".bmp",
        #             #srcimg[120:])
                    
        #             airstangle = -1
        #             #lanepoint = [0, 0]
        #             logit("middle line miss-- turn left")
        #         else:
        #             #cv2.imwrite("lostLine/2/blackwallimg_"+str(time.time())+"_"+str(-1)+"_"+str(320)+".bmp",
        #             #blackwallimg)
        #             #cv2.imwrite("TestImage3/2/srcimg_"+str(time.time())+".bmp",
        #             #srcimg[120:])
                    
        #             airstangle = 1
        #             #lanepoint = [320, 0]
        #             logit("middle line miss-- turn right")

        if Settings.DEBUG_IMG:
            #GreyLines.Drawline(blackwallimg,lanepoint)
            cv2.circle(blackwallimg,(lanepoint[0], lanepoint[1]), 6,(255,0,0),4)
            #ImageProcessor.show_image(blackwallimg, "blackwallimg")

            cv2.circle(blackwallimg,(midareapoint[0], midareapoint[1]), 6,(0,0,255),4)
            ImageProcessor.show_image(blackwallimg, "blackwallimg")
        #     #ImageProcessor.show_image(LanesCoor, "Lanes")
            #ImageProcessor.show_image(Walls, "Walls")

        if lanepoint[0] == 0 and lanepoint[1] == 0:
            if Settings.DEBUG:
                logit("No lanes can be found")
        elif rstangle == 0:
            rstangle = GreyLines.GetAngle(blackwallimg.shape[1] / 2, blackwallimg.shape[0], lanepoint[0], lanepoint[1])
        
        logit("lanes info: rstangle, lanepoint (%s, %s, %s)" % (rstangle, lanepoint, middleareaangle))

        return rstangle, lanepoint, mapcount, airstangle, middleareaangle

    @staticmethod
    def GetEdgeImages(srcimg):
        blackwallimg = ImageProcessor.preprocess(srcimg, 0.5) # Get cropped image and convert image wall to black

        whilelanes = GreyLines.ChangeLaneToWhite(blackwallimg) # change all lanes to white
        #ImageProcessor.show_image(whilelanes, "whilelanes")
        whilelanes = cv2.medianBlur(whilelanes, 5)
        #ImageProcessor.show_image(whilelanes, "medianBlurwhilelanes")

        Walls = GreyLines.TractGrey(whilelanes) # Get wall edges
        #ImageProcessor.show_image(Walls, "Walls")
        WallAndLanes = GreyLines.TractGrey(blackwallimg) # Get lanes and wall edges
        return WallAndLanes, Walls, blackwallimg
        #GreyLines.DrawPoints(LanesCoor, "LanesCoorpoint")
        #GreyLines.DrawPoints(WallAndLanesCoor, "WallAndLanesCoorpoint")

    @staticmethod
    def ChangeLaneToWhite(srcimg):
        #print "change lan to white"
        # Convert all non-black to white
        r, g, b = cv2.split(srcimg)
        r_filter = (r == 255) & (g == 0) & (b == 0)
        g_filter = (r == 0) & (g == 255) & (b == 0)
        b_filter = (r == 0) & (g == 0) & (b == 255)

        r[b_filter], g[b_filter] = 255, 255
        b[r_filter], g[r_filter] = 255, 255
        r[g_filter], b[g_filter] = 255, 255
        
        redlane = cv2.merge((r, g, b))
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
