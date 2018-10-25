# coding=utf-8
import time
from ImageProcessor import ImageProcessor
import threading
import os
import logger
from PID import PID
import Globals
import Settings
import TrafficSignType
from Car import Car
from RoadConditionCheck import RoadConditionCheck
import json
import random 
from GreyLines import GreyLines
from GreyLines import CarAction
from GreyLines import PureWall
import math
import traceback
import numpy as np
from TrafficSignType import detect_obstacle
from JeffreyDrive import JeffreyDrive_v1
from GreyLinesEx2 import GreyLines as GreyLineEx

logger = logger.get_logger(__name__)
def logit(msg):
    pass
    #if Settings.DEBUG:
    #   logger.info("%s" % msg)
class CrazyDrive(object):
    debug = Settings.DEBUG

    def __init__(self, car, road_condition_check):
        self._car = car
        self._car.register(self)
        self._roadcondition = road_condition_check
        self._roadcondition = road_condition_check
        self._roadcondition.register(self)
        self._last_traffic_sign = None
        self._changelane_direction = 0 # -1: to left, 1:to right, 2: u-turn
        self._last_changelane_direction = 0
        self.last_see_six_road_count = 0
        self._throttle_to_return = 0
        self._steering_angle_to_return = 0
        #self._steering_action_to_return = ""
        self._last_traffic_sign_time = None
        self._last_singal_lane_time = 0
        self._being_handle_stuck = None
        self._lanesangle = 0
        self._lastanglezerotime = 0
        self._lanepoint = 0
        self._mapcount = -1
        self._airstangle = 0
        self._middleareaangle = 0
        self._quaters = None
        self._quatersutrun = None
        self._quatersbirdview = None
        self._backward = False
        self._wallfound = False
        self._wallsharpturn = False
        self._action = "follow lanes"
        self._last_lanes_angle = 0
        self._corppedimageX = 360
        self._corppedimageY = 120
        self._last_road_condition_time = None
        self._last_road_condition = None
        # 行进中墙在哪边，决定了掉头时方向转角
        self._wrongway_direction = None 
        self._being_handle_wrongway = None
        self._wrongway_handle_begin_time = None
        self._wrongwayhandle_step_1_time_cost = 0 #为了在三赛道的场景下掉头，分两步：第一步指定转弯角度倒车与墙垂直，第二步指定转弯角度前进
        self._next_to_wall_history = []

        # 撞向哪边位置的墙，决定了恢复时方向转角，比如撞到行进中左边的墙，则左转方向倒车
        self._stuck_direction = None 
        self._being_handle_stuck = None
        self._stuck_handle_begin_time = None
        
        self._speed_history = []
        self._throttle_history = []

        self._lapCount = 1
        self._previousLapCount = 1
        self._wrongWayLap = []
        self._hardcodeTrafficSignSequence = Settings.HARDCODE_TRAFFIC_SIGN_SEQUENCE
        self._useHardCodeTrafficSignBeforeLap = Settings.USE_HARDCODE_TRAFFIC_SEQUENCE_BEFORE_LAP
        self._trafficSignOrderNumber = 0
        self._previousLapCountWhenTrafficSignFound = 0
        self.frame_count = 0
        self._singlelanelazy = Settings.eagle_turn_time_single_lane
        self._guidelineAngle = 0
        
        self._last_guidelineangle_time = 0
        self._guidelineangle_time = 0
        self._last_guidelineangle_list = np.zeros(5, dtype=np.int)
        self._last_guidelineangle = 0 

        self._double_lane = False
        self._double_lane_time = 0
        self._on_side = 0

        self.last_response_time = 0
        self.start_left = -30
        self.start_right = 30
        self.last_see_six_road_count = 0
        self._single = True
        self.trace_road = 5
        self.road_change = False
        self.last_pos = [160]*10
        #self.frame_count = 0
        self._previousLapCount = 1        

        self.trace_road = 5
        self.changing_road = False
        self.backing = False
        self.backing_count = 0
        self.starting = True
        self.starting_count = 0
        self._last_angle = 0
        self._begin_sharp_turn = 0

        self.last_line_number = 0

        
    @staticmethod
    def distance(p0, p1):
        #logit("(%s)<--> (%s)" % (p0, p1))
        
        if p0[0] == 0 and p0[1] == 0:
            return 0 
        else:
            return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) 

    def GetMin(self, item1, item2):
        item = 0
        if item1 != 0 and item2 != 0:
            item = min(item1, item2)
        elif item1 != 0 or item2 != 0:
            item = item1 if item1 != 0 else item2
        return item

    def QuartWallProcess(self):
        toturnagle = 0
        self._wallfound = True
        self._nearwall = False
        quarters = self._quaters
        
        leftmin = self.GetMin(quarters._lefttobottom, quarters._leftmidtobottom)
        rightmin = self.GetMin(quarters._righttobottom, quarters._rightmidtobottom)
        
        if leftmin !=0 or rightmin !=0:
            self._nearwall = True
            
        minlen = self.GetMin(leftmin, rightmin)
       
        if self._lanepoint[0] == 0 and self._lanepoint[1] == 0:
            if quarters._turnaction == CarAction.forward:
                self.Backwards("No wall, no lanes")
            elif quarters._turntopointbottom >= Settings.eagle_wall_turn_Y and minlen > Settings.eagle_wall_distance_Y:
                self._wallsharpturn = True
                #self.Backwards("_wallsharpturn pointY, and minlen: %s %s " %
                #(quarters._turntopointbottom, minlen))
            
                if (quarters._turnaction == CarAction.toleft or quarters._turnaction == CarAction.turnahead) and quarters._angletoturn != 0: #turn left
                    toturnagle = quarters._angletoturn
                    self._action = "wall to left"
                elif (quarters._turnaction == CarAction.toright or quarters._turnaction == CarAction.turnahead) and quarters._angletoturn != 0: 
                    toturnagle = quarters._angletoturn
                    self._action = "wall to right"
                else:
                    self.Backwards("debuginfo: exception: can not get a valid angle")

                if quarters._purewall != PureWall.nowall:
                    if quarters._purewall == PureWall.onleft:
                        toturnagle = 30
                        self._action = "debuginfo: pure wall to right"
                    elif quarters._purewall == PureWall.onRight:
                        toturnagle = -30
                        self._action = "debuginfo: pure wall to left"
                    else:
                        self.Backwards("debuginfo: backwards pure wall on both")
            else:
                if minlen < Settings.eagle_wall_distance_Y:
                    self.Backwards("debuginfo: %s minlen, action backward" % minlen)
        else:
            toturnagle = self._lanesangle
          
            if quarters._turnaction == CarAction.turnahead:
                self._action = "debuginfo: wall ahead turnahead sharp turn"
                self._wallsharpturn = True
            
            xinterval = abs(self._corppedimageX - quarters._pointtoturn[0]) 
            if quarters._turnaction == CarAction.toleft and quarters._angletoturn != 0 and xinterval < Settings.eagle_wall_distance_X:
                if self._lanesangle > 0:
                    toturnagle = quarters._angletoturn
                else:
                    toturnagle = quarters._angletoturn if abs(quarters._angletoturn) > abs(self._lanesangle) else self._lanesangle
                
                self._action = "wall to left21"
            elif quarters._turnaction == CarAction.toright and quarters._angletoturn != 0 and xinterval < Settings.eagle_wall_distance_X:
                    if self._lanesangle < 0:
                        toturnagle = quarters._angletoturn
                    else:
                        toturnagle = quarters._angletoturn if abs(quarters._angletoturn) > abs(self._lanesangle) else self._lanesangle
                        self._action = "wall to right22" if toturnagle > 0 else "wall to left22"

            if minlen <= Settings.eagle_wall_turn_Y or xinterval < Settings.eagle_wall_distance_X - 10:
                self._wallsharpturn = True
                self._action = "wall to right sharp turn" if toturnagle > 0 else "wall to left sharp trun"
    
        self._steering_angle_to_return = toturnagle
        
        # if self._backward ==False:
        #     self._steering_angle_to_return = toturnagle if toturnagle !=0
        #     else self._last_lanes_angle

        return toturnagle

    def Backwards(self, action):
        self._action = "backwards: %s" % action
        self._wallsharpturn = False
        self._backward = True
        #print self._action

    # return None if not stuck, otherwise, return any of 'right', 'left',
    # 'center'
    def _is_stuck(self, img):
        if self._wrongway_handle_begin_time is not None and time.time() - self._wrongway_handle_begin_time<5.0:
            return False

        if len(self._speed_history)>=60 and sum(self._speed_history[-20:])<0.08*20:
            if sum(self._throttle_history) < -3.0:
                self._stuck_direction = 'back'
            else:
                self._last_road_condition = TrafficSignType.check_wall_obstacle(img, Settings.SeeWallFactor)
                if self._last_road_condition in [Globals.StuckBlackWall, Globals.StuckRGWall]:
                    left_wall_count = 0
                    right_wall_count = 0

                    #logit('next_to_wall_history
                    #{}'.format(self._next_to_wall_history))
                    for next_to_wall in self._next_to_wall_history:                
                        if next_to_wall < 0.5:
                            left_wall_count += 1
                        elif next_to_wall > 0.5:
                            right_wall_count += 1

                    if left_wall_count > right_wall_count and left_wall_count > 0:               
                        self._stuck_direction = 'left'
                    else:
                        self._stuck_direction = 'right'                    
                elif self._last_road_condition == Globals.StuckObstacle:
                    # stuck to the car
                    self._stuck_direction = 'centercar'
                else:
                    self._stuck_direction = 'center'                
            
            logit('stuck_direction=%s' % (self._stuck_direction))
            self._next_to_wall_history = []
            self._last_road_condition = None
            self._speed_history = []
            return True
        else:
            return False

    def _handle_stuck(self, img):
        time_now = time.time()            
        if self._stuck_direction in ['left', 'right'] and time_now - self._stuck_handle_begin_time < Settings.HANDLE_STUCK_TIME:                         
            # 如果是向右撞墙（一般情况几乎和墙垂直），需要回到右后方，方向右转倒车:steering_angle设到20度，throttle设为-0.1倒车
            # 如果是向左撞墙，需要回到左后方，方向左转倒车:steering_angle设到-20度，throttle设为-0.1倒车
            self._steering_angle_to_return = Settings.HANDLE_STUCK_ANGLE_VALUE
            self._throttle_to_return = Settings.HANDLE_STUCK_THROTTLE_VALUE
            if self._stuck_direction == 'left':
                self._steering_angle_to_return = -1 * Settings.HANDLE_STUCK_ANGLE_VALUE
        elif self._stuck_direction == 'centercar':
            #正向行驶撞小车，倒车
            if time_now - self._stuck_handle_begin_time >= Settings.HANDLE_STUCK_TIME * 0.5:
                self._being_handle_stuck = False
                self._steering_angle_to_return = 0.0
                self._throttle_to_return = 0.1                         
            else:  
                self._steering_angle_to_return = 10.0
                self._throttle_to_return = -0.5
        elif self._stuck_direction == 'center':
            if time_now - self._stuck_handle_begin_time >= Settings.HANDLE_STUCK_TIME * 0.5:
                self._being_handle_stuck = False
                self._steering_angle_to_return = 0.0
                self._throttle_to_return = 0.1                         
            else:  
                self._steering_angle_to_return = 0
                self._throttle_to_return = -0.5           
        elif self._stuck_direction == 'back':
            #后退撞墙，简单右转前进 （又可能走错方向，由后面的wrongway流程处理）
            if time_now - self._stuck_handle_begin_time >= Settings.HANDLE_STUCK_TIME:
                self._being_handle_stuck = False
                self._steering_angle_to_return = 0.0
                self._throttle_to_return = 0.1                         
            else:  
                self._steering_angle_to_return = 0.5
                self._throttle_to_return = 0.2
        else:
            self._being_handle_stuck = False
            self._steering_angle_to_return = 0.0
            self._throttle_to_return = 0.1                             

    def _is_wrong_way(self):
        #发现方向走错
        if self._last_road_condition == Globals.WrongWay and (time.time() - self._last_road_condition_time) < 1.0:
            left_wall_count = 0
            right_wall_count = 0
            for next_to_wall in self._next_to_wall_history:                
                if next_to_wall < 0.50:
                    left_wall_count += 1
                elif next_to_wall > 0.50:
                    right_wall_count += 1
            if left_wall_count > right_wall_count and left_wall_count > 3:               
                self._wrongway_direction = 'left'
            elif right_wall_count > left_wall_count and right_wall_count > 3:
                self._wrongway_direction = 'right'
            else:
                self._wrongway_direction = 'center'

            logit('wrongway_direction=%s' % (self._wrongway_direction))
            self._next_to_wall_history = []
            self._last_road_condition = None
            self._last_road_condition_time = None
            self._speed_history = []
            self._wrongWayLap.append(self._lapCount)
            return True
        else:
            self._previousLapCount = self._lapCount                
            return False

    def _handle_wrong_way(self, img):
        time_now = time.time()
        # 如果是向右靠墙，左边空间宽裕，方向左转倒车然后再右转前进
        # 如果是向左靠墙，右边空间宽裕，方向右转倒车然后再左转前进
        steering_angle = 0.0
        throttle = 0.0
        if self._wrongwayhandle_step_1_time_cost == 0:
            throttle = Settings.HANDLE_WRONGWAY_THROTTLE_VALUE
            steering_angle = Settings.HANDLE_WRONGWAY_ANGLE_VALUE              
            if  time_now - self._wrongway_handle_begin_time > Settings.HANDLE_WRONGWAY_TIME * 0.70:
                self._wrongwayhandle_step_1_time_cost = time_now - self._wrongway_handle_begin_time     
                logit('in _handle_wrong_way, go ahead')      
                if self._wrongway_direction == 'left':                        
                    steering_angle = -1 * Settings.HANDLE_WRONGWAY_ANGLE_VALUE
            else:
                #倒车
                #logit('in _handle_wrong_way, turn around')
                if self._wrongway_direction == 'right':
                    steering_angle = -1 * Settings.HANDLE_WRONGWAY_ANGLE_VALUE                  
        elif time_now - self._wrongway_handle_begin_time <= Settings.HANDLE_WRONGWAY_TIME:    
            #logit('in _handle_wrong_way, go ahead')
            throttle = -1 * Settings.HANDLE_WRONGWAY_THROTTLE_VALUE
            steering_angle = Settings.HANDLE_WRONGWAY_ANGLE_VALUE      
            if self._wrongway_direction == 'left':                        
                steering_angle = -1 * Settings.HANDLE_WRONGWAY_ANGLE_VALUE
        else:
            #
            logit('complete _handle_wrong_way')
            steering_angle = 0.0
            throttle = 0.1
            self._being_handle_wrongway = False
            self._wrongwayhandle_step_1_time_cost = 0
        self._steering_angle_to_return = steering_angle
        self._throttle_to_return = throttle

    def on_dashboard(self, src_img, converted_last_steering_rad, last_steering_angle, speed, throttle, info):
        try:
            if self.frame_count == 0:
                logit('race started')

            self.frame_count += 1
            self._lapCount = int(info['lap'])
            self._roadcondition.add_latest_frame(src_img)
            if speed > 0.8:
                next_to_wall = TrafficSignType.check_wall_direction(src_img, 0.02)
                if next_to_wall is not None:
                    self._next_to_wall_history.append(next_to_wall)
                    self._next_to_wall_history = self._next_to_wall_history[-120:]
            self._speed_history.append(speed)
            self._speed_history = self._speed_history[-60:]
            if self._being_handle_wrongway:         
                self._handle_wrong_way(src_img)        
            elif self._is_wrong_way():
                logit('_is_wrong_way')
                self._being_handle_wrongway = True
                self._wrongway_handle_begin_time = time.time()
                self._handle_wrong_way(src_img)
            elif self._being_handle_stuck:
                self._handle_stuck(src_img)
            elif self._is_stuck(src_img):
                logit('_is_stuck')
                # time point when stuck is found
                self._being_handle_stuck = True 
                self._stuck_handle_begin_time = time.time()
                self._handle_stuck(src_img)
            else:
                self.HandleTrafficeSign()

                self._lanesangle, self._lanepoint, self._mapcount, self._airstangle, self._middleareaangle, self._targetxy, self._blackwallimg = GreyLines.GetLanesAngle(src_img, self._changelane_direction)   
                #print("self._mapcount %s" % self._mapcount)
                if self._mapcount >= 2:
                    self._last_guidelineangle_list[self.frame_count % 5] = self._mapcount
                elif self._mapcount == 1:
                    self._last_guidelineangle_list[self.frame_count % 5] = -1
                else:
                    self._last_guidelineangle_list[self.frame_count % 5] = 0

                self.HandleLaneInfo()

                left_pos, right_pos, top_pos, bottom_pos, bFound = GreyLineEx.checkObstacle_v2(src_img)

                if self._lanesangle == 0 or (time.time() - self._lastanglezerotime < Settings.eagle_high_misslane_time) or bFound or self._single == True: 
                    print("In process of turn %s" % self._changelane_direction) 
                    if self._changelane_direction == -1:
                        self.trace_road = 2
                        #self.changing_road = True
                    elif self._changelane_direction == 1:
                        self.trace_road = 8
                        #self.changing_road = True
                    elif self._changelane_direction == 0:
                        # isSeeSixRoad = GreyLines.SeeSixRoad(src_img)
                        if self.last_see_six_road_count > 3:
                            self.trace_road = 5

                    self.use_white = True
                    self.AdjustAngleAndThrottle_v2(speed, src_img, self.trace_road)
                else:
                    #print("-- double lane")
                    WallAndLanes, Walls, blackwallimg = GreyLines.GetEdgeImages(src_img) # Get lanes and wall edges
                    
                    self._corppedimageX = blackwallimg.shape[1]
                    self._corppedimageY = blackwallimg.shape[0]
                    
                    #self._on_side 
                    
                    # if self._lanesangle == 0 and self._double_lane == False:
                    #     self._lanesangle  = self._last_lanes_angle 
                    self._quaters = GreyLines.GetWallAngle(Walls, blackwallimg)

                    #there's a wall
                    self._wallfound = False
                    self._backward = False
                    self._wallsharpturn = False
                    if self._quaters != None and (self._quaters._turnaction != CarAction.forward or (self._lanepoint[0] == 0 and self._lanepoint[1] == 0)):
                        toturnagle = self.QuartWallProcess()
                        if Settings.DEBUG:
                            infolist = "see wall action:(to %s, %s) lane (%s %s)" % (self._action, toturnagle, self._lanesangle, self._lanepoint)
                        #logit(infolist)
                        #print (infolist)
                    else:
                        self._steering_angle_to_return = self._lanesangle

                    self.AdjustAngleAndThrottle(speed, src_img)

                    # if bFound == True:
                    #     self.HandleObstacle(left_pos, right_pos, top_pos, bottom_pos)
                    
                    if Settings.DEBUG_IMG:
                        ImageProcessor.show_image(blackwallimg, "blackwallimg")

                    if self._throttle_to_return == 0:
                        self._throttle_to_return = 0.03
                    if self._lanesangle != 0:
                        self._last_lanes_angle = self._lanesangle

            self._throttle_history.append(self._throttle_to_return)
            self._throttle_history = self._throttle_history[-30:]
            
            # if abs(self._steering_angle_to_return) < 1 and
            # self._steering_angle_to_return != 0:
            #     self._steering_angle_to_return = 1 if
            #     self._steering_angle_to_return > 0 else -1
            #print "Final: angle and throttle (%s, %s)" %
            #(self._steering_angle_to_return, self._throttle_to_return)
            if self._throttle_to_return == 0:
                self._throttle_to_return = 0.03
            
            #if self._steering_angle_to_return == 0 or self._lanesangle:

            self._car.control(self._steering_angle_to_return, self._throttle_to_return)

            if self._lanesangle != 0:
                    self._last_lanes_angle = self._lanesangle
        except Exception as exception:
            logger.error(exception)
            traceback.print_exc()


    def AdjustAngleAndThrottle_v2(self, speed, src_img, trace_road):

        # 1.5m/s
        # P_steer = 0.1
        # D_steer = 0.2
        # Road color is[Red, Blue,  Red, Green, Blue, Green]
        # trace_road is[ 0  1  2  3  4  5  6   7  8  9  10 ]
        # trace_road = 8

        P_steer = 0.06
        D_steer = 0.08
        if trace_road == 5:
            # if self.use_white:
            #     P_steer = 0.13
            #     D_steer = 0.1
            # else:
            #     P_steer = 0.1
            #     D_steer = 0.1
            P_steer = 0.13
            D_steer = 0.1
            speed_set = 2.5
        elif trace_road == 2 or trace_road == 8:
            P_steer = 0.1
            D_steer = 0.17
            speed_set = 2.5

        P_speed = 1
        D_speed = 0


        
        
        ##print("self.last_pos",self.last_pos)
        if (self._changelane_direction == -1 or self._changelane_direction == 1) and self._last_changelane_direction == 0:
            self.changing_road = True
        # if trace_road == 2 and self._changelane_direction == 1:
        #     self.changing_road = True
        start_time = time.time()
        car_pos, isNearWall, isSeeSixRoad, label_number = GreyLineEx.GetCarPos(src_img, self.last_pos, trace_road, speed,(self.frame_count+9)%10,self.changing_road,self._changelane_direction,self.use_white)
        
        if trace_road != 5:
            if isSeeSixRoad and self._changelane_direction == 0:
                self.last_see_six_road_count += 1
            else:
                self.last_see_six_road_count = 0
        else:
            self.last_see_six_road_count = 0
        print("get car pos cost time",time.time()- start_time,"label_number",label_number)
        #print("car_pos",car_pos,"isNearWall",isNearWall,"self._changelane_direction",self._changelane_direction)
        if isNearWall and trace_road == 5:
            P_steer = 0.2
            D_steer = 0
        elif trace_road == 5:
            P_steer = 0.13
            D_steer = 0.1
            # if self.use_white:
            #     P_steer = 0.13
            #     D_steer = 0.1
            # else:
            #     P_steer = 0.1
            #     D_steer = 0.1

        # if trace_road == 2 and self.changing_road and self._changelane_direction == 1 and label_number == 1:
        #     label_number = 2

        
        if self.changing_road and label_number != 1:#(car_pos < 40 or car_pos > 280):      
            #print "self._lanepoint[0]",self._lanepoint[0]
            #print "Changing road....." 
            if self._changelane_direction == -1:
                car_pos = 105
            elif self._changelane_direction == 1:
                car_pos = 215
            else:
                car_pos = 160
            #D_steer = 0
        else:
            self.changing_road = False
            #self.pos_start_left = 120
            #D_steer = 0.1
        

        # print "self._changelane_direction", self._changelane_direction
        # if self._changelane_direction == -1:
        #     car_pos = 0
        # elif self._changelane_direction == 1:
        #     car_pos = 320

        car_angle = ( car_pos - (src_img.shape[1]/2) )
        # if abs(car_angle)>140 and (trace_road == 2 or trace_road == 8):
        #     P_steer = 0.5
        #     # D_steer = 0.1
        # if abs(car_angle)>140 and trace_road == 5:
        #     P_steer = 0.5
        #     # D_steer = 0
        # if (trace_road == 2 or trace_road == 8) and abs(car_angle) <50:
        #     speed_set = 2.5
 
        

        self.last_pos[self.frame_count%10] = car_pos
        #self.frame_count+=1
        car_angle_temp = car_angle
        if car_angle_temp < 0:
            car_angle_temp =  car_angle_temp

        self._steering_angle_to_return = 0.0 + P_steer*car_angle_temp + D_steer * (car_angle_temp - self._last_angle)

        if trace_road == 5:
            if abs(car_angle)>140  and isNearWall == False:
                if car_angle > 140:
                    self._steering_angle_to_return = 40
                    
                else:
                    self._steering_angle_to_return = self.start_left
                    self.start_left-=1
                # D_steer = 0
            else:
                self.start_left = -30
        else:
            if abs(car_angle)>140:
                if car_angle > 140:
                    self._steering_angle_to_return = self.start_right
                    self.start_right+=2
                else:
                    self._steering_angle_to_return = self.start_left
                    self.start_left-=2
            else:
                self.start_left = -40
                self.start_right = 40
        print ("self._steering_angle_to_return",self._steering_angle_to_return)
        
        self._last_changelane_direction = self._changelane_direction
        ##print("self._steering_angle_to_return",self._steering_angle_to_return)
        # if abs(car_angle) != 160:
        #     self._last_angle = car_angle_temp
        # else:
        #     self._last_angle = 160

        self._last_angle = car_angle

        self._throttle_to_return = P_speed*(speed_set - speed)

        if isNearWall:
            if speed > 2.5:
                self._throttle_to_return = -1

    #前方障碍物或fork road，变道
    def HandleTrafficeSign(self):
        if self._last_traffic_sign_time is None:
            return False

        time_now = time.time()
        if (self._last_traffic_sign==Globals.ForkLeftSign or self._last_traffic_sign==Globals.RightObstacle) and (time_now-self._last_traffic_sign_time) < 4:
            #turn left
            self._changelane_direction = -1
            return True
        elif (self._last_traffic_sign==Globals.ForkRightSign or self._last_traffic_sign==Globals.LeftObstacle) and (time_now-self._last_traffic_sign_time) < 4:
            #turn right
            self._changelane_direction = 1
            return True
        else:
            self._changelane_direction = 0
            return False  
    
    def AdjustAngleAndThrottle(self, speed, src_img):
        self.SpeedControl(speed)

        anratio = self.BaseAngleAdjust()
     
        # if self._wallfound == True:
        #    if self._double_lane == True and self._nearwall == False:
        #        anratio += 1
        #    # if speed >= 0 and self._quaters._purewall == True:
        #    # anratio = 1
            
        self._steering_angle_to_return = self._steering_angle_to_return / anratio

        #if self._double_lane == True and self._airstangle != 0:
        #    self._steering_angle_to_return = 45 if self._airstangle > 0 else
        #    -45
        #    print ("middle lanes")

        if self._backward == True:
            if self._last_lanes_angle > 0:
                self._steering_angle_to_return = -45
            else:
                self._steering_angle_to_return = 45 
        
        self.HandleUTurn(speed)
        self.HandleForkSign()

    def HandleForkSign(self):
        #handle road fork sign
        if self._changelane_direction != 0 and self._last_traffic_sign in [1, 2]:
            if (time.time() - self._last_traffic_sign_time) < 1.5:
                self._steering_angle_to_return = self._lanesangle
                print("*fork turn")
            else:
                if self._quaters._turnaction == CarAction.turnahead:
                    if (self._quaters._leftangle != 0 and self._quaters._lefthalftobottom > Settings.eagle_far_wall_distanceY) or (self._quaters._rightangle != 0 and self._quaters._righthalftobottom > Settings.eagle_far_wall_distanceY):
                        self._steering_angle_to_return = self._steering_angle_to_return / 2
          

    def HandleUTurn(self, speed): 
        if self._on_side == -1:
            self._steering_angle_to_return = -40
            print("---turn left")
        elif  self._on_side == 1:
            self._steering_angle_to_return = 40
            print("---turn right")
            
        if min(self._quaters._lefthalftobottom, self._quaters._righthalftobottom) > Settings.eagle_far_wall_distanceY_Uturn:
            if self._double_lane == True:
                self._throttle_to_return = Settings.eagle_max_turn_speed_U_brake
        else:
            if self._quaters._turnaction == CarAction.turnahead and speed > Settings.eagle_max_turn_speed:
                    self._throttle_to_return = Settings.eagle_max_turn_speed_brake
                    info = "slow down speed %s throttle %s angle %s " % (speed, self._throttle_to_return, self._steering_angle_to_return)
                    print (info)
                    logit(info)

    def HandleLaneInfo(self):
        forksign = False
        currenttime = time.time()
       
        a = np.array(self._last_guidelineangle_list)
        
        if a[a < 0].size >= 3:
            self._single = True
            self._double_lane = False
            self._double_lane_time = currenttime
            self._last_guidelineangle_list = np.zeros(5, dtype=np.int)
            #print ("single lane set to True")
            logit("double lane set to True") 
            # In case of stuck status

        #print ("self._last_guidelineangle_list: %s" % self._last_guidelineangle_list)
        if (a[a > 0].size > 2 or a[a > 3].size > 1 ) and self._changelane_direction == 0:
            self._double_lane = True
            self._single = False
            self._double_lane_time = currenttime
            #print ("double lane set to True")
            logit("double lane set to True") 
            self._last_guidelineangle_list = np.zeros(5, dtype=np.int)
            # In case of stuck status
        

        if self._changelane_direction != 0 and self._last_traffic_sign in [1, 2]:
            forksign = True
            #self._singlelanelazy = Settings.eagle_turn_time_single_lane * 2 if forksign == True else Settings.eagle_turn_time_single_lane
            self._double_lane = False
            self._last_guidelineangle_list = np.zeros(5, dtype=np.int)
        
        if self._lanesangle == 0:
            self._lastanglezerotime = currenttime

    def BaseAngleAdjust(self):
        anratio = Settings.eagle_angle_ratio
        absangble = abs(self._steering_angle_to_return)   
        if absangble > 45:
            anratio = 1
            #self._throttle_to_return = 0.01
        if absangble > 40:
            anratio = 1.1
            #self._throttle_to_return = 0.02
        elif absangble > 30:
            anratio = 2
            #self._throttle_to_return = 0.03
        elif absangble > 20:
            anratio = 4
            #self._throttle_to_return = 0.04
        elif absangble > 10:
            anratio = 5
            #self._throttle_to_return = 0.04
        elif absangble > 0:
            anratio = 3                    
            #self._throttle_to_return = 0.04
        return anratio  

    def SpeedControl(self, speed):
        # if speed > Settings.eagle_maxspeed:
        #     self._throttle_to_return = Settings.eagle_maxspeed_throttle
        # elif speed >= 0:
        #     base = Settings.eagle_maxspeed_throttle
        #     if speed < 0.5:
        #         self._throttle_to_return = 0.5 + base
        #     elif speed < 0.8:
        #         self._throttle_to_return = 0.3 + base
        #     elif speed < 1.0:
        #         self._throttle_to_return = 0.2 + base
        #     elif speed < 1.2:
        #         self._throttle_to_return = 0.1 + base
        #     elif speed < 1.5:
        #         self._throttle_to_return = 0.02 + base
        #     elif speed < 1.8:
        #         self._throttle_to_return = 0.015 + base
        #     elif speed < 2.1:
        #         self._throttle_to_return = 0.015 + base
        #if self._throttle_to_return < Settings.eagle_maxspeed_throttle:
        self._throttle_to_return = Settings.eagle_maxspeed_throttle    

        if self._backward == True:
            self._throttle_to_return = -0.95

    def reload_setting(self):
        self._roadcondition.update_setting(Settings.ROAD_CHECK_FRAME_RATE, 
                Settings.StraightSpeed, 
                Settings.WALL_CHECK_PIXEL_COUNT_FACTOR, 
                Settings.TRAFFIC_SIGN_PIXEL_COUNT_THRESHOLD,
                Settings.DEBUG)

    def on_traffic_sign(self, sign_type, sign_box_ori):
        if self._previousLapCountWhenTrafficSignFound != self._lapCount:
            self._previousLapCountWhenTrafficSignFound = self._lapCount
            self._trafficSignOrderNumber = 0
        else:
            self._trafficSignOrderNumber += 1

        self._last_traffic_sign = sign_type
        self._last_traffic_sign_time = time.time()
        #print "lap: %d, trafice sign: %s " % (self._lapCount,
        #self._last_traffic_sign)
        print("lap: %d, trafice sign: %s " % (self._lapCount, self._last_traffic_sign))
        if self._lapCount in [1,2] and sign_box_ori is not None:
            ImageProcessor.force_save_image_to_log_folder(sign_box_ori, prefix = "orisign", suffix=str(sign_type))

    def on_road_condition(self, condition):
        if not self._being_handle_stuck:
            # wrong wall, stuck wall, stuck obstacles
            print("road condition: %s" % condition)
            self._last_road_condition = condition
            self._last_road_condition_time = time.time()
