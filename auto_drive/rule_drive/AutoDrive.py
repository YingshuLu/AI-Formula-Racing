# coding=utf-8
import time
from ImageProcessor import ImageProcessor
import threading
from importlib import reload
import os
import math
import logger
from PID import PID
import Globals
import Settings
import TrafficSignType
from Car import Car
from RoadConditionCheck import RoadConditionCheck
from GreyLinesEx import GreyLines
from GreyLinesEx import CarAction
from GreyLinesEx import PureWall
import json
import random 
import traceback

logger = logger.get_logger(__name__)
def logit(msg):
    if Settings.DEBUG:
        logger.info("%s" % msg)

class AutoReloadSetting(threading.Thread):
    # Auto reload Settings thread
    def __init__(self, auto_drive):
        super(AutoReloadSetting, self).__init__()
        self.modificationStamp = 0
        self.auto_drive = auto_drive
        self.start()
        
	# Stream delegation loop
    def run(self):
        while Globals.Running:
            newModificationStamp = os.path.getmtime('Settings.py')
            if newModificationStamp != self.modificationStamp:
                reload(Settings)
                self.auto_drive.reload_setting()
                self.modificationStamp = newModificationStamp
                print('Settings reload')
            time.sleep(1)

STRAIGHT_ROAD_STATUS = 0
TURN_ROAD_STATUS = 1
UTURN_ROAD_STATUS = 2

class AutoDrive(object):
    debug = Settings.DEBUG

    def __init__(self, car, road_condition_check):
        speed_to_set = Settings.StraightSpeed
        SpeedPIDMapping = Settings.SpeedPIDMapping
        kp = SpeedPIDMapping[speed_to_set][0]
        ki = SpeedPIDMapping[speed_to_set][1]
        kd = SpeedPIDMapping[speed_to_set][2]   

        self._steering_pid     = PID(kp, ki, kd, max_integral=Settings.STEERING_PID_max_integral)
        self._throttle_pid     = PID(Kp=Settings.THROTTLE_PID_Kp  , Ki=Settings.THROTTLE_PID_Ki  , Kd=Settings.THROTTLE_PID_Kd  , max_integral=Settings.THROTTLE_PID_max_integral  )
        self._throttle_pid.assign_set_point(speed_to_set)
        self._steering_history = []

        self._previous_line_angle = 0.0

        self._steering_angle_to_return = 0
        self._throttle_to_return = 0

        # use to check whether stuck to wall on the right/left side
        # when stuck, if found 
        self._next_to_wall_history = []
        self._speed_history = []

        self._throttle_history = []
        self._car = car
        self._road_condition_check = road_condition_check
        self._road_condition_check.register(self)
        self._car.register(self)
        self.frameCount = 0

        self._last_traffic_sign = None
        self._last_traffic_sign_time = None
        self._last_road_condition = None
        self._last_road_condition_time = None

        # 撞向哪边位置的墙，决定了恢复时方向转角，比如撞到行进中左边的墙，则左转方向倒车
        self._stuck_direction = None 
        self._being_handle_stuck = None
        self._stuck_handle_begin_time = None

        # 行进中墙在哪边，决定了掉头时方向转角
        self._wrongway_direction = None 
        self._being_handle_wrongway = None
        self._wrongway_handle_begin_time = None
        self._wrongwayhandle_step_1_time_cost = 0 #为了在三赛道的场景下掉头，分两步：第一步指定转弯角度倒车与墙垂直，第二步指定转弯角度前进

       
        # 前方有fork road ahead或障碍物转向标志时，需要刹车减速指定时间，需要知道turn方向决定变道方向
        self._being_handle_changelane = None
        self._changelane_handle_begin_time= None
        self._changelane_direction = None #变道方向
        self._changelane_step_1_time_cost = 0 #变道过程，方向转角分为两个阶段：第一阶段往变道方向转，第二阶段再反方向回转方向

        #
        #self._steering_history_for_straight_decision = []
        self._straight_begin_time = None

        self._begin_sharp_turn = 0
        self._being_sharp_turn = False
        self._uturn_handle_time_begin = None
        self._lapCount = 0
    # return None if not stuck, otherwise, return any of 'right', 'left', 'center'
    def _is_stuck(self, img):
        if len(self._speed_history)>=60 and sum(self._speed_history[-20:])<0.08*20:
            if sum(self._throttle_history)<-3.0:
                self._stuck_direction = 'back'
            else:
                self._last_road_condition = TrafficSignType.check_wall_obstacle(img, Settings.SeeWallFactor)
                if self._last_road_condition in [Globals.StuckBlackWall, Globals.StuckRGWall]:
                    left_wall_count = 0
                    right_wall_count = 0

                    #logit('next_to_wall_history {}'.format(self._next_to_wall_history))
                    for next_to_wall in self._next_to_wall_history:                
                        if next_to_wall<0.49:
                            left_wall_count += 1
                        elif next_to_wall>0.51:
                            right_wall_count += 1

                    if left_wall_count>right_wall_count and left_wall_count>3:               
                        self._stuck_direction = 'left'
                    else:
                        self._stuck_direction = 'right'                    
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
        if self._stuck_direction in ['left', 'right'] and time_now - self._stuck_handle_begin_time<Settings.HANDLE_STUCK_TIME:                         
            # 如果是向右撞墙（一般情况几乎和墙垂直），需要回到右后方，方向右转倒车:steering_angle设到20度，throttle设为-0.1倒车
            # 如果是向左撞墙，需要回到左后方，方向左转倒车:steering_angle设到-20度，throttle设为-0.1倒车            
            self._steering_angle_to_return = Settings.HANDLE_STUCK_ANGLE_VALUE
            self._throttle_to_return  = Settings.HANDLE_STUCK_THROTTLE_VALUE
            if self._stuck_direction == 'left':
                self._steering_angle_to_return = -1*Settings.HANDLE_STUCK_ANGLE_VALUE
        elif self._stuck_direction == 'center':
            #正向行驶撞墙，倒车
            if time_now - self._stuck_handle_begin_time>=Settings.HANDLE_STUCK_TIME*2.5:
                self._being_handle_stuck = False
                self._steering_angle_to_return = 0.0
                self._throttle_to_return = 0.1                         
            else:  
                self._steering_angle_to_return = 0
                self._throttle_to_return = -1.0
        elif self._stuck_direction == 'back':
            #后退撞墙，简单右转前进 （又可能走错方向，由后面的wrongway流程处理）
            if time_now - self._stuck_handle_begin_time>=Settings.HANDLE_STUCK_TIME:
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
        if self._last_road_condition_time is None:
            return False

        #如果6秒前已经开始掉头，则忽略wrongway判读，避免一直掉头失败的场景
        time_now = time.time()
        if self._wrongway_handle_begin_time is not None and time_now - self._wrongway_handle_begin_time<6.0:
            return False
        
        #发现方向走错
        if self._last_road_condition==Globals.WrongWay and (time_now-self._last_road_condition_time)<1.0:
            left_wall_count = 0
            right_wall_count = 0
            for next_to_wall in self._next_to_wall_history:                
                if next_to_wall<0.49:
                    left_wall_count += 1
                elif next_to_wall>0.51:
                    right_wall_count += 1
            if left_wall_count>right_wall_count and left_wall_count>3:               
                self._wrongway_direction = 'left'
            elif right_wall_count>left_wall_count and right_wall_count>3:
                self._wrongway_direction = 'right'
            else:
                self._wrongway_direction = 'center'

            logit('wrongway_direction=%s' % (self._wrongway_direction))
            self._next_to_wall_history = []
            self._last_road_condition = None
            self._last_road_condition_time = None
            self._speed_history = []
            return True
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
            if  time_now - self._wrongway_handle_begin_time>Settings.HANDLE_WRONGWAY_TIME*0.70:
                self._wrongwayhandle_step_1_time_cost = time_now - self._wrongway_handle_begin_time     
                logit('in _handle_wrong_way, go ahead')      
                if self._wrongway_direction == 'left':                        
                    steering_angle = -1 * Settings.HANDLE_WRONGWAY_ANGLE_VALUE
            else:
                #倒车
                #logit('in _handle_wrong_way, turn around')              
                if self._wrongway_direction == 'right':
                    steering_angle = -1 * Settings.HANDLE_WRONGWAY_ANGLE_VALUE                  
        elif time_now - self._wrongway_handle_begin_time<=Settings.HANDLE_WRONGWAY_TIME:    
            #logit('in _handle_wrong_way, go ahead')
            throttle = -1*Settings.HANDLE_WRONGWAY_THROTTLE_VALUE
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
                
    #前方障碍物或fork road，变道   
    def _is_changelane_ahead(self, src_img):
        if self._last_traffic_sign_time is None:
            return False

        time_now = time.time()
        if (self._last_traffic_sign==Globals.ForkLeftSign or self._last_traffic_sign==Globals.RightObstacle) and (time_now-self._last_traffic_sign_time)<0.5:
            wall_is_seen = TrafficSignType.wall_is_seen(src_img, 'left', Settings.SeeWallFactor*0.8)
            if wall_is_seen:
                logit('wall_is_seen, no change lane')
                if self._last_traffic_sign==Globals.RightObstacle:
                    #虽然不需要变道，但是需要降速到turn级别
                    
                    self._last_traffic_sign = Globals.LeftTurn
                return False
            else:
                self._changelane_direction = 'left'
                return True
        elif (self._last_traffic_sign==Globals.ForkRightSign or self._last_traffic_sign==Globals.LeftObstacle) and (time_now-self._last_traffic_sign_time)<0.5:
            wall_is_seen = TrafficSignType.wall_is_seen(src_img, 'right', Settings.SeeWallFactor*0.8)
            if wall_is_seen:
                logit('wall_is_seen, no change lane')
                if self._last_traffic_sign==Globals.RightObstacle:
                    #虽然不需要变道，但是需要降速到turn级别
                    
                    self._last_traffic_sign = Globals.RightTurn
                return False
            else:            
                self._changelane_direction = 'right'
                return True
        else:
            self._changelane_direction = None
            return False   

    
    def _handle_changelane(self, src_img, throttle):
        time_now = time.time()
        self._throttle_to_return = throttle 
        if time_now - self._changelane_handle_begin_time>self._handle_change_time*2:
            self._being_handle_changelane = False
            self._changelane_step_1_time_cost = 0
            self._steering_angle_to_return = 0.0         
        else:
            steering_angle = Settings.ChangeLaneSteeringAngleUnit
            if self._changelane_step_1_time_cost == 0:
                wall_is_seen = TrafficSignType.wall_is_seen(src_img, self._changelane_direction, Settings.SeeWallFactor*0.8)
                if wall_is_seen and time_now - self._changelane_handle_begin_time>=self._handle_change_time*0.60:
                    self._changelane_step_1_time_cost = time_now - self._changelane_handle_begin_time
                    #反方向打方向，视野回归道路前方          
                    #logit('in changelane, return back')      
                    if self._changelane_direction == 'right':                        
                        steering_angle = -1 * Settings.ChangeLaneSteeringAngleUnit
                else:
                    #继续变道
                    #logit('in changelane, continue')              
                    if self._changelane_direction == 'left':
                        steering_angle = -1 * Settings.ChangeLaneSteeringAngleUnit                  
            elif time_now - self._changelane_handle_begin_time<=self._changelane_step_1_time_cost*1.4:
                #反方向打方向，视野回归道路前方     
                #logit('in changelane, return back')           
                if self._changelane_direction == 'right':
                    steering_angle = -1 * Settings.ChangeLaneSteeringAngleUnit
            elif time_now - self._changelane_handle_begin_time<self._handle_change_time:
                steering_angle = 0.0
            else:
                #变道结束
                steering_angle = 0.0
                self._being_handle_changelane = False
                self._changelane_step_1_time_cost = 0
            self._steering_angle_to_return = steering_angle  

    # 前方直线路况，加速行驶   
    def _is_straight_ahead(self, current_angle_deg):
        if current_angle_deg>20.0:
            return False

        time_now = time.time()

        # 如果已经刚开始直线加速，不用重复判断
        if self._straight_begin_time is not None and time_now-self._straight_begin_time<1.5:
            return False

        return True



    # 前方有转弯标识，降速行驶
    def _is_uturn_ahead(self, speed, current_angle_deg):
        time_now = time.time()
        if self._last_traffic_sign_time is not None and (time_now-self._last_traffic_sign_time)<0.15 and self._last_traffic_sign in [Globals.LeftUTurn, Globals.RightUTurn]:
            return True
        else:
            return False

    def _is_turn_ahead(self, speed, current_angle_deg):
        time_now = time.time()
        if self._last_traffic_sign_time is not None and (time_now-self._last_traffic_sign_time)<0.15 and self._last_traffic_sign in [Globals.LeftTurn, Globals.RightTurn]:
            return True
        else:
            return False


    def _is_sharp_turn_ahead(self, src_img):
        if Settings.AlwaysDisableSharpTurnCheck:
            return False

        sharp_turn_ahead = time.time()-self._begin_sharp_turn>1.5 and TrafficSignType.check_sharp_turn_ahead(src_img, Settings.SharpTurnWallFactor)
        return sharp_turn_ahead

    def _handle_sharp_turn(self, src_img, converted_last_steering_rad):
        if time.time()-self._begin_sharp_turn>Settings.HandleSharpTurnTime:
            self._being_sharp_turn = False
            return 
        # when being handle sharp turn, also check the current road to extend the sharp turn handling time
        if self._is_sharp_turn_ahead(src_img):            
            self._begin_sharp_turn = time.time()

        track_img     = ImageProcessor.preprocess(src_img)
        current_angle = ImageProcessor.find_steering_angle_by_line(track_img, converted_last_steering_rad, Settings.DEBUG)
        current_angle_deg = ImageProcessor.rad2deg(current_angle)
        if Globals.RecordFolder is not None and Settings.DEBUG_IMG and self.frameCount % Settings.VERBOSE_DEBUG_FRAME_COUNT==0:                        
            ImageProcessor.show_image(track_img, "track")                

        # 避免其它line的干扰
        if time.time()-self._begin_sharp_turn>0.2 and abs(ImageProcessor.rad2deg(self._previous_line_angle))>0 and abs(current_angle_deg-ImageProcessor.rad2deg(self._previous_line_angle))>Settings.WrongAngleThreshold:
            #logit('wrong angle detected, use the previous one')
            current_angle = self._previous_line_angle
            current_angle_deg = ImageProcessor.rad2deg(current_angle)
        self._previous_line_angle = current_angle 
        steering_angle = self._steering_pid.update(-current_angle)
        self._steering_angle_to_return = ImageProcessor.rad2deg(steering_angle)
        return current_angle_deg
    
    def distance(self, p0, p1):
        if p0[0] == 0 and p0[1] == 0:
            return 0 
        else:
            return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2) 

    def Backwards(self, action):
        self._action = "backwards: %s" % action
        self._wallsharpturn = False
        self._backward =True
        
    
    def BaseAngleAdjust(self):
        anratio = Settings.eagle_angle_ratio
        absangble = abs(self._steering_angle_to_return)   
        if absangble > 45:
            anratio = 1.6
            #self._throttle_to_return = 0.01
        if absangble > 40:
            anratio = 2.5
            #self._throttle_to_return = 0.02
        elif absangble > 30:
            anratio = 4
            #self._throttle_to_return = 0.03
        elif absangble > 20:
            anratio = 5
            #self._throttle_to_return = 0.04
        elif absangble > 10:
            anratio = 6
            #self._throttle_to_return = 0.04
        elif absangble >0:
            anratio = 6                    
            #self._throttle_to_return = 0.04
        return anratio  

    def on_dashboard(self, src_img, converted_last_steering_rad, last_steering_angle, speed, throttle, info):
        self._lapCount = info['lap']        
        try:
            if self.frameCount == 0:
                logit('race started')
            self.frameCount += 1
            # self._steering_history_for_straight_decision.append(abs(last_steering_angle))
            # self._steering_history_for_straight_decision = self._steering_history_for_straight_decision[-Settings.StraightDecisionFrameCount:]

            verbose_debug_frame_count = Settings.VERBOSE_DEBUG_FRAME_COUNT
            if Settings.DEBUG and self.frameCount%verbose_debug_frame_count==0:
                logit('in on_dashboard, frameCount=%d, last_steering_angle=%f' % (self.frameCount, last_steering_angle))
                logit("info: %s" % json.dumps(info))

            SpeedPIDMapping = Settings.SpeedPIDMapping
            current_angle_deg = 0.0
            current_angle = 0.0

            if speed>0.4:
                next_to_wall = TrafficSignType.check_wall_direction(src_img, 0.1)
                if next_to_wall is not None:
                    self._next_to_wall_history.append(next_to_wall)
                    self._next_to_wall_history = self._next_to_wall_history[-30:]
            self._speed_history.append(speed)
            self._speed_history = self._speed_history[-60:]
            self._road_condition_check.add_latest_frame(src_img)  

            auto_follow_mode = True

            if self._being_handle_stuck:
                self._handle_stuck(src_img)
                auto_follow_mode = False
            elif self._being_handle_wrongway:         
                self._handle_wrong_way(src_img)
                auto_follow_mode = False
            elif self._being_handle_changelane:
                throttle = self._throttle_pid.update(speed)
                self._handle_changelane(src_img, throttle)
                if self._being_handle_changelane:
                    auto_follow_mode = False               
            elif self._is_wrong_way():
                logit('_is_wrong_way')
                self._being_handle_wrongway = True
                self._wrongway_handle_begin_time = time.time()
                self._handle_wrong_way(src_img)
                auto_follow_mode = False
            elif self._is_stuck(src_img):
                logit('_is_stuck')
                # time point when stuck is found 
                self._being_handle_stuck = True 
                self._stuck_handle_begin_time = time.time()
                self._handle_stuck(src_img)
                auto_follow_mode = False
            elif self._is_changelane_ahead(src_img):
                logit('_is_changelane_ahead, current speed= %f' % (speed))
                self._being_handle_changelane = True
                
                self._changelane_handle_begin_time= time.time()
                speed_to_set = round(0.6 * Settings.StraightSpeed, 1)
                self._handle_change_time = (2.0/speed_to_set)*Settings.ChangeLaneHandleTime
                kp = SpeedPIDMapping[speed_to_set][0]
                ki = SpeedPIDMapping[speed_to_set][1]
                kd = SpeedPIDMapping[speed_to_set][2]

                self._steering_pid.update_pid_factor(kp, ki, kd)
                self._throttle_pid.assign_set_point(speed_to_set)
                throttle = self._throttle_pid.update(speed)
                logit('to handle_changelane: throttle={}, changelane_direction={}'.format(throttle, self._changelane_direction))
                self._handle_changelane(src_img, throttle)
                auto_follow_mode = False
            elif self._being_sharp_turn:
                self._throttle_to_return = self._throttle_pid.update(speed)                   
                current_angle_deg =self._handle_sharp_turn(src_img, converted_last_steering_rad)
                if self._begin_sharp_turn:
                    auto_follow_mode = False
            elif self._is_sharp_turn_ahead(src_img):
                logit('_is_sharp_turn_ahead')       
                self._begin_sharp_turn = time.time()
                self._being_sharp_turn = True 
                auto_follow_mode = False
                SpeedPIDMapping = Settings.SpeedPIDMapping               
                speed_to_set = Settings.SharpTurnSpeed
                speed_to_set = round(speed_to_set, 1)
                kp = SpeedPIDMapping[speed_to_set][0]
                ki = SpeedPIDMapping[speed_to_set][1]
                kd = SpeedPIDMapping[speed_to_set][2]                 
                self._steering_pid.update_pid_factor(kp, ki, kd)
                self._throttle_pid.assign_set_point(speed_to_set)
                self._throttle_to_return = self._throttle_pid.update(speed)  
                current_angle_deg = self._handle_sharp_turn(src_img, converted_last_steering_rad)
            
            if auto_follow_mode:
                current_angle_deg = 0.0             
                speed_to_set = Settings.StraightSpeed
                WallAndLanes, Walls, blackwallimg = GreyLines.GetEdgeImages(src_img) # Get lanes and wall edges
                
                self._corppedimageX = blackwallimg.shape[1]
                self._corppedimageY = blackwallimg.shape[0]

                self._lanesangle, self._lanepoint = GreyLines.GetLanesAngle(WallAndLanes, Walls, blackwallimg, self._changelane_direction)
                self._quaters = GreyLines.GetWallAngle(Walls, blackwallimg)

                #there's a wall
                self._wallfound = False
                self._backward = False
                self._wallsharpturn = False
                if self._quaters != None and (self._quaters._turnaction != CarAction.forward or (self._lanepoint[0] == 0 and self._lanepoint[1] == 0)):
                    self._steering_angle_to_return = self.QuartWallProcess()
                else:
                    self._steering_angle_to_return = self._lanesangle
                
                current_angle_deg = self._steering_angle_to_return
                current_angle = ImageProcessor.deg2rad(current_angle_deg)
                absangble = abs(current_angle_deg)
                if absangble > 45:
                    self._uturn_handle_time_begin = time.time()
                    speed_to_set *= 0.30
                if absangble > 40:
                    self._uturn_handle_time_begin = time.time()
                    speed_to_set *= 0.40
                elif absangble > 30:
                    speed_to_set *= 0.60
                elif absangble > 20:                        
                    speed_to_set *= 0.75

                elif self._uturn_handle_time_begin is not None and time.time()-self._uturn_handle_time_begin<Settings.TurnHandleTime:
                    speed_to_set = Settings.StraightSpeed * Settings.SPEED_FACTOR_TURN              
                elif self._is_uturn_ahead(speed, current_angle_deg):
                    self._uturn_handle_time_begin = time.time()
                    logit('_is_uturn_ahead, current speed= %f' % (speed))
                    speed_to_set = Settings.StraightSpeed * Settings.SPEED_FACTOR_TURN                
                elif self._is_turn_ahead(speed, current_angle_deg):
                    self._uturn_handle_time_begin = time.time()
                    logit('_is_turn_ahead, current speed= %f' % (speed))
                    speed_to_set = Settings.StraightSpeed * Settings.SPEED_FACTOR_TURN                     
                elif TrafficSignType.may_have_traffic_sign(src_img, Settings.TRAFFIC_SIGN_PIXEL_COUNT_THRESHOLD*0.5):
                    #logit('may_have_traffic_sign, current speed= %f' % (speed))
                    speed_to_set = Settings.StraightSpeed * Settings.SPEED_FACTOR_MAY_HAVE_TRAFFIC_SIGN   
                elif self._is_straight_ahead(current_angle_deg):
                    logit('_is_straight_ahead, current speed= %f' % (speed))
                    self._straight_begin_time = time.time()
                    speed_to_set = Settings.StraightSpeed      


                anratio = self.BaseAngleAdjust()
                if self._wallfound == True:
                    if self._wallsharpturn == False:
                        anratio += 2
                    else:
                        anratio -= 1.3                    
                    if speed >= 0 and self._quaters._purewall == True:
                        anratio = 1
                    
                self._steering_angle_to_return = self._steering_angle_to_return/anratio

                
                self._previous_line_angle = current_angle                
                speed_to_set = round(speed_to_set, 1)
                self._throttle_pid.assign_set_point(speed_to_set)
                self._throttle_to_return = self._throttle_pid.update(speed)                           

            self._throttle_history.append(self._throttle_to_return)
            self._throttle_history = self._throttle_history[-30:]
            
            self._car.control(self._steering_angle_to_return, self._throttle_to_return)
            if Settings.DEBUG and self.frameCount%verbose_debug_frame_count==0:
                logit("current_angle_deg=%f, steering_angle_to_return=> %f " % (current_angle_deg, self._steering_angle_to_return))
                logit("current_speed=%f, throttle_to_return=> %f" % (speed, self._throttle_to_return))
                logit('out on_dashboard, frameCount=%d' % (self.frameCount)) 
        except Exception as exception:
            logger.error(exception)
            traceback.print_exc()


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
        quarters = self._quaters

        leftmin = self.GetMin(quarters._lefttobottom, quarters._leftmidtobottom)
        rightmin = self.GetMin(quarters._righttobottom, quarters._rightmidtobottom)
        minlen = self.GetMin(leftmin, rightmin)
                
        if self._lanepoint[0] == 0 and self._lanepoint[1] == 0:
            if quarters._turnaction == CarAction.forward:
                self.Backwards("No wall, no lanes")
            elif quarters._turntopointbottom >= Settings.eagle_wall_turn_Y and minlen > Settings.eagle_wall_distance_Y:
                self._wallsharpturn = True
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
                        toturnagle =  quarters._angletoturn
                    else:
                        toturnagle = quarters._angletoturn if abs(quarters._angletoturn) > abs(self._lanesangle) else self._lanesangle
                        self._action = "wall to right22" if toturnagle > 0 else "wall to left22"

            if minlen <= Settings.eagle_wall_turn_Y or xinterval < Settings.eagle_wall_distance_X - 10:
                self._wallsharpturn = True
                self._action = "wall to right sharp turn" if toturnagle > 0 else "wall to left sharp trun"

        return toturnagle

    def reload_setting(self):
        try:
            speed_to_set = Settings.StraightSpeed
            SpeedPIDMapping = Settings.SpeedPIDMapping
            kp = SpeedPIDMapping[speed_to_set][0]
            ki = SpeedPIDMapping[speed_to_set][1]
            kd = SpeedPIDMapping[speed_to_set][2]        
            self._steering_pid.update_pid_factor(kp, ki, kd)
            self._throttle_pid.update_pid_factor(Settings.THROTTLE_PID_Kp, Settings.THROTTLE_PID_Ki, Settings.THROTTLE_PID_Kd)
            self._throttle_pid.assign_set_point(speed_to_set)
            self._road_condition_check.update_setting(Settings.ROAD_CHECK_FRAME_RATE, 
                Settings.StraightSpeed, 
                Settings.WALL_CHECK_PIXEL_COUNT_FACTOR, 
                Settings.TRAFFIC_SIGN_PIXEL_COUNT_THRESHOLD,
                Settings.DEBUG)
        except Exception as exception:
            logger.error(exception)

    def on_road_condition(self, condition):
        if not self._being_handle_stuck:
            # wrong wall, stuck wall, stuck obstacles
            self._last_road_condition = condition
            self._last_road_condition_time = time.time()

    def on_traffic_sign(self, sign_type, sign_box_ori):
        if not self._being_handle_stuck:
            self._last_traffic_sign = sign_type
            self._last_traffic_sign_time = time.time()
            print("lap: %s, trafice sign: %s " % (self._lapCount, self._last_traffic_sign))
            logit("lap: %s, trafice sign: %s " % (self._lapCount, self._last_traffic_sign))
            if int(self._lapCount) in [1,2]:
                ImageProcessor.force_save_image_to_log_folder(sign_box_ori, prefix = "orisign", suffix=str(sign_type))
