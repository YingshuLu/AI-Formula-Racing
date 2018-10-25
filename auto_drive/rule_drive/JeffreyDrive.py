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
from GreyLinesEx2 import GreyLines
import math
import traceback
import cv2
import numpy as np
import sys
sys.path.append(r"./sign_lane")
import sign as SN
import lane
import copy

logger = logger.get_logger(__name__)
def logit(msg):
    if Settings.DEBUG:
        logger.info("%s" % msg)

class JeffreyDrive_v1(object):
    debug = Settings.DEBUG

    def __init__(self, car, road_condition_check):
        self.MAX_SPEED_HISTORY       = 18
        self.MAX_COLOR_HISTORY       = 120
        self.MAX_STEERING_HISTORY=50
        self._speed_history2 = []
        self._color_history = []
        self.dead=0
        self._back1=0
        self._back2=0
        self._back=0
        self.tt=-1
        self.last_tt=0
        self.mark_class = lane.Mark()
        self.roadcheck=0
        self._uturn=0
        self._uturn_d=0
        self.cputime=0
        self.BACKTIME=0.066
        self.server_time=0
        self.last_steering_angle=0
        self.lastlap=-1
        
        self._car = car
        self._car.register(self)
        self._roadcondition = road_condition_check
        self._roadcondition.register(self)
        self._last_road_condition_time = None
        self._last_road_condition = None        
        self._last_traffic_sign = None
        self._changelane_direction = 0
        self._throttle_to_return = 0
        self._steering_angle_to_return = 0
        self._steering_action_to_return = ""
        self._last_traffic_sign_time = None
        self._being_handle_stuck = None
        self._lanesangle =0
        self._lanepoint =0
        self._leftlanepoint = [0, 0] 
        self._leftwallangle = 0
        self._rightlanepoint = [0, 0]
        self._rightwallangle = 0
        self._backward = False
        self._wallfound = False

        self._corppedimageX = 360
        self._corppedimageY = 120
        self._last_angle = 0
        self._begin_sharp_turn = 0

        self.last_line_number = 0

        self.road_change = False
        self.last_pos = [160]*10
        self.frame_count = 0
        self._previousLapCount = 1        

        self.trace_road = 5
        self.changing_road = False
        self.backing = False
        self.backing_count = 0
        self.starting = True
        self.starting_count = 0

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

        self._lapCount = 0
        self._wrongWayLap = []
        self._hardcodeTrafficSignSequence = Settings.HARDCODE_TRAFFIC_SIGN_SEQUENCE
        self._useHardCodeTrafficSignBeforeLap = Settings.USE_HARDCODE_TRAFFIC_SEQUENCE_BEFORE_LAP
        self._trafficSignOrderNumber = 0
        self._previousLapCountWhenTrafficSignFound = 0
        self.frameCount = 0
        self._last_changelane_direction = 0
        self._previousLapCount = 1

        self.see_six_road_count = 0

        self.last_response_time = 0
        self.start_left = -30
        self.start_right = 30
        self.last_see_six_road_count = 0

        self.road_label_num =0

        self.use_white = True
        self.seeSixRoadTime = 0
        self.changingSixRoad = False
        self.pos_start_left = 0
        self.steer_history = np.zeros(30)
        

    # return None if not stuck, otherwise, return any of 'right', 'left', 'center'
    def _is_stuck(self, img):
        if self._wrongway_handle_begin_time is not None and time.time() - self._wrongway_handle_begin_time<5.0:
            return False

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
                        if next_to_wall<0.5:
                            left_wall_count += 1
                        elif next_to_wall>0.5:
                            right_wall_count += 1

                    if left_wall_count>right_wall_count and left_wall_count>0:               
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
    
    @staticmethod
    def count_color(img_color, mask_color1, mask_color2=None):
        mask1 = cv2.inRange(img_color, mask_color1[0], mask_color1[1])
        if mask_color2:
            mask2 = cv2.inRange(img_color, mask_color2[0], mask_color2[1])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = mask1
        mask_color_count = cv2.countNonZero(mask)
        return mask_color_count
        
    def _handle_stuck(self, img, speed):
        # car_pos, isNearWall = GreyLines.GetCarPos(img, self.last_pos, self.trace_road, speed,(self.frame_count+9)%10)
        # if car_pos>50 and car_pos<270:
        #     logit('stuck alreay recovered, go ahead')
        #     self._being_handle_stuck = False
        #     self._steering_angle_to_return = 0.0
        #     self._throttle_to_return = 0.1 
        #     return

        time_now = time.time()            
        if self._stuck_direction in ['left', 'right'] and time_now - self._stuck_handle_begin_time<Settings.HANDLE_STUCK_TIME:                         
            # 如果是向右撞墙（一般情况几乎和墙垂直），需要回到右后方，方向右转倒车:steering_angle设到20度，throttle设为-0.1倒车
            # 如果是向左撞墙，需要回到左后方，方向左转倒车:steering_angle设到-20度，throttle设为-0.1倒车            
            self._steering_angle_to_return = Settings.HANDLE_STUCK_ANGLE_VALUE
            if self.trace_road == 5:
                self._steering_angle_to_return =  Settings.HANDLE_STUCK_ANGLE_VALUE * 0.5
            self._throttle_to_return  = Settings.HANDLE_STUCK_THROTTLE_VALUE
            if self.trace_road == 5:
                self._throttle_to_return  = Settings.HANDLE_STUCK_THROTTLE_VALUE*1.5
            if self._stuck_direction == 'left':
                self._steering_angle_to_return = -1*Settings.HANDLE_STUCK_ANGLE_VALUE
                if self.trace_road == 5:
                    self._steering_angle_to_return = -0.5*Settings.HANDLE_STUCK_ANGLE_VALUE
        elif self._stuck_direction == 'centercar':
            #正向行驶撞小车，倒车
            if time_now - self._stuck_handle_begin_time>=Settings.HANDLE_STUCK_TIME*0.5:
                self._being_handle_stuck = False
                self._steering_angle_to_return = 0.0
                self._throttle_to_return = 0.1                         
            else:  
                self._steering_angle_to_return = 10.0
                self._throttle_to_return = -0.5
        elif self._stuck_direction == 'center':
            if time_now - self._stuck_handle_begin_time>=Settings.HANDLE_STUCK_TIME*0.5:
                self._being_handle_stuck = False
                self._steering_angle_to_return = 0.0
                self._throttle_to_return = 0.1                         
            else:  
                self._steering_angle_to_return = 0
                self._throttle_to_return = -0.5            
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
        #发现方向走错
        #return False
        if self._last_road_condition == Globals.WrongWay and (time.time() - self._last_road_condition_time) < 1.0:
            left_wall_count = 0
            right_wall_count = 0
            for next_to_wall in self._next_to_wall_history:                
                if next_to_wall<0.50:
                    left_wall_count += 1
                elif next_to_wall>0.50:
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
            self._wrongWayLap.append(self._lapCount)
            return True
        else:
            self._previousLapCount = self._lapCount                
            return False

    def _handle_wrong_way(self, img, speed):
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
                if self._wrongway_direction in ['right', 'center']:
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
    
    def my_uturn(self,src_img,speed) :
      # Mark detection
             
        start_time = time.time()
        self.server_time=start_time- self.last_response_time 
        mark = self.mark_class
        direct = mark.predict(src_img)
        
        if direct == 1 :
          self.roadcheck=self.roadcheck+1
        
        
        track_img = cv2.GaussianBlur(src_img, (5, 5), 0)
        
        # wall angle start
        left_wall_distance, right_wall_distance, left_wall_count, right_wall_count = self.wall_detector(track_img)
        
        self._color_history.append((left_wall_distance,right_wall_distance))
        self._color_history = self._color_history[-self.MAX_COLOR_HISTORY:]
        if len(self._color_history)<2 :  self._color_history.append((left_wall_distance,right_wall_distance))
        self._speed_history2.append(speed)
        self._speed_history2 = self._speed_history2[-self.MAX_SPEED_HISTORY:]
        
        track_count = 6
        
        self.tt=self.tt+1
        if self.tt > 15*60 :
            self.tt = 0
            self.roadcheck = 0
            #print("cputime=",self.cputime/(15*60),float(dashboard['time']))
            #print("cputime=",self.cputime/(15*60),float(dashboard['time']),file=self.f)
            
            self.cputime = 0
         
        if self._uturn>0 or self.roadcheck>10 :
            #print("wrong road u-turn",self.laptime,self.laptime-self.lastlap_time,self.distance,file=self.f)
            self.roadcheck=0
            if self._uturn==0 :
            	self._uturn=10
            	
            	for (cc,cc1) in reversed(self._color_history):
            	  if cc>cc1 :	
                         self._uturn_d=0
                         
                         break
            	  elif cc1>cc:  
                         self._uturn_d=1
                         break
            if track_count==3 :
              self._uturn=self._uturn-1
            self._uturn=self._uturn-1
            if self._uturn_d==0 : 
             #print("wrong road u-turn,left",self.laptime,self.laptime-self.lastlap_time,self.distance)  
             #print("wrong road u-turn,left",self.laptime,self.laptime-self.lastlap_time,self.distance,file=self.f)
             self._back2=self.MAX_SPEED_HISTORY/2+4
             x=self.BACKTIME-(time.time()-start_time)-self.server_time      
             if x>0:
                    print(x)
                    time.sleep(x) 
             self._car.control(-40, 1)
            else : 
             #print("wrong road u-turn,right",self.laptime,self.laptime-self.lastlap_time,self.distance)  
             #print("wrong road u-turn,right",self.laptime,self.laptime-self.lastlap_time,self.distance,file=self.f)
             self._back1=self.MAX_SPEED_HISTORY/2+4
             x=self.BACKTIME-(time.time()-start_time)-self.server_time      
             if x>0:
                    print(x)
                    time.sleep(x)  
             self._car.control(40, 1)
            
            self.steering_angle=0   
            self.cputime=self.cputime+time.time()-start_time  
            return 0
        
        
        
        
        
        if (sum(self._speed_history2[-3:-1])/3<0.04 and  len(self._speed_history2)==self.MAX_SPEED_HISTORY) \
            or self._back>0 or self._back1>0 or self._back2>0  :
        
           #print("go back",self.laptime,self._back,self._back1,self._back2)
           
           if self._back1>0 :
                print("left",self._back1)
                self._back1=self._back1-1
                if track_count==3 and self._back1>12  :
                 self._back1=self._back1-1
                if self._back>0 :
                        self._back=self._back-1 
                self.cputime=self.cputime+time.time()-start_time  
                x=self.BACKTIME-(time.time()-start_time)-self.server_time  
                #print(x)    
                if x>0:
                    print(x)
                    time.sleep(x)        
                if self._back1>6 :self._car.control(-40, -0.4)
                else:  self._car.control(40, 1)
                
                
                return 0
           
           if self._back2>0 :
                print("right",self._back2)
                self._back2=self._back2-1
                if track_count==3 and self._back2>12  :
                 self._back2=self._back2-1
                if self._back>0 :
                  self._back=self._back-1 
                self.cputime=self.cputime+time.time()-start_time
                x=self.BACKTIME-(time.time()-start_time)-self.server_time      
                if x>0:
                    print(x)
                    time.sleep(x)  
                if self._back2>6 :self._car.control(40, -0.4)
                else:  self._car.control(-40, 1)
                
                
                return 0
           
           if self._back>0 :
            
            self._back=self._back-1 
            self.cputime=self.cputime+time.time()-start_time     
            x=self.BACKTIME-(time.time()-start_time)      
            if x>0:
                    print(x)
                    time.sleep(x)  
            self._car.control(0, -1)
            return 0
           if True:
            
            for (cc,cc1) in reversed(self._color_history):
             #print(cc,cc1)
             if (cc>cc1):
                
                 
                
                 
                
                if self._back1==0 : 
                 
                  self._back1=self.MAX_SPEED_HISTORY/2+4
                self._car.control(-40, -1)
                
                self.steering_angle=0
                
                self.cputime=self.cputime+time.time()-start_time
                #print("left",self.laptime,self.distance,self._back1,self.tt,self.last_tt)
                #print("left",self.laptime,self.distance,self._back1,self.tt,self.last_tt,file=self.f)
                self.last_tt=self.tt
                return 0
                
             if cc1>cc :
                
                
                
                 
                if self._back2==0 : self._back2=self.MAX_SPEED_HISTORY/2+4
                self._car.control(40, -1)
                
                self.steering_angle=0
                
                self.cputime=self.cputime+time.time()-start_time
                #print("right",self.laptime,self.distance,self._back2,self.tt,self.last_tt)
                #print("right",self.laptime,self.distance,self._back2,self.tt,self.last_tt,file=self.f)
                self.last_tt=self.tt
                return 0
           
                
               
           if self._back==0 : 
            if self.dead==1 :
             if (random.randint(10, 20)>=18) :
               self._back2=self.MAX_SPEED_HISTORY/2+4
             elif (random.randint(10, 20)<=13) :  
              self._back1=self.MAX_SPEED_HISTORY/2+4
             else:
              self._back=int(self.MAX_SPEED_HISTORY)+random.randint(10, 20)
            else:
             self._back=int(self.MAX_SPEED_HISTORY)
             #print("center",self.laptime,self.distance,self._back,self.tt,self.last_tt)
             #print("center",self.laptime,self.distance,self._back,self.tt,self.last_tt,file=self.f)
           self.cputime=self.cputime+time.time()-start_time     
           x=self.BACKTIME-(time.time()-start_time)-self.server_time      
           if x>0:
                    print(x)
                    time.sleep(x)  
           self._car.control(0, -1)
           return 0
        return 1   
    
    def wall_detector(self,img, ratio = 0.5):
        img_height = img.shape[0]
        img_width = img.shape[1]
        half_width = int(img_width * ratio)
        half_height = int(img_height * 0.2)
        img_left = img[half_height:img_height, 0:half_width].copy()
        img_right = img[half_height:img_height, half_width:img_width].copy()

        black_wall = [(0, 0, 0), (10, 10, 10)]
        yellow_wall = [(0, 160, 160), (0, 180, 180)]
        max_color_count = half_height * half_width * 4
        left_color_count = float(JeffreyDrive_v1.count_color(img_left, black_wall, yellow_wall))
        right_color_count = float(JeffreyDrive_v1.count_color(img_right, black_wall, yellow_wall))
        left_wall_distance = left_color_count / max_color_count
        right_wall_distance = right_color_count / max_color_count

        return left_wall_distance, right_wall_distance, left_color_count, right_color_count  

    def on_dashboard(self, src_img, converted_last_steering_rad, last_steering_angle, speed, throttle, info):
        try:
            print("Simulator time ",time.time()- self.last_response_time)
            
            lap= int(info["lap"]) if "lap" in info else 0
            if self.lastlap!= lap and lap>0:
             self.roadcheck=0
             self.lastlap=lap
            print (lap,self.lastlap)  
            src_img2=copy.copy(src_img)
            if self.my_uturn(src_img2,speed)==0 : 
                self.last_response_time = time.time()
                return  
             
            if self.frameCount == 0:
                logit('race started')
            if self.frameCount%35==0:
                print('speed=%f' % (speed))
            self.frameCount += 1
            self._lapCount = int(info['lap'])
            self._roadcondition.add_latest_frame(src_img)
            if speed>0.8:
                next_to_wall = TrafficSignType.check_wall_direction(src_img, 0.1)
                if next_to_wall is not None:
                    self._next_to_wall_history.append(next_to_wall)
                    self._next_to_wall_history = self._next_to_wall_history[-120:]
            self._speed_history.append(speed)
            self._speed_history = self._speed_history[-60:]
            
            if True:            
                # Road color is[Red, Blue,  Red, Green, Blue, Green]
                # trace_road is[ 0  1  2  3  4  5  6   7  8  9  10 ]
                #print time.time(), 'autodrive'

                # isWrongDirection = GreyLines.CheckWrongDirectionByColor(src_img, self.last_pos)
                # print "isWrongDirection",isWrongDirection
                start_time = time.time()
                # WallAndLanes, Walls, blackwallimg = GreyLines.GetEdgeImages(src_img) # Get lanes and wall edges
                
                # self._corppedimageX = blackwallimg.shape[1]
                # self._corppedimageY = blackwallimg.shape[0]

                # self._lanesangle, self._lanepoint = GreyLines.GetLanesAngle(WallAndLanes, Walls, blackwallimg, self._changelane_direction)   
                

                self.HandleTrafficeSign()
                    #print "In process of turn %s" % self._changelane_direction
                print("In process of turn %s" % self._changelane_direction) 
                if self._changelane_direction == -1:
                    self.trace_road = 2
                    #self.changing_road = True
                elif self._changelane_direction == 1:
                    self.trace_road = 8
                    #self.changing_road = True
                elif self._changelane_direction == 0:
                    # isSeeSixRoad = GreyLines.SeeSixRoad(src_img)
                    if self.last_see_six_road_count > 4:
                        self.trace_road = 5
                    # if isSeeSixRoad:
                    #     self.see_six_road_count += 1
                    #     #self.changing_road = False
                    #     #self.trace_road = 5
                    # if self.see_six_road_count > 10:
                    #     self.see_six_road_count = 0
                    #     self.trace_road = 5
                #logit('self.trace_road=%s' % (self.trace_road))
                #self.trace_road = 5
                #print "self.trace_road",self.trace_road
                right_count = np.where(self.steer_history >= 40)
                left_count = np.where(self.steer_history <= -30)
                print("self.steer_history",self.steer_history)
                if len(right_count[0]) > 13 and len(left_count[0])>13:
                    self.trace_road = 2
                self.AdjustAngleAndThrottle_v2(speed, src_img, self.trace_road)  
                end_time = time.time()
                print("end_time",end_time,"Cost time",end_time - start_time)

            self._throttle_history.append(self._throttle_to_return)
            self._throttle_history = self._throttle_history[-30:]
            self._car.control(self._steering_angle_to_return, self._throttle_to_return) 

            self.last_response_time = time.time()
               
        except Exception as exception:
            logger.error(exception)
            traceback.print_exc()

        #前方障碍物或fork road，变道
    def HandleTrafficeSign(self):
        if self._last_traffic_sign_time is None:
            return False

        time_now = time.time()
        if (self._last_traffic_sign==Globals.ForkLeftSign or self._last_traffic_sign==Globals.RightObstacle) and (time_now-self._last_traffic_sign_time) < 3:
            #turn left
            self._changelane_direction = -1
            return True
        elif (self._last_traffic_sign==Globals.ForkRightSign or self._last_traffic_sign==Globals.LeftObstacle) and (time_now-self._last_traffic_sign_time) < 3:
            #turn right
            self._changelane_direction = 1
            return True
        else:
            self._changelane_direction = 0
            return False  

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
            P_steer = 0.1
            D_steer = 0.1
            speed_set = 2.5
        elif trace_road == 2 or trace_road == 8:
            P_steer = 0.1
            D_steer = 0.15
            speed_set = 2.5

        P_speed = 1
        D_speed = 0


        
        
        ##print("self.last_pos",self.last_pos)
        if (self._changelane_direction == -1 or self._changelane_direction == 1) and self._last_changelane_direction == 0:
            self.changing_road = True
        # if trace_road == 2 and self._changelane_direction == 1:
        #     self.changing_road = True
        start_time = time.time()
        car_pos, isNearWall, isSeeSixRoad, label_number = GreyLines.GetCarPos(src_img, self.last_pos, trace_road, speed,(self.frame_count+9)%10,self.changing_road,self._changelane_direction)
        
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
            P_steer = 0.1
            D_steer = 0.1

        if trace_road == 2 and self.changing_road and self._changelane_direction == 1 and label_number == 1:
            label_number = 2

        
        if self.changing_road and label_number != 1:#(car_pos < 40 or car_pos > 280):      
            #print "self._lanepoint[0]",self._lanepoint[0]
            #print "Changing road....." 
            if self._changelane_direction == -1:
                car_pos = 100
            elif self._changelane_direction == 1:
                car_pos = 220
            else:
                car_pos = 160
            #D_steer = 0
        else:
            self.changing_road = False
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
        self.frame_count+=1
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
        self.steer_history[self.frame_count%30] = self._steering_angle_to_return
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
        print("lap: %d, trafice sign: %s " % (self._lapCount, self._last_traffic_sign))
        logit("lap: %d, trafice sign: %s %f" % (self._lapCount, self._last_traffic_sign, self._last_traffic_sign_time))
        # if self._lapCount in [1,2] and sign_box_ori is not None:
        #     ImageProcessor.force_save_image_to_log_folder(sign_box_ori, prefix = "orisign", suffix=str(sign_type))

    def on_road_condition(self, condition):
        if not self._being_handle_stuck:
            # wrong wall, stuck wall, stuck obstacles
            self._last_road_condition = condition
            self._last_road_condition_time = time.time()
