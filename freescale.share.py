from __future__ import print_function

from time import time
from PIL  import Image
from io   import BytesIO

import os
import cv2
import math
import numpy as np
import base64
import logging
import collections
from imageprocess import ImageProcessor
from imageprocess import is_blue, is_green, is_red, is_black
import colorsys
from logging.config import fileConfig
import logging
fileConfig('logging_config.ini')
logger = logging.getLogger('free')
from datetime import datetime
import sys
import random
sys.path.append(r"./release")
import sign as SN
#sys.path.append(r"./lane")
import lane

def logit(msg):
    print("%s" % msg)

def get_dominant_color(image):
    max_score = 0.0001
    dominant_color = None
    for count,(r,g,b) in image.getcolors(image.size[0]*image.size[1]):
        
        saturation = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)[1]
        y = min(abs(r*2104+g*4130+b*802+4096+131072)>>13,235)
        y = (y-16.0)/(235-16)
 
        
        if y > 0.9:
            continue
        score = (saturation+0.1)*count
        if score > max_score:
            max_score = score
            dominant_color = (r,g,b)
    return dominant_color

class Car(object):
    MAX_STEERING_ANGLE = 40.0
    MAX_SPEED_HISTORY       = 20
    MAX_COLOR_HISTORY       = 30
    DEFAULT_SPEED_RATE          = 1
    
    def __init__(self, control_function):
        self._control_function = control_function
        self.servo_err_history = collections.deque([0], 30)
        self.motor_err_history = collections.deque([0,0,0,0,0,0], 90)
        self.speed_sum = 0
        self.last_time = 0
        self.steering_angle = 0
        self.throttle = 1
        self.car_status = 'INITIAL'
        self.track = 5
        self.LOW_SPEED = 1.85
        self.last_throttle = 0
        self.lane_id = None
        self._speed_history = []
        self._color_history = []
        
        self._back1=0
        self._back2=0
        self._back=0
        self.tt=-1
        self.last_tt=0
        self._uturn=0
        self._uturn_d=0
        self.test=0
        self._turnright=0
        self._speedrate=self.DEFAULT_SPEED_RATE
        self.roadcheck=0
        self.lastlap=-1
        self.lastlap_time=0
        self.target_track=5
        for i in range(0, 30):
            self.servo_err_history.appendleft(0)

        self.init_cnt = 15*100

    def on_dashboard(self, dashboard,test2):
        #normalize the units of all parameters
        throttle            = float(dashboard["throttle"])
        speed               = float(dashboard["speed"])
        oriimg                 = np.asarray(Image.open(BytesIO(base64.b64decode(dashboard["image"]))))
        img2            = Image.open(BytesIO(base64.b64decode(dashboard["image"])))
        del dashboard["image"]
        
        nocropimg = ImageProcessor.preprocess(oriimg)
        if self.car_status == 'INITIAL':
            self.track = ImageProcessor.which_color_and_track(nocropimg)
            self.car_status = 'RUNNING'
            self.track = list(self.track)
            self.track[8] = 5
        
        if test2==1 :print((datetime.now(), dashboard));
        
        lap= int(dashboard["lap"]) if "lap" in dashboard else 0
        if self.lastlap!= lap :
         
         
         print("lap= ",lap,float(dashboard['time'])-self.lastlap_time)
         self.lastlap_time=float(dashboard['time'])
         self.lastlap=lap
         if self.track[8] == 2 :
               self.target_track  = 5
               #self.control(40, 0.04) 
               #return
        
        if self.target_track!=self.track[8] :
          if self.track[8]>self.target_track :
           self.track[8]=self.track[8]-1
          else:
           self.track[8]=self.track[8]+1 
         
        #this_time = dashboard['time']
        #if this_time == self.last_time:
            #print(this_time)
            #self.control(self.steering_angle, self.throttle)
            #return
        #self.last_time = this_time
        
        
        # preprocess of raw image
        
        # apply filter
        filterimg = ImageProcessor.image_filter(nocropimg)
        # perspective transformation
        img = self.perspective(filterimg)
        #ImageProcessor.show_image(img, "nocropimg")
        #ImageProcessor.save_image('log', img)
        #
        # car state machine
        # 
        

        track_count = self.how_many_tracks(filterimg)

        # normal
        mid, lost1 = self.determine_middle(img, row_default=20, color=self.track)
        mid2, lost2 = self.determine_middle(img, row_default=5, color=self.track)
        #if track_count==3 :
        # mid2, lost2 = self.determine_middle(img, row_default=3, color=2)
        # calculate set speed
        set_speed = self.cal_speed(lost2)
        set_speed = 2 if set_speed>2 else set_speed
        #set_speed = 1
        #cv2.line(img,(0,40),(320,40),(0,0,255),2)
        #ImageProcessor.save_image('log', img)
        if lost1 == False :
            self.steering_angle = self.servo_control(mid, mid2, lost1, lost2)
            self.throttle = self.motor_control(mid, speed, set_default=set_speed)
        else:
           if lost2 == False :
             self.steering_angle = self.servo_control(mid2, mid2, lost1, lost2)
             self.throttle = self.motor_control(mid2, speed, set_default=set_speed)
           else:
            # if lost line, slow down and keep original direction
            self.throttle = self.motor_control(mid, speed, set_default=1)
            if self.steering_angle > 0:
                self.steering_angle += 0.5
            if self.steering_angle < -0:
                self.steering_angle -= 0.5
            #ImageProcessor.save_image('log', oriimg)
        #logger.info(str(mid), str(mid2), str(lost1), str(lost2), str(self.steering_angle), str(self.car_status), str(set_speed), str(track_count))
        if self.init_cnt > 0:
            self.init_cnt -= 1
            logger.info('%s %s %s %s %s', mid, mid2, lost1, lost2, self.steering_angle)
        
        mcolor=get_dominant_color(img2.crop((0, 100, 320,200)).convert('RGB'))
        color_left=get_dominant_color( (img2.crop((0, 180, 159,200))).convert('RGB'))
        color_right=get_dominant_color( (img2.crop((160, 180, 320,200))).convert('RGB'))   
        if test2==1 :
         ImageProcessor.show_image(oriimg, "img")
         #ImageProcessor.show_image(np.asarray(img2.crop((0, 180, 159,200)).convert('RGB') ), "img_l")
         #ImageProcessor.show_image(np.asarray(img2.crop((160, 180, 320,200)).convert('RGB') ), "img_r")
        
        self.class_id = -1 
        if self.test==0 :
         sn = SN.Sign()
        
         if SN.blank(ImageProcessor.bgr2rgb(oriimg))==1 :
           self.roadcheck= self.roadcheck+20
         self.class_id = sn.predict(ImageProcessor.bgr2rgb(oriimg))
        
        self.lane_id = lane.locate(ImageProcessor.bgr2rgb(oriimg))+1
        on_which=lane.on_main_lanes(ImageProcessor.bgr2rgb(oriimg))
        
        if track_count==3  :
         if self.lane_id is not None and self.lane_id==5 :
          self.throttle =0.04
         else :
          self.throttle =0.024
        #if track_count==6 and on_which!=-1 and  self.test==0 and self.tt<15 :
        #  self.track[8] = 2
        #  self.test=30
        
        '''
        if   self.test==0 : 
         if color_right[0]>200 and self.track[8] == 5 :
               self.track[8] = 2
               print("on  3 track 5 to 2",track_count)
         elif color_right[1]>171 and self.track[8] ==2 :
               self.track[8] = 5 
               
               print("on  3 track 2 to 5",track_count)
        '''
       
               
        if test2==1 and self.test==0 :
        
          if self.class_id==0 and self.track[8] == 5: 
            self.class_id=1
            
            if self.test==0:self.test=30
            
         
        
        
        
        if self.test>0 :
         self.test=self.test-1
        
        
        self.tt=self.tt+1
        if self.tt>15*15 : 
            self.tt=0
            self.roadcheck=0
        
        
        
        if test2==1 :print("lane is ",self.lane_id,self.roadcheck,track_count,mid, lost1,mid2,lost2,self.class_id,test2, self.track[8],on_which,mcolor,color_left,color_right)   
        if self.lane_id is not None and self.lane_id<=3 and self.class_id==0  :
               self.target_track  = 5
               print("change to road 5")
               if self.test==0:self.test=30
               self.control(40, 0.04) 
               return
        
        if self.lane_id is not None and self.lane_id>=4 and (self.class_id==1 ):
               self.target_track  = 2
               print("change to road 2")
               if self.test==0:self.test=30
               self.control(-40, 0.04) 
               return    
        
        if  ((self.class_id==4 or self.class_id==5 )  and track_count==3) \
        or (self.test>5 or self.class_id==1 ):
               print("Slow Slow")
               self.throttle = self.motor_control(mid2, speed, set_default=0.1)
        else :
         if abs(ImageProcessor.rad2deg(self.steering_angle))<8 :  
              self.throttle=self.throttle*4       
        
        self._color_history.append((mcolor,color_left,color_right))
        self._color_history = self._color_history[-self.MAX_COLOR_HISTORY:]
        if len(self._color_history)<2 :  self._color_history.append((mcolor,color_left,color_right))
        self._speed_history.append(speed)
        self._speed_history = self._speed_history[-self.MAX_SPEED_HISTORY:]
        
        if ((sum(self._speed_history)/len(self._speed_history)<0.04) and (mcolor==(0,0,0) or mcolor==(51,51,51)) and len(self._speed_history)==self.MAX_SPEED_HISTORY)  \
        or ((sum(self._speed_history)/len(self._speed_history)<0.04) and (sum(self._speed_history)/len(self._speed_history)>0) \
        and color_left!=color_right and len(self._speed_history)==self.MAX_SPEED_HISTORY)  \
        or self._back>0 or self._back1>0 or self._back2>0 \
        or ((sum(self._speed_history)/len(self._speed_history)<0.04) and mcolor==(171,170,0) and color_left==(171,170,0) and color_right==(171,170,0)):
           if test2==1 : print(("go back",speed,sum(self._speed_history)/len(self._speed_history),self._back,self._back1,self._back2,mcolor,color_left,color_right))
           
           if self._back1>0 :
                print("red")
                self._back1=self._back1-1
                if track_count==3 and self._back1>0  :
                 self._back1=self._back1-1
                if self._back>0 :
                        self._back=self._back-1 
                if self._back1>6 :self.control(-40, -0.04)
                else:  self.control(-40, -0.04)
                
                return
           
           if self._back2>0 :
                print("green")
                self._back2=self._back2-1
                if track_count==3 and self._back2>0  :
                 self._back2=self._back2-1
                if self._back>0 :
                  self._back=self._back-1 
                if self._back2>6 :self.control(40, -0.04)
                else:  self.control(40, -0.04)
                return
            
           for (cc,cc1,cc2) in reversed(self._color_history):
            
            if (cc2[0]>200 and cc[0]!=171 and cc[1]!=170)  or (cc[0]==171 and cc[1]==170 and cc2[1]>171 ) or  mcolor==(51,51,51)  :
                self._speedrate=max(self._speedrate*0.9,0.4)
                if (cc1[0]>200 and cc[0]!=171 and cc[1]!=170)  :
                 print("red",self.tt,self.last_tt)
                 
                else:
                 print("yellow left",self.tt,self.last_tt)
                 
                
                if self._back1==0 : 
                 if mcolor==(51,51,51) :
                  self._back1=int(self.MAX_SPEED_HISTORY)+random.randint(10, 60)
                 else:
                  self._back1=self.MAX_SPEED_HISTORY/2+4
                self.control(-40, -1)
                self.roadcheck=0
                if abs(self.tt-self.last_tt)<60 :
                 if self.track[8] == 5 :
                   self.track[8] = 2
                 self._back1=self._back1+random.randint(1, 10)
                for i in range(0, 30):
                 self.servo_err_history.appendleft(0)
                self.last_tt=self.tt
                return
                
            if cc2[1]>171 or (cc[0]==171 and cc[1]==170 and cc2[0]>200 ) :
                self._speedrate=max(self._speedrate*0.9,0.4)
                if cc1[1]>171  :
                 print("green",self.tt,self.last_tt)
                 
                else:
                 print("yellow right",self.tt,self.last_tt)
                 
                if self._back2==0 : self._back2=self.MAX_SPEED_HISTORY/2+4
                self.control(40, -1)
                self.roadcheck=0
                if abs(self.tt-self.last_tt)<60 :
                 if self.track[8] == 2 :
                   self.track[8] = 5
                 self._back2=self._back1+random.randint(1, 10)
                for i in range(0, 30):
                 self.servo_err_history.appendleft(0)
                self.last_tt=self.tt
                return
                
           if self._back==0 : 
            if mcolor==(51,51,51) :
             self._back=int(self.MAX_SPEED_HISTORY)+random.randint(10, 20)
            else:
             self._back=int(self.MAX_SPEED_HISTORY)
           if self._back>0 :
            self._back=self._back-1     
           self.control(0, -1)
           self.roadcheck=0
           
           return  
        
        mcolor_4=get_dominant_color( (img2.crop((0, 60, 20,100))).convert('RGB'))
        mcolor_6=get_dominant_color( (img2.crop((320-20, 60, 320,100))).convert('RGB'))
        if test2==1 : 
          ImageProcessor.show_image(np.asarray(img2.crop((0, 60, 20,100)).convert('RGB')), "4")
          ImageProcessor.show_image(np.asarray(img2.crop((320-20, 60, 320,100)).convert('RGB')), "6")
        if mcolor_4==(0,0,0) and self.lane_id is not None and self.lane_id==5 \
        or mcolor_6==(0,0,0) and self.lane_id is not None and self.lane_id==2  \
        or mcolor_4==(171,170,0) and self.lane_id is not None and self.lane_id==2 \
        or mcolor_6==(171,170,0) and self.lane_id is not None and self.lane_id==5   \
        :
         if self._back==0 and self._back1==0 and self._back2==0 and self._uturn==0 : 
          self.roadcheck= self.roadcheck+1
          if self.roadcheck>100 :
           print(self.roadcheck)
        
        
           
        if self.roadcheck>120 or self._uturn>0 :
            
            if self._uturn==0 :
            	self._uturn=10
            	self.roadcheck=0
            	if self.lane_id >=3 : self._uturn_d=1
            if track_count==3 :
              self._uturn=self._uturn-1
            self._uturn=self._uturn-1
            if self._uturn_d==0 : 
             print("wrong road u-turn,left")
             self.control(-45, 0.2)
            else : 
             print("wrong road u-turn,right")
             self.control(45, 0.2)
            for i in range(0, 30):
                 self.servo_err_history.appendleft(0)
            return
        
          
        #if     self.throttle<0 : self.throttle=0  
        self.control(self.steering_angle, self.throttle)

    def control(self, steering_angle, throttle):
        self._control_function(steering_angle, throttle)

    def servo_control(self, mid1, mid2, lost1, lost2):
        err = mid1 - 160
        self.servo_err_history.appendleft(err)
        delta_err = err - self.servo_err_history[5]
        Kp = 0.1
        Kd = 0.0
        angle = Kp * err + Kd * delta_err
        #print(err, angle, delta_err, Kp * err, Kd * delta_err)
        # add filter, because steer is too sensitive
        return int(angle*3)/3.0

    def cal_speed(self, lost2):
        # in straight line a long time
        cnt = 0
        for err in self.servo_err_history:
            if abs(err) > 6:
                break
            cnt += 1
        #print(cnt)
        if cnt < 2:
            return self.LOW_SPEED
        if cnt < 4:
            return self.LOW_SPEED + 0.05
        if cnt < 6:
            return self.LOW_SPEED + (0 if lost2 else 0.15)
        if cnt < 8:
            return self.LOW_SPEED + (0 if lost2 else 0.25)
        else:
            return self.LOW_SPEED + (0 if lost2 else 0.3)

    def motor_control(self, middle, speed, set_default=0.5):
        Kp = 0.5
        Ki = 0.03

        set_speed = set_default
        err = set_speed - speed
        delta_err = err - self.motor_err_history[0]
        self.motor_err_history.appendleft(err)
        #self.speed_sum = sum(self.motor_err_history)

        throttle = self.last_throttle + delta_err * Kp + Ki * err
        if throttle < -0.4:
            throttle = -0.4
        if throttle > 1:
            throttle = 1
        self.last_throttle = throttle
        #print (err, self.speed_sum, err * Kp, Ki * self.speed_sum)
        return throttle

    # color is not used in this version
    def find_track(self, img, irow, color=2):
        left = -1
        right = 320

        track=self.track
        which_track = track[8]
        color = track[which_track]
        l_color = track[which_track-1]
        r_color = track[which_track+1]
        func_mapping = {
            0 : is_red,
            1 : is_green,
            2 : is_blue
        }
        is_color = func_mapping[color]
        is_l_color = func_mapping[l_color]
        is_r_color = func_mapping[r_color]

        # count from left to right
        # in right-center or left blue
        if which_track == 4 or which_track == 2:
            for i in range(0, 320):
                if is_color(img[irow, i]):
                    left = i
                    for j in range(i if i+10> 320 else i+10, 320):
                        if not is_color(img[irow, j]):
                            right = j
                            break
                    break
        # count from right to left
        # in left-center or right blue
        if which_track == 3 or which_track == 5:
            for i in range(319, -1, -1):
                if is_color(img[irow, i]):
                    right = i
                    for j in range(i if i-10 < 0 else i-10, -1, -1):
                        if not is_color(img[irow, j]):
                            left = j
                            break
                    break

        l_valid = False
        r_valid = False
        for i in range(left-1, -1 , -1):
            if is_black(img[irow, i]):
                continue
            if is_l_color(img[irow, i]):
                # should not too far
                if abs(left - i) < 30:
                    l_valid = True
            #else:
                break
        for i in range(right+1, 320):
            if is_black(img[irow, i]):
                continue
            if is_r_color(img[irow, i]):
                # should not too far
                if abs(right - i) < 30:
                    r_valid = True
                #print (1, i-right)
            #else:
                break
        if l_valid or r_valid:
            return left, right
        else:
            return -1, 320

        
    def perspective(self, img):
        cropimg = img[130:240, :]

        src = np.array([
            [0, 0],
            [320 , 0],
            [0, 110 ],
            [320 , 110]
            ], dtype = "float32")

        dst = np.array([
            [0, 0],
            [320, 0],
            [320 * 0.35, 110],
            [320 * 0.65, 110]], dtype = "float32") 

        M = cv2.getPerspectiveTransform(src, dst)
        wrapped = cv2.warpPerspective(cropimg, M, (320, 110))
        return  wrapped 

    def determine_middle(self, img, row_default=18, color=2):
        #ImageProcessor.show_image(img, "source")

        left, right = self.find_track(img, row_default, color=color)

        if right < 100:
            middle = 50
        elif left > 220:
            middle = 270
        else:
            middle = (right+left)/2
        
        lost = False
        if right == 320 or left == -1:
            lost = True
        return middle, lost

    # how many tracks can we see in the camera
    def how_many_tracks(self, img):
        which_track = self.track[8]

        func_mapping = {
            0 : is_red,
            1 : is_green,
            2 : is_blue
        }

        if which_track == 4 or which_track == 3:
            return 6
        if which_track == 2:
            # in left blue, check right border if has track 4 color 
            is_color = func_mapping[self.track[4]]
            for i in range(239, 70, -1):
                # means we can see track 4's color
                if is_color(img[i, 290]):
                    return 6
            return 3
        if which_track == 5:
            # in right blue, check left border if has track 3 color
            is_color = func_mapping[self.track[3]]
            for i in range(239, 70, -1):
                if is_color(img[i, 30]):
                    return 6
            return 3
        
        return 3

if __name__ == "__main__":
    import shutil
    import argparse
    from datetime import datetime

    import socketio
    import eventlet
    import eventlet.wsgi
    from flask import Flask
    test=0
    
    parser = argparse.ArgumentParser(description='AutoDriveBot')
    parser.add_argument(
        'test',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder to record the images.'
    )
    args = parser.parse_args()
    
    if args.test:
        test=1
        
    sio = socketio.Server()
    def send_control(steering_angle, throttle):
        sio.emit(
            "steer",
            data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
            },
            skip_sid=True)

    car = Car(control_function = send_control)

    @sio.on('telemetry')
    def telemetry(sid, dashboard):
        if dashboard:
            car.on_dashboard(dashboard,test)
        else:
            sio.emit('manual', data={}, skip_sid=True)

    @sio.on('connect')
    def connect(sid, environ):
        car.control(0, 0)

    app = socketio.Middleware(sio, Flask(__name__))
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)