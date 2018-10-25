import numpy as np

from modules.image_processor import ImageProcessor
from time import time,sleep
import random
import sys
sys.path.append(r"./sign_lane")
import sign as SN
import lane
import math
import cv2
import bird
from collections import Counter

# escape obstacle
OBSTACLE_RATIO = 2

CROSS_ANGLE = False

# fork turn
SIGN_FRAMES = 6
SIGN_RATIO = 5

# angle switch
LANES_FRAMES = 6

# mid line cut
LINE_RATIO = 0.545

# escape fork obstacle count
FORK_OBSTACLE = 10

# wrong way check
WRONG_MARK_CNT = 10

#hit wall delay
HIT_WALL_DELAY = 10

my_debug = False

def im_show(name, img):
    if my_debug:
        cv2.imshow(name, img)


class AutoDrive(object):

    def __init__(self, car, debug=False):
        self.MAX_SPEED_HISTORY       = 18
        self.MAX_COLOR_HISTORY       = 120
        self.MAX_STEERING_HISTORY=50
        self.MAX_STEERING=6.5
        self.BACKTIME=0.066
        self._car = car
        self._car.register(self)
        self.debug = debug
        self.half_pi = np.pi / 2
        self._speed_history = []
        self._color_history = []
        self.dead=0
        self._back1=0
        self._back2=0
        self._back=0
        self.tt=-1
        self.last_tt=0
        self.obstacle=0
        self.cputime=0
        self.lastlap=-1
        self.lastlap_time=-300
        self.lastlap_time6=-300
        self.bestlap_time=300
        self.bestlap_time6=300
        self.lap_gap=0
        self.last_speed=0
        self.laptime=0
        self.last_laptime=0
        self.distance=0
        self.hitwall=0
        self.total_hitwall=0
        self.steering_history=[]
        self.last_steering_angle = 0
        self.class_id=-1
        self.throttle=0
        self.steering_angle=0
        self._uturn=0
        self._uturn_d=0
        self.roadcheck=0
        self.roadcheck0=0
        
        # vars
        self.wall_counts = []
        self.sign_class = SN.Sign()
        self.sign_history = []
        self.last_lanes = -1
        self.lanes_history = []
        self.lanes_frames = LANES_FRAMES
        self.on_obstacle = True
        self.mark_class = lane.Mark()
        self.lane_id = -1
        self.on_which = -1
        self.last_lane_id = -1
        self.hit_wall_delay = 0
        self.f = open("./log/mylog.txt", "w")         
        self.fork_cnt = 0
        self.fork_id = None
        self.last_fork_id = None
        self.follow_fork = 0
        self.red_conts = None
        self.fork_run = 0
        self.last_sign_time = None
        self.start_locate = False
        self.last_hit_wall = False

    # switch to red lane, and follow it (direct = 0)
    def _on_red_lane(self, lanes):
        lane_thresh = 9000
        wtrend = self.wall_trend(lanes)
        red_conts = self.red_lanes(lanes)

        if red_conts is None or len(red_conts) == 0:
            return 0

        num = len(red_conts)
        if num == 1:
            return wtrend
        else:
            #if red_conts[0].type() == lanes.RED_LANE and red_conts[0].area() > lane_thresh:
            if red_conts[0].area() > lane_thresh:
                return -1
            #if red_conts[-1].type() == lanes.RED_LANE and red_conts[-1].area() > lane_thresh:
            if red_conts[-1].area() > lane_thresh:
                return 1
        return 0

    def on_red_lane(self, lanes, direct):
        lane_thresh = 9000
        wtrend = self.wall_trend(lanes)
        red_conts = self.red_lanes(lanes)

        if red_conts is None or len(red_conts) == 0:
            return False

        num = len(red_conts)
        if num == 1:
            idx = 0 if direct == -1 else 1
            wc = self.wall_counts[idx]
            if wc > 9000:
                return True
            return False
        else:
            #if red_conts[0].type() == lanes.RED_LANE and red_conts[0].area() > lane_thresh:
            if red_conts[0].area() > lane_thresh:
                return -1 == direct
            #if red_conts[-1].type() == lanes.RED_LANE and red_conts[-1].area() > lane_thresh:
            if red_conts[-1].area() > lane_thresh:
                return 1 == direct
        return False


    def red_lanes(self, lanes):
        if self.red_conts is not None:
            return self.red_conts

        red_mask = lanes.red_mask()
        h,w = red_mask.shape[:2]
        red_conts = []
        binary, contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

        areas = []
        for i in range(len(contours)):
            areas.append(cv2.contourArea(contours[i])) 

        if len(areas) == 0:
            return None
        
        areas = np.array(areas)
        idx = np.argsort(-areas)

        for i in idx:
            cont = contours[i]
            cont = lane.Contour(lanes, cont)
            cmask = cont.mask()
            if cont.area() < 150:
                continue 

            wmask = lanes.mask_add(cmask, lanes.white_mask())
            if np.sum(wmask > 0) > cont.area()/3:
                continue

            locs = np.where(cmask > 0)
            loc_y = np.sum(locs[0])/len(locs[0])
            if loc_y  < h*4/5:
                red_conts.append(cont)

        if len(red_conts) == 0:
            return None

        if len(red_conts) > 3:
            red_conts = red_conts[0:3]

       # sort contour with min x with bird view
        if len(red_conts) > 1:
            red_conts.sort(key=lambda x : x.min_location()[0])
            #red_conts = sorted(sorted(red_conts, key = lambda x:x.min_location()[0]), key = lambda y:y.min_location()[1], reverse = True)

        num = len(red_conts)
        if num == 3:
            if red_conts[0].type() != lanes.RED_LANE:
                red_conts[0].ctype = lanes.RED_LANE
            if red_conts[2].type() != lanes.RED_LANE:
                red_conts[2].ctype = lanes.RED_LANE
            if red_conts[1].type() != lanes.RED_LINE:
                red_conts[1].ctype = lanes.RED_LINE
        elif num == 2:
            if red_conts[0].type() == lanes.RED_LINE and red_conts[1].type() == lanes.RED_LINE:
                if red_conts[0].bird_area() > red_conts[1].bird_area():
                    red_conts[0].ctype = lanes.RED_LANE
                else:
                    red_conts[1].ctype = lanes.RED_LANE
            elif red_conts[0].type() == lanes.RED_LANE and red_conts[1].type() == lanes.RED_LANE:
                if red_conts[0].bird_area() < red_conts[1].bird_area():
                    red_conts[0].ctype = lanes.RED_LINE
                else:
                    red_conts[1].ctype = lanes.RED_LINE
        elif num == 1:
            red_conts[0].ctype = lanes.RED_LANE

        self.red_conts = red_conts

        i = 0
        for cont in red_conts:
            im_show("red cont " + str(i), cont.mask())
            i += 1

        return self.red_conts

    def wall_count(self, lanes):
        wmask = lanes.wall_mask()
        h,w = wmask.shape[:2]
        left_mask = wmask[:, 0:int(w/2)]
        right_mask = wmask[:, int(w/2):w]

        left_count = np.sum(left_mask > 0)
        right_count = np.sum(right_mask > 0)
        return left_count, right_count

    def entrance_fork_lane(self, lanes, direct):
        red_conts = self.red_lanes(lanes)
        if red_conts is None:
            return True

        if len(red_conts) != 1:
            return False

        left, right = self.wall_counts[0], self.wall_counts[1] 
        diff = left - right

        val = direct * diff 
        if val > 0:
            return True
        return False

    def wall_trend(self, lanes):
        left_count, right_count = self.wall_counts[0], self.wall_counts[1] 
        return right_count - left_count

    def red_lane_angle(self, lanes, direct = 0):
        red_conts = self.red_lanes(lanes)
        if red_conts is None:
            return None
        num = len(red_conts)
        if num == 0:
            return None
        
        wtrend = self.wall_trend(lanes)
        fix_angle = 5 * direct
        target_cont = None
        bird_max_cont = None
        if bird_max_cont is None:
            area = 0
            for cont in red_conts:
                if cont.area() > area:
                    area = cont.area()
                    bird_max_cont = cont

        #decide red lane
        if direct == 0:
            target_cont = bird_max_cont
       
        elif self.on_which == 0:
            #if direct * wtrend > 0:
            if self.on_red_lane(lanes, direct):
                idx = 0 if direct == -1 else -1
                target_cont = red_conts[idx]
            else:
                return fix_angle * 2
        else:
            if direct == 1:
                if num == 3:
                    target_cont = red_conts[-1]
                elif num == 2:
                    if red_conts[1].type() == lanes.RED_LANE:
                        target_cont = red_conts[1]
                    elif red_conts[0].type() == lanes.RED_LANE:
                        return fix_angle
                else:
                    if direct * wtrend > 0:
                        target_cont = bird_max_cont
                    else:
                        return fix_angle

            if direct == -1:
                if num == 3:
                    target_cont = red_conts[0]
                elif num == 2:
                    if red_conts[0].type() == lanes.RED_LANE:
                        target_cont = red_conts[0]
                    elif red_conts[1].type() == lanes.RED_LANE:
                        return fix_angle
                else: 
                    if direct * wtrend > 0:
                        target_cont = bird_max_cont
                    else:
                        return fix_angle
                    
            angle = None
            if target_cont is not None:
                mask = target_cont.mask()
                angle = self.red_angle(mask)

                #print("target cont area",target_cont.area())
                #img = lanes.crop_img()
                #cv2.drawContours(img, [target_cont.cont], -1, (255,0,0), 3)
                #im_show("target lane", img)
                    
                if angle is None:
                    return fix_angle
            return angle

    def red_angle(self, mask):
        red_mask = mask
        red_mask = bird.view(red_mask)
        h,w = red_mask.shape[:2]
        line = red_mask[ int(LINE_RATIO * h), :]
        locs = np.where(line > 0)
        if locs is None or len(locs[0].tolist()) < 10:
            return None

        x = sum(locs[0].tolist()) / len(locs[0].tolist())
        angle =  (x - 160) * 0.1
        return angle

    def max_red_lane_angle(self, lanes):
        red_conts = self.red_lanes(lanes)
        if red_conts is None:
            return None
        target_cont = None
        area = 0
        for cont in red_conts:
            carea = cont.area()
            if carea > area:
                area = carea
                target_cont = cont
            
        if target_cont is None:
            return None

        #img = lanes.crop_img().copy()
        #cv2.drawContours(img, [target_cont.cont], -1, (0,255,0), 3)
        #im_show("follow max cont", img)
        #bird_img =  bird.view(img)
        #im_show("bird follow max cont", bird_img)
        return self.red_angle(target_cont.mask())

    def on_main_lanes(self):
        if len(self.lanes_history) < self.lanes_frames:
            return 1
        history = np.array(self.lanes_history)
        num = len(self.lanes_history)
        first_half = self.lanes_history[int(num/2):num]
        second_half = self.lanes_history[0:int(num/2)]
        main_sum = np.sum(np.array(first_half) == 2)
        fork_sum = np.sum(np.array(second_half) == 1)
        main = main_sum > self.lanes_frames / 4 or main_sum > fork_sum
        if main:
            return 1
        fork_sum = np.sum(np.array(first_half) == 1)
        main_sum = np.sum(np.array(second_half) == 2)
        fork =  fork_sum > main_sum
        if fork:
            return 0
        return -1

            
    def on_lane_id(self, lanes):
        self.lane_id = lanes.locate()
        if self.lane_id == -1 and self.last_lane_id != -1:
            if self.last_steering_angle > 0:
                if self.last_lane_id > -1:
                    if self.last_lane_id == 5:
                        self.lane_id = 5
                    elif self.last_lane_id == 2.5:
                        self.lane_id = 3
                    else:
                        self.lane_id = self.last_lane_id + 1
            elif self.last_steering_angle < 0:
                if self.last_lane_id > -1:
                    if self.last_lane_id == 0:
                        self.lane_id = 0
                    elif self.last_lane_id == 2.5:
                        self.lane_id = 2
                    else:
                        self.lane_id = self.last_lane_id - 1
        self.last_lane_id = self.lane_id
        return self.lane_id


    def hit_wall_reset(self):
        self.fork_id = self.last_fork_id
        self.hit_wall_delay = HIT_WALL_DELAY
        #self.lanes_history = []
        self.on_which = -1
        self.roadcheck=0
        self.roadcheck0=0
        self.last_hit_wall = True
        print("# hit wall")
                

    def on_dashboard(self, src_img,dashboard):
        start_time = time()
        self.laptime=float(dashboard["time"])
        speed               = float(dashboard["speed"])
        lap= int(dashboard["lap"]) if "lap" in dashboard else 0
        self.distance=self.distance+(speed+self.last_speed)/2.*(self.laptime-self.last_laptime)
        self.last_speed=speed
        self.last_laptime=self.laptime
        if self.lastlap!= lap and lap>0:
         if lap>2 : self.lap_gap=self.lap_gap+abs(float(dashboard['time'])-self.lastlap_time-self.bestlap_time  )
         
         if float(dashboard['time'])-self.lastlap_time < self.bestlap_time :
           self.bestlap_time=float(dashboard['time'])-self.lastlap_time
         self.total_hitwall= self.total_hitwall+ self.hitwall
         print("lap= ",lap," %3.3f"%(float(dashboard['time'])-self.lastlap_time),"best lap=%3.3f"%self.bestlap_time,"D=%4.3f"%self.distance,"avg gap=%3.3f"%(self.lap_gap/max(lap-2.,1.)),"TH=", self.total_hitwall,"sp=",self.MAX_STEERING)
         
         
         self.lastlap_time=float(dashboard['time'])
         self.lastlap=lap
         self.lastlap_time6=float(dashboard['time'])
         self.distance=0
         self.hitwall=0
         
        # reset state
        if self.hit_wall_delay > 0:
            self.hit_wall_delay -= 1

        self.red_conts = None

        #preprocess
        track_img = ImageProcessor.preprocess(src_img)
        lanes = lane.Lanes(src_img)

        #blanes = bird.view(lanes.crop_img(),1)
        #im_show("bird lanes", blanes)

        #self.red_lanes(lanes)
        on_which = lanes.on_main_lanes()
        if len(self.lanes_history) > 30:
            self.lanes_history = self.lanes_history[1:-1]
        #if not self.start_locate:
        #    on_which = 1
        self.lanes_history.append(on_which + 1)

        # lanes info
        self.lane_id = self.on_lane_id(lanes)
        self.on_which  = self.on_main_lanes()
        if self.on_which == -1:
            self.on_which = self.last_lanes
        self.last_lanes = self.on_which
        if self.on_which == 1:
            self.start_locate = False

        if self.on_which == 0:
            self.fork_run += 1
        else:
            self.fork_run = 0

        class_id = self.sign_class.predict(src_img)
        obs = None
        if class_id == 0 or class_id == 1:
            obs = lanes.detect_obstacle()
        else:
            obs = lanes.obstacle_mask()

        obs = lanes.detect_obstacle(True)

        #print("# on which", self.on_which)
        #print("# lane id", self.lane_id)
        #print("# self.fork id", self.fork_id)
        #print("wall counts", self.wall_counts)

        # wall angle start
        left_wall_distance, right_wall_distance, left_wall_count, right_wall_count = self.wall_detector(track_img)
        self.wall_counts = [left_wall_count, right_wall_count]

        # escape obstacle
        if not obs is None:
            obs_ratio = OBSTACLE_RATIO
            # fork obstacle
            if self.on_which == 0:
                obs_ratio /= 2
            obs_mask = obs
            h, w = obs_mask.shape[:2]
            left_mask = obs_mask[:, 0:int(w/2)]
            right_mask = obs_mask[:, int(w/2) : w]

            left_mask_count = np.sum(left_mask > 0) * obs_ratio
            right_mask_count = np.sum(right_mask > 0) *obs_ratio

            left_wall_count += left_mask_count
            right_wall_count += right_mask_count
            base = h * 0.2 *w

            left_wall_distance = left_wall_count/base
            right_wall_distance = right_wall_count/base 

        # fork turn
        sign_frames = SIGN_FRAMES
        if len(self.sign_history) >= sign_frames:

           string='''
           num = len(self.sign_history)
           class0 = np.sum(np.array(self.sign_history) == 1)
           class1 = np.sum(np.array(self.sign_history) == 2)

           cid = -1
           if class1 > class0 and class1 > num/4:
              cid = 1
           if class0 > class1 and class0 > num/4:
              cid = 0

           '''

           num = len(self.sign_history)
           before = np.array(self.sign_history[0:int(num/2)])
           after = np.array(self.sign_history)[int(num/2):num]
           turn = np.sum(after) == 0

           fork = False

           class1 = np.sum(before == 2)
           class0 = np.sum(before == 1)

           cid = -1
           if class1 > class0 and class1 > num/4:
              cid = 1
           if class0 > class1 and class0 > num/4:
              cid = 0

           if cid != -1:
                fork = True
           fork = turn and fork

           repeat = False
           if self.last_sign_time is not None:
               if start_time - self.last_sign_time < 3.000:
                    repeat = True

           if fork and not repeat and self.fork_id is None and self.follow_fork == 0 and cid != -1:
                self.fork_id = 1 if cid == 0 else -1
                self.last_sign_time = start_time
                self.start_locate = True

           self.sign_history = self.sign_history[1:-1]
                
        if len(self.sign_history) < sign_frames:
            if class_id >= 2:
                class_id = -1
            self.sign_history.append(class_id +1)
    
        # fork lane obstacle escape
        loc = SN.detect_obstacle(src_img)
        if True:
            if loc != [-1, -1]:
                self.on_obstacle = True
                self.fork_id = None
            else:
                self.on_obstacle = False

        # angle switch
        switch = False

        # cross angle
        if CROSS_ANGLE and not self.on_obstacle and self.fork_id is not None and self.on_which == 1:
            cross = lanes.cross_angle()
            angle = cross[0]
            if angle is not None:
                switch = True
                print("# cross angle")
                self.throttle = 1
                self.steering_angle = angle
            else:
                switch = False

        # mid red angle

       # print("# hit wall delay", self.hit_wall_delay)
#        print("# on which", self.on_which)

        need_follow = self.follow_fork > 0 or self.fork_id is not None
        need_mid = self.hit_wall_delay == 0 and not self.on_obstacle and self.on_which == 0

        #print("# need follow", need_follow)
        #print("# need mid", need_mid)
        if switch:
            pass
        #elif self.follow_fork > 0 or self.fork_id is not None or self.hit_wall_delay == 0 and not self.on_obstacle and (self.on_which + self.last_lanes) < 2:
        elif need_follow or need_mid:
           angle = None
           if self.fork_id is not None:
                #if self.on_red_lane(lanes) == self.fork_id or self.on_which + self.last_lanes < 2:

                entrance = self.entrance_fork_lane(lanes, self.fork_id)
                if entrance:
                    pass
                    #print("# fork lane entrance")
                if self.on_red_lane(lanes, self.fork_id):
                    #print("# strick to red lane")
                    self.last_fork_id = self.fork_id
                    self.fork_id = None
                    self.follow_fork = 20
                    if entrance:
                        self.last_fork_id = None
                        self.follow_fork = 0
                    # now need follow max contour
                else:
                    #print("# switch to red lane")
                    angle = self.red_lane_angle(lanes, self.fork_id)
                    self.follow_fork = 0

           if self.last_fork_id is not None or self.fork_id is not None:
               direct = self.fork_id if self.fork_id is not None else self.last_fork_id
               if self.entrance_fork_lane(lanes, direct):
                   self.last_fork_id = None
                   self.fork_id = None
                   self.follow_fork = 0

           if self.fork_run > 8:
                pass
#                print("predict on fork")
           #     self.last_fork_id = None
                #add
           #     self.follow_fork = 0
           #     self.fork_id = None


           # on target lane and follow it
           if self.fork_id is None:
                angle = self.max_red_lane_angle(lanes)
                #print("# follow red lane")

           if angle is not None:
                direct = 1 if angle > 0 else -1
                if math.fabs(angle) >=15.9:
                    angle = None

           if angle is not None:
                self.throttle = 1
                self.steering_angle = angle
                switch = True
           else:
                switch = False

        if self.follow_fork > 0:
            self.follow_fork -=1
                
        # mid wall angle
        if not switch:
            self.steering_angle = self.find_wall_angle(left_wall_distance, right_wall_distance)
            self.throttle = 1

            if left_wall_count < 2450 and right_wall_count < 2450:
                self.steering_angle = self.steering_angle / 5
            if left_wall_count > 3400 and right_wall_count > 3400:
                self.steering_angle = self.steering_angle * 6
            if right_wall_distance == left_wall_distance == 1:
                self.steering_angle = self.last_steering_angle
            else:
                self.last_steering_angle= self.steering_angle
#            print("# wall angle")

        if my_debug:
            im_show("source", src_img)
            cv2.waitKey(1)
            self._car.control(0,0)
            return

        # mark detection

        if self.last_hit_wall:
            mark = self.mark_class
            direct = mark.predict(src_img)
            if direct == 1 :
                self.roadcheck=self.roadcheck+1
            elif direct == 0:
                self.roadcheck0 -= 1

        if self.roadcheck0 < -5:
            self.roadcheck0 = 0
            self.last_hit_wall = False

        ############# Hit wall recover ##########
        if self.debug:
            ImageProcessor.show_image(track_img, "track")
        
        track_count = 6
        self.tt=self.tt+1
        if self.tt>15*60 : 
            self.tt=0
            self.roadcheck=0
            self.cputime=0
        
        if self._uturn>0 or self.roadcheck>5 :
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
             print("wrong road u-turn,left")
             self._back2=self.MAX_SPEED_HISTORY/2+4
             x=self.BACKTIME-(time()-start_time)      
             if x>0:
                    sleep(x) 
             self._car.control(-40, 1)
            else : 
             print("wrong road u-turn,right")
             self._back1=self.MAX_SPEED_HISTORY/2+4
             x=self.BACKTIME-(time()-start_time)      
             if x>0:
                    sleep(x)  
             self._car.control(40, 1)
            
            self.steering_angle=0   
            self.cputime=self.cputime+time()-start_time  
            return
        
        
        self._color_history.append((left_wall_distance,right_wall_distance))
        self._color_history = self._color_history[-self.MAX_COLOR_HISTORY:]
        if len(self._color_history)<2 :  self._color_history.append((left_wall_distance,right_wall_distance))
        self._speed_history.append(speed)
        self._speed_history = self._speed_history[-self.MAX_SPEED_HISTORY:]
        
        
        if sum(self._speed_history[-3:-1])/3>0.04 : self.dead=0
        if (sum(self._speed_history[-3:-1])/3<0.04 and (self.dead==0  or  self.obstacle>0)  and len(self._speed_history)==self.MAX_SPEED_HISTORY) \
            or self._back>0 or self._back1>0 or self._back2>0  :
           self.hitwall=self.hitwall+1
           if self.hitwall > 0:
               self.hit_wall_reset()

           if self._back1>0 :
                self._back1=self._back1-1
                if track_count==3 and self._back1>6  :
                 self._back1=self._back1-1
                if self._back>0 :
                        self._back=self._back-1 
                self.cputime=self.cputime+time()-start_time  
                x=self.BACKTIME-(time()-start_time)  
                if x>0:
                    sleep(x)        
                if self._back1>6 :self._car.control(-40, -0.4)
                else:  self._car.control(40, 1)
                return
           
           if self._back2>0 :
                self._back2=self._back2-1
                if track_count==3 and self._back2>6  :
                 self._back2=self._back2-1
                if self._back>0 :
                  self._back=self._back-1 
                self.cputime=self.cputime+time()-start_time
                x=self.BACKTIME-(time()-start_time)      
                if x>0:
                    sleep(x)  
                if self._back2>6 :self._car.control(40, -0.4)
                else:  self._car.control(-40, 1)
                return
           
           if self._back>0 :
            self._back=self._back-1 
            self.cputime=self.cputime+time()-start_time     
            x=self.BACKTIME-(time()-start_time)      
            if x>0:
                sleep(x)  
            self._car.control(0, -1)
            return
           if self.obstacle>=0:
            for (cc,cc1) in reversed(self._color_history):
             if (cc>cc1):
                if self._back1==0 : 
                  self._back1=self.MAX_SPEED_HISTORY/2+4
                self._car.control(-40, -1)
                self.steering_angle=0
                self.cputime=self.cputime+time()-start_time
                self.last_tt=self.tt
                return
                
             if cc1>cc :
                if self._back2==0 : self._back2=self.MAX_SPEED_HISTORY/2+4
                self._car.control(40, -1)
                self.steering_angle=0
                self.cputime=self.cputime+time()-start_time
                self.last_tt=self.tt
                return
               
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
           self.cputime=self.cputime+time()-start_time     
           x=self.BACKTIME-(time()-start_time)      
           if x>0:
               sleep(x)  
           self._car.control(0, -1)
           return
           ##########################        

        if self.on_which == 0:
            abs_angle = math.fabs(self.steering_angle)
            if abs_angle > 45:
                abs_angle = 45
            beta = 1/(1+ math.exp(-(45 - abs_angle))/3)
            self.throttle *= beta
        if not switch and self.on_which == 1:
            direct = 1 if self.steering_angle > 0 else -1
            if direct > 0 and self.steering_angle > 22:
                self.steering_angle = direct * (direct * self.steering_angle + 10)
            
            
        #print(self.steering_angle, self.throttle)
        self.last_steering_angle = self.steering_angle
        self._car.control(self.steering_angle, self.throttle)
        return
            
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
        left_color_count = float(ImageProcessor.count_color(img_left, black_wall, yellow_wall))
        right_color_count = float(ImageProcessor.count_color(img_right, black_wall, yellow_wall))
        left_wall_distance = left_color_count / max_color_count
        right_wall_distance = right_color_count / max_color_count

        return left_wall_distance, right_wall_distance, left_color_count, right_color_count

    
    def find_wall_angle(self,left_wall_distance, right_wall_distance):
        #global steering_history
        count_rate = (right_wall_distance / left_wall_distance) if left_wall_distance > 0 else 0
        count_rate = max(min(count_rate * 40 , self.MAX_STEERING), 0)
        
        steering_angle = -count_rate if right_wall_distance > left_wall_distance else count_rate
        '''
        self.steering_history.append(steering_angle)
        self.steering_history = self.steering_history[-self.MAX_STEERING_HISTORY:]
        if abs(sum(self.steering_history))>self.MAX_STEERING_HISTORY*self.MAX_STEERING-0.01 :
         steering_angle=-0.7 if right_wall_distance > left_wall_distance else 0.7
         self.steering_history[0]=0
        '''
        #print(steering_angle,abs(sum(self.steering_history)))
        return steering_angle
