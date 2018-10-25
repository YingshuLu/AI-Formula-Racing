# coding=utf-8
import logger
import Globals
import threading
import Settings
from time import time
from ImageProcessor import ImageProcessor
import TrafficSignType

logger = logger.get_logger(__name__)


class RoadConditionCheck(threading.Thread):
    # Auto reload Settings thread
    def __init__(self, checkFrameRate, straightSpeed, debug = False):
        super(RoadConditionCheck, self).__init__()
        self.event = threading.Event()
        self.event.clear()
        self.eventWait = 1.0 / checkFrameRate
        self.signFoundSkipSecs = (2.0/straightSpeed)*2
        self.roadconditionCheckSkipSecs = 3.0
        self.nextFrame = None
        self.frameAddTime = time()
        self.signDetectBeginTime = None
        self.mayHaveTrafficSignTime = None
        self.trafficSignFoundTime = time()
        self.wrongwayFoundTime = time()
        self.trafficSignDetectedBuff = []
        self.wrongwayDetectBeginTime = None
        self.wrongwayDetectedBuff = []
        self.trafficSignOriImgDic = {}
        self.signDetectContinueTime = 0.1
        self.roadConditionFoundTime = time()
        self.obstacleFoundTime = time()         
        self.driver = None
        self.debug = debug
        self.start()
        
    
    def logit(self, msg):
        if self.debug:
            #print(msg)
            logger.info("%s" % msg)

    def register(self, driver):
        self._driver = driver
    
    def add_latest_frame(self, image):
        self.nextFrame = image
        self.event.set()
        
    def update_setting(self, checkFrameRate, straightSpeed, wallFactor, trafficSignPixelCountThreshold, debug = False):
        self.eventWait = 1.0 / checkFrameRate
        self.signFoundSkipSecs = 3.0/straightSpeed
        self.signDetectContinueTime = 0.15/straightSpeed
        self.roadconditionCheckSkipSecs = 3.0
        self.wallFactor = wallFactor
        self.trafficSignPixelCountThreshold = trafficSignPixelCountThreshold
        self.debug = debug

    def detect_wrong_way(self, image):
        time_now = time()
        if not Settings.IgnoreTrafficSign:
            if self.wrongwayDetectBeginTime is None:
                iswrongway, wrong_way_img = TrafficSignType.detect_wrongway(image)
                if iswrongway:                                                   
                    self.logit('seems iswrongway found')
                    #ImageProcessor.force_save_image_to_log_folder(sign_box_ori, prefix = "orisign", suffix=str(sign_type))
                    self.wrongwayDetectBeginTime = time_now
                    self.wrongwayDetectedBuff.append(iswrongway)
                                          
            elif self.wrongwayDetectBeginTime is not None and time_now-self.wrongwayDetectBeginTime<self.signDetectContinueTime*3:
                iswrongway, wrong_way_img = TrafficSignType.detect_wrongway(image)
                if iswrongway:                                                   
                    self.logit('seems iswrongway found')
                    #ImageProcessor.force_save_image_to_log_folder(sign_box_ori, prefix = "orisign", suffix=str(sign_type))
                    self.wrongwayDetectedBuff.append(iswrongway)

            elif self.wrongwayDetectBeginTime is not None and len(self.wrongwayDetectedBuff)>1:
                self.logit('iswrongway found')                  
                self._driver.on_road_condition(Globals.WrongWay)
                self.wrongwayFoundTime = time_now
                self.wrongwayDetectBeginTime = None
                self.wrongwayDetectedBuff = []
            else:
                self.wrongwayDetectBeginTime = None
                self.wrongwayDetectedBuff = []
                        
        

    def procee_image(self, image):        
        time_now = time()
        # if (time_now - self.roadConditionFoundTime)>=self.roadconditionCheckSkipSecs: 
        #     condition = TrafficSignType.check_wall_obstacle(image, self.wallFactor)
        #     if condition is not None:
        #         self.roadConditionFoundTime = time_now
        #         self._driver.on_road_condition(condition)
        #         #ImageProcessor.save_image(image, prefix = "orig", suffix=str(condition))
        #         #self.logit('after check_wall_obstacle, condition={}'.format(condition))
        #         # when stuck, skip the traffic sign check
        #         if condition==Globals.StuckBlackWall or condition==Globals.StuckObstacle:
        #             return

        # check obstacle   
        if (time_now - self.obstacleFoundTime)>=0.15:
            obstacle_on_side = TrafficSignType.detect_obstacle(image, Settings.OBSTACLE_PIXEL_COUNT_THRESHOLD)
            if obstacle_on_side is not None:
                self.obstacleFoundTime = time_now
                if obstacle_on_side == -1:
                    self._driver.on_road_condition(Globals.LeftObstacle)
                elif obstacle_on_side == 1:
                    self._driver.on_road_condition(Globals.RightObstacle)
                else:
                    self._driver.on_road_condition(Globals.RightObstacle)
                print('find obstacle, obstacle_on_side='+str(obstacle_on_side))

            
        if not Settings.IgnoreTrafficSign:
            # for traffic sign, we will try to do five times of traffic detection
            if self.signDetectBeginTime is None and (time_now - self.trafficSignFoundTime)>=self.signFoundSkipSecs:
                sign_type, sign_box_ori = TrafficSignType.detect_traffic_sign_type(image, self.trafficSignPixelCountThreshold)
                if sign_type in range(1,9) and sign_box_ori is not None:                                                   
                    self.logit('sign_type=%d' % (sign_type))
                    #ImageProcessor.force_save_image_to_log_folder(sign_box_ori, prefix = "orisign", suffix=str(sign_type))
                    self.signDetectBeginTime = time_now
                    self.trafficSignDetectedBuff.append(sign_type)
                    self.trafficSignOriImgDic[sign_type] = sign_box_ori
                        
            elif self.signDetectBeginTime is not None and time_now-self.signDetectBeginTime<self.signDetectContinueTime:
                sign_type, sign_box_ori = TrafficSignType.detect_traffic_sign_type(image, self.trafficSignPixelCountThreshold)
                if sign_type in range(1,9) and sign_box_ori is not None:             
                    self.logit('sign_type=%d' % (sign_type))
                    #ImageProcessor.force_save_image_to_log_folder(sign_box_ori, prefix = "orisign", suffix=str(sign_type))
                    self.trafficSignDetectedBuff.append(sign_type)
                    self.trafficSignOriImgDic[sign_type] = sign_box_ori

            elif self.signDetectBeginTime is not None and len(self.trafficSignDetectedBuff)>0:
                # get the sign type that has the most number of detection, which would be notified to the driver
                sign_number_map={}
                for sign_type in self.trafficSignDetectedBuff:
                    if sign_type in sign_number_map:
                        sign_number_map[sign_type] +=1
                    else:
                        sign_number_map[sign_type] = 1
                max_number_sign = 0
                max_number_sign_type = None
                for sign_type in sign_number_map:
                    sign_number = sign_number_map[sign_type]
                    if sign_number>max_number_sign:
                        max_number_sign = sign_number
                        max_number_sign_type = sign_type
                #print (max_number_sign, max_number_sign_type)
                sign_box_ori = self.trafficSignOriImgDic[max_number_sign_type]
                self._driver.on_traffic_sign(max_number_sign_type, sign_box_ori)
                self.trafficSignFoundTime = time_now
                self.signDetectBeginTime = None
                self.mayHaveTrafficSignTime = None
                self.trafficSignDetectedBuff = []
                self.trafficSignOriImgDic = {}
            
                

    def run(self):
        while Globals.Running:            
            self.event.wait(self.eventWait)
            if self.event.isSet():
                if self.nextFrame is not None:
                    self.event.clear()
                    image = self.nextFrame
                    self.nextFrame = None
                    self.procee_image(image)
                    self.detect_wrong_way(image)
