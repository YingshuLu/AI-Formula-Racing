import numpy as np
from PIL  import Image
from io   import BytesIO
import base64
import copy
from ImageProcessor import ImageProcessor
class Car(object):
    def __init__(self, control_function):
        self._driver = None
        self._control_function = control_function

    def register(self, driver):
        self._driver = driver

    def on_dashboard(self, dashboard):
        #normalize the units of all parameters
        converted_last_steering_angle = np.pi/2 - float(dashboard["steering_angle"]) / 180.0 * np.pi
        last_steering_angle = float(dashboard["steering_angle"])
        throttle            = float(dashboard["throttle"])
        speed               = float(dashboard["speed"])
        img                 = ImageProcessor.bgr2rgb(np.asarray(Image.open(BytesIO(base64.b64decode(dashboard["image"])))))

        total_time = dashboard["time"].split(":") if "time" in dashboard else []
        seconds    = float(total_time.pop()) if len(total_time) > 0 else 0.0
        minutes    = int(total_time.pop())   if len(total_time) > 0 else 0
        hours      = int(total_time.pop())   if len(total_time) > 0 else 0
        elapsed    = ((hours * 60) + minutes) * 60 + seconds

        info = copy.copy(dashboard)
        info['elapsed'] = elapsed
        del info['image']
        self._driver.on_dashboard(img, converted_last_steering_angle, last_steering_angle, speed, throttle, info)


    def control(self, steering_angle, throttle):
        self._control_function(steering_angle, throttle)
    
     

