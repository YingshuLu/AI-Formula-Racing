import base64
from io import BytesIO

import numpy as np
from PIL import Image

from modules.image_processor import ImageProcessor


class Car(object):
    MAX_STEERING_ANGLE = 40.0

    def __init__(self, control_function, restart_function):
        self._driver = None
        self._control_function = control_function
        self._restart_function = restart_function

    def register(self, driver):
        self._driver = driver

    def on_dashboard(self, dashboard):
        img = ImageProcessor.bgr2rgb(np.asarray(Image.open(BytesIO(base64.b64decode(dashboard["image"])))))
        del dashboard["image"]
        #print(dashboard)
        self._driver.on_dashboard(img,dashboard)

    def control(self, steering_angle, throttle):
        # convert the values with proper units
        # steering_angle = min(max(ImageProcessor.rad2deg(steering_angle), -Car.MAX_STEERING_ANGLE),
        #                      Car.MAX_STEERING_ANGLE)
        self._control_function(steering_angle, throttle)
