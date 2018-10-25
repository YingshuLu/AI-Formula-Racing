# coding=utf-8
#!env python
#
# Auto-driving Bot
#
# Revision:      v1.2
# Released Date: Aug 20, 2018
#
import time
from ImageProcessor import ImageProcessor
import threading
import os
import logger
from PID import PID
import Globals
import Settings
from Car import Car
from RoadConditionCheck import RoadConditionCheck
from AutoDrive import AutoDrive, AutoReloadSetting
from CrazyDrive import CrazyDrive
from JeffreyDrive import JeffreyDrive_v1

logger = logger.get_logger(__name__)


def logit(msg):
    logger.info("%s" % msg)

       
if __name__ == "__main__":
    import shutil
    import argparse
    from datetime import datetime

    import socketio
    import eventlet
    import eventlet.wsgi
    from flask import Flask

    #print Settings.MAX_STEERING_ANGLE
    parser = argparse.ArgumentParser(description='AutoDriveBot')
    parser.add_argument(
        'record',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder to record the images.'
    )
    args = parser.parse_args()

    if args.record:
        if not os.path.exists(args.record):
            os.makedirs(args.record)
        logit("Start recording images to %s..." % args.record)
        Globals.RecordFolder = args.record

    sio = socketio.Server()

    def send_control(steering_angle, throttle):
        sio.emit(
            "steer",
            data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
            },
            skip_sid=True)
    Globals.Running = True

    car = Car(control_function = send_control)

    roadConditionCheck = RoadConditionCheck(Settings.ROAD_CHECK_FRAME_RATE, Settings.StraightSpeed, Settings.DEBUG)
    drive = None
    if Settings.RUN_MODE==0:
        drive = AutoDrive(car, roadConditionCheck)
    elif Settings.RUN_MODE==1:
        drive = CrazyDrive(car, roadConditionCheck)
    else:
        drive = JeffreyDrive_v1(car, roadConditionCheck)
        
    autoReloadSettingThread = AutoReloadSetting(drive)

    @sio.on('telemetry')
    def telemetry(sid, dashboard):
        if dashboard:
            car.on_dashboard(dashboard)
        else:
            sio.emit('manual', data={}, skip_sid=True)

    @sio.on('connect')
    def connect(sid, environ):
        car.control(0, 0)

    app = socketio.Middleware(sio, Flask(__name__))
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    Globals.Running = False
    autoReloadSettingThread.join()
    roadConditionCheck.join()
    print('AI Bot terminated')

# vim: set sw=4 ts=4 et :

