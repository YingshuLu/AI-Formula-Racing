from modules.auto_drive import AutoDrive
from modules.car import Car

if __name__ == "__main__":
    import argparse
    import socketio
    import eventlet
    import eventlet.wsgi
    from flask import Flask

    parser = argparse.ArgumentParser(description='AutoDriveBot')
    parser.add_argument("-d", "--debug", help="open track view", action='store_true', required=False)

    args = parser.parse_args()
    debug = args.debug
    sio = socketio.Server()


    def send_control(steering_angle, throttle):
        if debug:
            sio.emit(
                "steer",
                data={
                    'steering_angle': str(0),
                    'throttle': str(0)
                },
                skip_sid=True)
        else:
            sio.emit(
                "steer",
                data={
                    'steering_angle': str(steering_angle),
                    'throttle': str(throttle)
                },
                skip_sid=True)


    def send_restart():
        sio.emit(
            "restart",
            data={},
            skip_sid=True)


    car = Car(control_function=send_control, restart_function=send_restart)
    drive = AutoDrive(car, debug)
    test_result = []
    last_lab = 1


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
