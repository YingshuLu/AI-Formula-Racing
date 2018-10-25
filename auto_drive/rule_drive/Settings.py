# coding=utf-8
# 0 for AutoDrive, 1 for CrazyDrive, otherwise for JeffreyDrive
RUN_MODE = 2

# For crazy drive, we will try to use the hard traffic sign sequence for better race speed.
# we can get the traffic sign sequence in each lap from the practice race, and stored here for the following real race on Wednesday and Thursday.
HARDCODE_TRAFFIC_SIGN_SEQUENCE=[4,4,4,4,4,4,3,4,4,3,4,4,3]

# before the lap number, use hard code traffice sign sequence.
# set it to number bigger than 10 to not use the hard code traffic since the car race 10 laps at most
USE_HARDCODE_TRAFFIC_SEQUENCE_BEFORE_LAP=0

# expected speed in good road condition
StraightSpeed = 2.0
#sharp turn speed
SharpTurnSpeed = 0.7
IgnoreTrafficSign = False
#for some track, there would be no sharp turn, so we can always disable this check for good speed
AlwaysDisableSharpTurnCheck = False
#0 with lightgbm, 1 with mobilenetv2, 2 with leNet
TRAFFIC_CLASSIFY_MODE = 0
CLASSIFY_THRESHOLD = 0.5
SPEED_FACTOR_MAY_HAVE_TRAFFIC_SIGN=0.9
SPEED_FACTOR_TURN=0.7

DEBUG = True
# do not turn it in docker
DEBUG_IMG = False
# verbose log frequency
VERBOSE_DEBUG_FRAME_COUNT = 30000

# 处理掉头，WrongWay
HANDLE_STUCK_TIME = 1.5
HANDLE_WRONGWAY_TIME = 3.0
HANDLE_STUCK_THROTTLE_VALUE = -0.15
HANDLE_WRONGWAY_THROTTLE_VALUE = -0.15
HANDLE_STUCK_ANGLE_VALUE = 45
HANDLE_WRONGWAY_ANGLE_VALUE = 45

SharpTurnWallFactor = 0.50
HandleSharpTurnTime = 4.0
TurnHandleTime = 4.0

# 当墙的像素比达到0.5的时候，认为赛道旁边就是墙，这个时候需要谨慎处理变道
WALL_CHECK_PIXEL_COUNT_FACTOR = 0.50

# 不同的值会影响到识别正确性
# 用机器学习模型lightGBM去分类traffic sign图标 
TRAFFIC_SIGN_PIXEL_COUNT_THRESHOLD = 100#110
OBSTACLE_PIXEL_COUNT_THRESHOLD = 95
ROAD_CHECK_FRAME_RATE = 30


#前方有fork road ahead标志或障碍物避让标志，如果当前是直线行驶，需要降低速度变道,更新对应的PID参数
ChangeLaneSteeringAngleUnit = 4.0
ChangeLaneHandleTime = 0.85 #正常变一个道时间 2m/s 
MaxChangeLaneHandleTime = 2.0 #最长变道时间（可能需要变多个道）
SeeWallFactor = 0.25

WrongAngleThreshold = 45
SpeedPIDMapping = {
2.0:[0.055, 0.05, 0.02],
1.9:[0.060, 0.05, 0.02],
1.8:[0.065, 0.05, 0.02],
1.7:[0.070, 0.05, 0.02], 
1.6:[0.075, 0.05, 0.02],
1.5:[0.080, 0.05, 0.02],  
1.4:[0.085, 0.05, 0.02], 
1.3:[0.095, 0.05, 0.02], 
1.2:[0.105, 0.05, 0.02],
1.1:[0.115, 0.05, 0.02], 
1.0:[0.130, 0.05,0.02],
0.95:[0.135, 0.05,0.02],
0.9:[0.140, 0.05, 0.02], 
0.85:[0.145, 0.05, 0.02], 
0.8:[0.150, 0.05, 0.02],
0.75:[0.155, 0.05, 0.02], 
0.7:[0.185, 0.05, 0.02], 
0.65:[0.205, 0.05, 0.01],
0.6:[0.235, 0.05, 0.01],
0.55:[0.245, 0.05, 0.01],
0.5:[0.305, 0.05, 0.01],
0.4:[0.315, 0.05, 0.01]}

STEERING_PID_max_integral   = 10
THROTTLE_PID_Kp             = 0.20 #0.02
THROTTLE_PID_Ki             = 0.05 #0.005
THROTTLE_PID_Kd             = 0.05 #0.02
THROTTLE_PID_max_integral   = 0.5

MAX_STEERING_ANGLE = 60
MAX_STEERING_HISTORY = 1
MAX_THROTTLE_HISTORY = 1

# for crazy drive mode
eagle_maxspeed = 1.99
#eagle_utrun_maxspeed = 1.4
eagle_maxspeed_throttle = 0.99
eagle_angle_ratio = 10
eagle_max_turn_speed = 1.4
eagle_max_turn_speed_brake = -0.4
eagle_max_turn_speed_U_brake = 0.1
eagle_wall_ylevel = 0.4
eagle_wall_ylevel_uturn = 0.1
eagle_wall_ylevel_birdview = 0.06
eagle_wall_throttle = 40
eagle_lanes_throttle = 25
eagle_traffice_slice_ratio = 0.4
eagle_wall_density = 1.5
eagle_wall_distance_Y = 40
eagle_wall_turn_Y = 40
eagle_wall_distance_X = 50
eagle_far_wall_distanceY = 75
eagle_far_wall_distanceY_Uturn = 80
eagle_wall_bird_view = 80
eagel_traffic_sign_angle = 20
eagel_pure_wall_points = 5
eagle_wall_horizon_line_interval = 3
eagle_wall_vertical_linex = 6
eagle_turn_angle_wall_head = 20
eagle_turn_time_single_lane = 60
eagle_high_middlelane_yvalue = 9
eagle_high_alllane_yvalue = 9
eagle_high_misslane_time = 0.4

UIMG = 10 #1:nomarl, 2: birdview2