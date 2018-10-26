# AI Formula Racing
AI based driving formula

![Simulator](https://raw.githubusercontent.com/YingshuLu/AI-Formula-Racing/master/sample/race.png)

# Terminology


## Race
A race is a car driving on a track trying to complete one or more objectives in the shortest amount of time.
There can be several races per day

## Lap
One lap consists of the car having driven around the entire track once from start to start.
The time it takes to complete one lap is recorded as lap time, and it is the time measurement unit of this game.
A player's best lap time of the day is used to decide their score.

## Simulator ([Download](https://drive.google.com/file/d/15HoSwbtZnr35t4MGaDBXbpkm4gNy6QhS/view?usp=sharing))
An executable file to run a virtual environment for the game.
The simulator contains several tracks for racing and practice.
The simulator and a team's AI agent communicate via Socket.IO.

## Self-driving car
A self-driving car (also called an autonomous car or driverless car) is a vehicle that uses a combination of sensors, cameras, radar and artificial intelligence (AI) to navigate an environment without human input.

## Obstacles
Obstacles can be one or more stationary vehicles placed on the track.
Formula Trend Preliminary Competition
Formula Trend is a racing competition that is designed specifically for AI agents. In the competition, an AI agent has to control a car based on footage from the front bumper camera and other information, such as speed, throttle, brakes and so on. Just like Formula 1, the faster you finish a lap, the higher your rank will be.
![OBSTACLE](https://raw.githubusercontent.com/YingshuLu/AI-Formula-Racing/master/sample/Obstable.jpg)

## Traffic sign
Traffic sign tell AI agent which track is the shortest one, AI agent should follow the direction of the sign.

![Left fork](https://raw.githubusercontent.com/YingshuLu/AI-Formula-Racing/master/sample/sign-08.png)
![Right Fork](https://raw.githubusercontent.com/YingshuLu/AI-Formula-Racing/master/sample/sign-07.png)

# Description
In the Competition, your AI agent has to complete a certain number of laps per race, and there can be several races per day. Additionally, the AI agent also has to navigate some challenges on the track but there are traffic signs to assist you as shown below. The length of a track ranges from 60m to 90m and the maximum speed of the car is no more than 2 m/s (meters per second).

The simulator has a manual mode and an autonomous mode. In the manual mode, you can manually control the car and record data for training your AI agent. In the autonomous mode, you can see how your AI agent performs after processing the training data. The recorded data includes images and driving information. The game server will not allow you to record any data.


# Example
Please refer to [my video](https://www.youtube.com/watch?v=bvX_Qhs79-E&t=466s) to realize more details.


# Dirving control

## Turn
We try to separate the image to left bottom and right bottom, and count black wall pixel.
```
@staticmethod
def wall_detector(img):
    img_height = img.shape[0]
    img_width = img.shape[1]
    half_width = int(img_width * 0.5)
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

    return left_wall_distance, right_wall_distance
```

## Angle:
if left black color count bigger than right black color count, then turn right, else turn left.

we limit the angle from +8° to -8° to keep speed. (0.15)


```
@staticmethod
def find_wall_angle(left_wall_distance, right_wall_distance):
    count_rate = (right_wall_distance / left_wall_distance) if left_wall_distance > 0 else 0
    count_rate = max(min(count_rate, 0.15), 0)
    steering_angle = -count_rate if right_wall_distance > left_wall_distance else count_rate
    return steering_angle

```

## Throttle:
always set to 1.

Performance
the code is running very fast, because all we need is count the black color and calculate rate.
strength
if the map change road color, it's still work.
if you need big angle like U turn, you can just write a logical like: when wall rate is bigger than "0.3" then the max angle can be increase to "1.0".
weakness
because the cam view is not 180 degree, so sometime the image from cam didn't have any wall, but actually the wall is very close, you may crash when the car is too closing the wall.
the railing under traffic sign is black too, if you don't have rule to clean it, the rate will be inaccurate.
