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
