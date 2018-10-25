# self-driving-formula-racing
AI driving formula

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

There is one major difference to the Championship Competition. In the Preliminary Competition, all races are done in a simulator. In the Championship Competition, your AI agent will control an actual self-driving car.

In the Preliminary Competition, your AI agent has to complete a certain number of laps per race, and there can be several races per day. Additionally, the AI agent also has to navigate some challenges on the track but there are traffic signs to assist you as shown below. The length of a track ranges from 60m to 90m and the maximum speed of the car is no more than 2 m/s (meters per second).

The simulator has a manual mode and an autonomous mode. In the manual mode, you can manually control the car and record data for training your AI agent. In the autonomous mode, you can see how your AI agent performs after processing the training data. The recorded data includes images and driving information. The game server will not allow you to record any data.
