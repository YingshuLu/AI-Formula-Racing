# TrendFormula
TrendFormula Initial code
## Requirement and Environment
```
Python 2.7
pip install pillow
pip install flask-socketio
pip install opencv-python
pip install eventlet
pip install pandas
pip install lightgbm
pip install sklearn
pip install tensorflow
pip install keras
```
## models
* pretrained lightgbm model, mobilenetv2 and lenet model for traffic sign classification (which will do classification for a 30*20 small image that contains the traffic sign: currently, we can use the lightgbm model in python27 on windows, othewise, three models are supported, and recommend to use the lenet model, which is quick and also has high accuracy)

## Run example 
```
python AI_Bot.py
```
## Settings.py
* We can update the settings.py during the bot running, which will be reloaded immediately by the running bot.
* Update StraightSpeed in settings.py for expected running speed. (it could use higher value for track with no sharp turn or U turn)
* Update RUN_MODE in Settings.py for different drive mode. (set as 0 to run with AutoDrive.py, otherwise run with CrazyDrvie.py, which is still in development by Marc)
* Update TRAFFIC_CLASSIFY_MODE in Settings.py for different traffic sign classification model. (On windows python 27 where tensorflow is unavailable, must use 0, otherwise, suggest to use 2, which would use LeNet that has high accuracy and good speed)
* Debug setting: 
```
DEBUG=True
DEBUG_IMG=False #(only for local simulator)
```
* Sharp turn check for different track race in different week: on the practice day, we can know whether there is sharp turn in narrow checks, if no, we can disable sharp turn check for goode race speed.
```
AlwaysDisableSharpTurnCheck=True
```

## Train.py
* based on the script provied by Cao liang, which use the lightgbm to train a classifier to classify the traffic sign.

## SignType_Train.ipynb
* Jupyter notebook provided by Jeffrey that used to train the leNet CNN model for traffic sign classification

## ReplayRace.py
* Script to replay the race with the download drive log from the Race running in docker. 
* Just copy this script to the folder, such as team309-race3280-12c2bce4d75744ee83432fce2c295477-gamelog, which contains the folder 'IMG', and then run
    ```
    python ReplayRace.py IMG
    ```
## How to build the docker
* You can build different version of docker image for your different code that may works for different track.
* After the according image is built, you can tag the image for the acccording one in ai.registry.trendmicro.com
* Some example:
```
docker build -t formula-trend .
docker tag formula-trend ai.registry.trendmicro.com/309/formula-trend:rank
docker tag formula-trend ai.registry.trendmicro.com/309/formula-trend:rank.1
docker tag formula-trend ai.registry.trendmicro.com/309/formula-trend:rank.2
docker tag formula-trend ai.registry.trendmicro.com/309/formula-trend:rank.3

docker login ai.registry.trendmicro.com
docker push ai.registry.trendmicro.com/309/formula-trend:rank
docker push ai.registry.trendmicro.com/309/formula-trend:rank.1
docker push ai.registry.trendmicro.com/309/formula-trend:rank.2
docker push ai.registry.trendmicro.com/309/formula-trend:rank.3
```

## Debug log
* debug is enabled at default in the Settings.py, and log would be stored in the /log/bot.log, and traffic sign image got during the race would also be stored in the folder /log, so we can check them after the race.

