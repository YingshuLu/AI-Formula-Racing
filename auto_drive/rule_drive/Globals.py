# 
Running=False
RecordFolder = None

# traffic sign definition
MayHaveTrafficSign = 0
ForkRightSign = 1
ForkLeftSign = 2
RightTurn = 3
LeftTurn = 4
LeftUTurn = 5
RightUTurn = 6

# LeftObstacle traffic sign or found obstacle in the left side when wall is seen on the left side, need turn right
LeftObstacle = 7

# RightObstacle traffic sign or found obstacle in the right side when wall is seen on the right side, need turn left
RightObstacle = 8

# move in the wrong direction
WrongWay = 9

# stuck to wall and not moving
StuckBlackWall = 30
StuckRGWall = 31
StuckObstacle = 32

# running next to the wall
OnlyBlackWallRight = 22
OnlyBlackWallLeft = 23
OnlyRGWallRight = 24
OnlyRGWallLeft = 25
RGWallLeftBlackWallRight = 26
BlackWallLeftRGWallRight = 27
RGWallBoth = 28
BlackWallBoth = 29

