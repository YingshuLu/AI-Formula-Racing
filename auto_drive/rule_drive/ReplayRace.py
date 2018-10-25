import cv2
import sys
import os
from time import time, sleep
imagesFolder = sys.argv[1]
frameIntervalSec = 1.0/10

def show_image(img, name = "image", scale = 1.0, newsize = None):
    if scale and scale != 1.0:
        img = cv2.resize(img, newsize, interpolation=cv2.INTER_CUBIC) 

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, img)
    cv2.waitKey(1)

images_in_folder = [x for x in os.listdir(imagesFolder) if x.endswith('.jpg')]
for image in images_in_folder:
    time_begin = time()
    img = cv2.imread(os.path.join(imagesFolder,image))
    show_image(img)
    secs_to_sleep = frameIntervalSec - (time()-time_begin)
    if secs_to_sleep>0:
        sleep(secs_to_sleep)