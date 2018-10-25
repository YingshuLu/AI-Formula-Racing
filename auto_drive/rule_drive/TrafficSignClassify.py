import pickle
import os
import cv2
import numpy as np
import Settings
import Globals

import sys
import numpy as np
from datetime import datetime


lightgbm_model_path = 'models/model.bin'
lightgbm_arrowmodel_path = 'models/arrowmodel.bin'
# lenet_model_path = 'models/leNet.h5'
lenet_model_path = 'models/leNet_v2.h5'

# models for lightgbm, mobilenetv2, and letnet
lightgbm_model = None
lightgbm_arrowmodel = None
mobilenetv2_model = None
lenet_model = None

# label for mobilenetv2
label_lines = None

if Settings.TRAFFIC_CLASSIFY_MODE==0:
    np.random.seed(1337)
    f = open(lightgbm_model_path, 'rb')
    lightgbm_model = pickle.load(f, encoding='latin1')
else:
    from keras.models import load_model
    import tensorflow as tf
    lenet_model = load_model(lenet_model_path)
    global graph
    graph = tf.get_default_graph()

f = open(lightgbm_arrowmodel_path, 'rb')
lightgbm_arrowmodel = pickle.load(f, encoding='latin1')


def get_arrow_type(image):
    image = cv2.resize(image, (30,20), interpolation=cv2.INTER_AREA)
    image = np.array(image).flatten().reshape(1, -1)
    predict_prob =  lightgbm_arrowmodel.predict_proba(image)
    #print predict_prob
    prob_index = 0
    highest_score = 0
    highest_index = 0
    for prob in predict_prob[0]:
        prob_index += 1
        if prob>highest_score:
            highest_index = prob_index
            highest_score = prob
    #print highest_score, highest_index
    if highest_score>0.90:
        return highest_index
    else:
        return 0

# return the traffic sign type according to the definition in Globals.py (from 1 to 9)
def _get_traffic_sign_type_with_lightgbm(image):
    image = cv2.resize(image, (30,20), interpolation=cv2.INTER_AREA)
    image = np.array(image).flatten().reshape(1, -1)
    predict_prob =  lightgbm_model.predict_proba(image)
    #print predict_prob
    prob_index = 0
    highest_score = 0
    highest_index = 0
    for prob in predict_prob[0]:
        prob_index += 1
        if prob>highest_score:
            highest_index = prob_index
            highest_score = prob

    return (highest_index, highest_score)

def _get_traffic_sign_type_with_lenet(image):
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # resize_image = cv2.resize(gray_image,(32, 32), interpolation=cv2.INTER_CUBIC)
    # out_put = resize_image[np.newaxis,:,:,np.newaxis]

    resize_image = cv2.resize(image,(32, 32), interpolation=cv2.INTER_CUBIC)
    out_put = resize_image[np.newaxis,:,:]
    with graph.as_default():
        predict_prob =  lenet_model.predict(out_put)
    #print predict_prob

    prob_index = 0
    highest_score = 0
    highest_index = 0
    for prob in predict_prob[0]:
        prob_index += 1
        if prob>highest_score:
            highest_index = prob_index
            highest_score = prob

    return (highest_index, highest_score)

# return the traffic sign number
def get_traffic_sign_type(image):
    if Settings.TRAFFIC_CLASSIFY_MODE==0:
        label_index, score = _get_traffic_sign_type_with_lightgbm(image)
        #print (label_index, score)
        if score>=Settings.CLASSIFY_THRESHOLD:
            return label_index
        else:
            return None 
    else:
        label_index, score = _get_traffic_sign_type_with_lenet(image)
        #print (label_index, score)
        if score>=Settings.CLASSIFY_THRESHOLD:
            return label_index
        else:
            return None           

def verify_model(image_path):
    image = cv2.imread(image_path)
    return get_traffic_sign_type(image)


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-v", "--verify", action="store", dest="imageToVerify", type="string", default="", help="verify image")
    parser.add_option("-f", "--folder", action="store", dest="folderToVerify", type="string", default="", help="verify images")
    options, args = parser.parse_args()
    
    # load model
    if options.imageToVerify:
        result = verify_model(options.imageToVerify)
        print(result)
    elif options.folderToVerify:
        images_folder = options.folderToVerify
        images_in_folder = [x for x in os.listdir(images_folder) if x.endswith('.jpg')]
        from time import time
        time_begin = time()
        for image in images_in_folder:
            result = verify_model(os.path.join(images_folder, image))
            print(image, result)  
        print('cost seconds %f for %d images' % (time()-time_begin, len(images_in_folder)))