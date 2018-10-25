### Importing packages. 
import os
import math
import base64
import itertools
import pickle
import cv2
import json

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, accuracy_score

from PIL import Image
from io import BytesIO


np.random.seed(1337) 

def preprocessImage(image, new_size_row=20, new_size_col=30):
    # note: numpy arrays are (row, col)!
    image = cv2.resize(image,(new_size_col, new_size_row), interpolation=cv2.INTER_AREA)    
    return image 


def preprocess_image_file(line_data):
    # Preprocessing training files and augmenting
    path_file = line_data['filepath'].strip()    
    y_sign = line_data['signtype']
    image = cv2.imread(path_file)
    image = preprocessImage(image)
    image = np.array(image)
    
    return image,y_sign


def generate_valid_from_PD(data, verbose=False):
    # Validation generator
    while 1:
        for i_line in range(len(data)):
            line_data = data.iloc[i_line]
            if verbose: print('to handle data: %d, %s'%(i_line, line_data['center']))
            
            x,y = line_data['XY']
            
            yield x, y-1


def train_model(folderToTrain, train_ratio = 0.9):
    # split the images to train and valid part

    files_to_train = []
    files_to_valid = []
    ### Loading CSV data
    class_num = 0
    for sub_folder in os.listdir(folderToTrain):
        sub_folder_path = os.path.join(folderToTrain, sub_folder)
        if os.path.isdir(sub_folder_path):
            class_num += 1
            images_in_folder = [x for x in os.listdir(sub_folder_path) if x.endswith('.jpg')]
            images_count_to_train = int(len(images_in_folder)*train_ratio)
            images_to_train = images_in_folder[0:images_count_to_train]
            images_to_valid = images_in_folder[images_count_to_train:]

            images_to_train_with_y = [('%s/%s'%(sub_folder_path, image_name), int(sub_folder)) for image_name in images_to_train]
            images_to_valid_with_y = [('%s/%s'%(sub_folder_path, image_name), int(sub_folder)) for image_name in images_to_valid]
            files_to_train += images_to_train_with_y
            files_to_valid += images_to_valid_with_y

    data_files_s_train = pd.DataFrame(files_to_train, columns=['filepath', 'signtype'])
    data_files_s_valid = pd.DataFrame(files_to_valid, columns=['filepath', 'signtype'])
    
    ### pre-load the data
    data_files_s_train['XY'] = data_files_s_train.apply(preprocess_image_file, axis=1)
    data_files_s_valid['XY'] = data_files_s_valid.apply(preprocess_image_file, axis=1)

    #data_files_s_train = data_files_s.sample(frac=1)
    print('loaded train files: %d'%len(data_files_s_train))
    print('loaded validate files: %d'%len(data_files_s_valid))    
    
    dtrain = [[x.flatten(), y] for x,y in itertools.islice(generate_valid_from_PD(data_files_s_train), len(data_files_s_train))]
    train_x, train_y = zip(*dtrain)
    train_x, train_y = np.array(train_x), np.array(train_y)

    dtrain = [[x.flatten(), y] for x,y in itertools.islice(generate_valid_from_PD(data_files_s_valid), len(data_files_s_valid))]
    valid_x, valid_y = zip(*dtrain)
    valid_x, valid_y = np.array(valid_x), np.array(valid_y)
    print (valid_x, valid_y)
    
    ## Define model 
    clf = LGBMClassifier(
        nthread=class_num,
        objective='multiclass',
        num_class=class_num,
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=40,
        colsample_bytree=0.70,
        subsample=0.9,
        max_depth=-1,
        reg_alpha=0.08,
        reg_lambda=0.08,
        min_split_gain=0.04,
        min_child_weight=25,
        random_state=0,
        silent=-1,
        verbose=-1,
        device='cpu',
        gpu_platform_id=0,
        gpu_device_id=0,
    )
    
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric='multi_error', verbose=1000, early_stopping_rounds=200)
    
    oof_preds = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)

    print('LogLoss: %.6f' % log_loss(valid_y, oof_preds))
    print('Accuracy: %.6f' % accuracy_score(valid_y, np.argmax(oof_preds, axis=1)))
    return clf

def verify_model(image_path):
    f = open('models/model.bin', 'rb')
    model = pickle.load(f)
    image = cv2.imread(image_path)
    image = preprocessImage(image)
    image = np.array(image).flatten().reshape(1, -1)
    return model.predict_proba(image)  

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-t", "--train", action="store", dest="folderToTrain", type="string", default="", help="to train the ML model with images in folderToTrain")
    parser.add_option("-v", "--verify", action="store", dest="imageToVerify", type="string", default="", help="verify image")

    options, args = parser.parse_args()
    
    # load model
    if options.folderToTrain:
        print (options.folderToTrain)
        model = train_model(options.folderToTrain)
        f = open('models/model.bin', 'wb')
        pickle.dump(model, f)
    elif options.imageToVerify:
        result = verify_model(options.imageToVerify)
        print(result)
    
    else:
        raise Exception('start without argument')
