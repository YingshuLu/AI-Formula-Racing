import cv2
import sys
import os
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

HEIGHT, WEIGHT = 20, 20

CLASS_NUM=7

MODEL_NAME="./release/libsvm.dat"

class Classify:
    def __init__(self):
        self._model = svm.LinearSVC()
        self._train_dataset = []
        self._train_labels =  []
        self._model_file = MODEL_NAME

    def save_model(self):
        joblib.dump(self._model, self._model_file)

    def load_model(self, model_file=MODEL_NAME):
        self._model = joblib.load(model_file)

    def _hog(self, img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bins = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        return hist

    def _preprocess(self, img):
        b,g,r = cv2.split(img)
        r = cv2.resize(r, (HEIGHT, WEIGHT), interpolation=cv2.INTER_CUBIC)
        h, w = r.shape[:2]
        r = cv2.threshold(r, 127, 255, cv2.THRESH_BINARY)[1]
        
        r = r.reshape((1, h*w)) 
        v =  r.tolist()[0]
        return v

    def _load_train_dataset(self, dirname):
        for i in range(CLASS_NUM):
            subdir = os.path.join(dirname, str(i))
            files = os.listdir(subdir)
            for f in files:
                file_name = os.path.join(subdir, f)
                img = cv2.imread(file_name)
                vector = self._preprocess(img)
                self._train_dataset.append(vector)
                self._train_labels.append(i)

    def train(self, dirname):
        self._load_train_dataset(dirname)
        self._model.fit(self._train_dataset, self._train_labels)

    def predict(self, img):
        v = self._preprocess(img)
        res = self._model.decision_function([v])[0]
        class_id = np.argmax(res)
        if class_id == CLASS_NUM - 1:
            return -1
        return class_id
          

if __name__ == '__main__':
    
    clf = Classify()
    if len(sys.argv) < 2:
        clf.train('./formula')
        clf.save_model()
        exit()

    clf.load_model()
    img = cv2.imread(sys.argv[1])
    print(clf.predict(img))

        
    

    
    
