from sklearn import svm
from sklearn.externals import joblib
# import pickle
# import matplotlib.pyplot as plt
import numpy as np
import datetime
import cv2

import dataset

pathTrainImages = "../Data/train-images-idx3-ubyte.gz"
pathTrainLabels = "../Data/train-labels-idx1-ubyte.gz"
pathTestImages = "../Data/t10k-images-idx3-ubyte.gz"
pathTestLabels = "../Data/t10k-labels-idx1-ubyte.gz"
pathLNSVC = "../Predict/Classifier/lnsvc.pkl"

def linearSVC(pathTrainImages, pathTrainLabels, pathCLF):
	trainData, trainImages = dataset.trainImages(pathTrainImages)
	trainLabels = dataset.trainLabels(pathTrainLabels)

	clf = svm.LinearSVC()
	previous = datetime.datetime.now()
	clf.fit(trainData, trainLabels)
	now = datetime.datetime.now()
	time = now - previous
	joblib.dump(clf, pathCLF)
	return time
if __name__ == "__main__":
	print "LinearSVC training..."
	time = linearSVC(pathTrainImages, pathTrainLabels, pathLNSVC)
	print "Time: ", time
	print "LinearSVC train done!"
