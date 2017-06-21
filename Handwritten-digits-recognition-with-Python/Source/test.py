from sklearn.externals import joblib

import dataset

pathTrainImages = "../Data/train-images-idx3-ubyte.gz"
pathTrainLabels = "../Data/train-labels-idx1-ubyte.gz"
pathTestImages = "../Data/t10k-images-idx3-ubyte.gz"
pathTestLabels = "../Data/t10k-labels-idx1-ubyte.gz"
pathLNSVC = "../Predict/Classifier/lnsvc.pkl"

def classifier(pathTestImages, pathTestLabels, pathCLF):
	testData, testImages = dataset.testImages(pathTestImages)
	testLabels = dataset.testLabels(pathTestLabels)

	clf = joblib.load(pathCLF)
	print "Predict: ", clf.predict(testData)
	print "Real: ", testLabels
	print "Accuracy: ", clf.score(testData, testLabels)
	pass
if __name__ == "__main__":
	print "Linear SVC:"
	classifier(pathTestImages, pathTestLabels, pathLNSVC)