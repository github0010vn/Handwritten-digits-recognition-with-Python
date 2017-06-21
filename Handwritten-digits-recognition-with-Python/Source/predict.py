from sklearn.externals import joblib
import numpy as np
import cv2

import dataset

pathTrainImages = "../Data/train-images-idx3-ubyte.gz"
pathTrainLabels = "../Data/train-labels-idx1-ubyte.gz"
pathTestImages = "../Data/t10k-images-idx3-ubyte.gz"
pathTestLabels = "../Data/t10k-labels-idx1-ubyte.gz"
pathLNSVC = "../Predict/Classifier/lnsvc.pkl"

pathImage01 = "../Predict/Images/digits.png"
pathImage02 = "../Predict/Images/digits02.jpg"
pathImage03 = "../Predict/Images/digits03.png"
pathImage04 = "../Predict/Images/digits04.png"

def predict(pathImage):
	original = cv2.imread(pathImage, cv2.IMREAD_COLOR)
	gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
	ret, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
	contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	c = 1
	clf = joblib.load(pathLNSVC)
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		print x, y, w, h
		rect = thresh[y:y+h, x:x+w]
		rect = cv2.copyMakeBorder(rect, 10, 10, 10, 10, cv2.BORDER_CONSTANT,value=[0, 0, 0])
		new = cv2.resize(rect, (28, 28))
		d = np.reshape(new, (784, ))

		# cv2.imshow(str(c), cv2.resize(new, (112, 112)))
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		print "LNSVC predict: ", int(clf.predict([d]))

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(original, str(int(clf.predict([d]))), (x + w, y + h), font, 0.5, (0, 0, 255), 1)

		c += 1
		cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 1)
	cv2.imshow('original', original)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	pass
if __name__ == "__main__":
	print "1."
	predict(pathImage01)
	print "2."
	predict(pathImage02)
	print "3."
	predict(pathImage03)
	print "4."
	predict(pathImage04)
