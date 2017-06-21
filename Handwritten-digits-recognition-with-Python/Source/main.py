import dataset 
import train
import test

pathTrainImages = "../Data/train-images-idx3-ubyte.gz"
pathTrainLabels = "../Data/train-labels-idx1-ubyte.gz"
pathTestImages = "../Data/t10k-images-idx3-ubyte.gz"
pathTestLabels = "../Data/t10k-labels-idx1-ubyte.gz"
pathLNSVC = "../Predict/Classifier/lnsvc.pkl"

def main():
	case = 1
	if case == 0:
		print "--------------------------------------------------"
		print "1. TRAIN"
		print "LinearSVC training..."
		time = train.linearSVC(pathTrainImages, pathTrainLabels, pathLNSVC)
		print "Time: ", time
		print "LinearSVC train done!"
		print "--------------------------------------------------"
		print "--------------------------------------------------"
		print "2. TEST"
		print "Linear SVC:"
		test.classifier(pathTestImages, pathTestLabels, pathLNSVC)
		print "--------------------------------------------------"

	else:  
		print "TODO"
	pass
if __name__ == "__main__":
	main()