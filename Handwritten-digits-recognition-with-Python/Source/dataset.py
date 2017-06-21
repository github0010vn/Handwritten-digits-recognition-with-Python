import numpy as np
import gzip
import struct

pathTrainImages = "../Data/train-images-idx3-ubyte.gz"
pathTrainLabels = "../Data/train-labels-idx1-ubyte.gz"
pathTestImages = "../Data/t10k-images-idx3-ubyte.gz"
pathTestLabels = "../Data/t10k-labels-idx1-ubyte.gz"

def trainImages(pathTrainImages):
	with gzip.open(pathTrainImages, "rb") as f:
		magic, nImages, nRows, nColumns = struct.unpack('>iiii', f.read(struct.calcsize('>iiii')))
		data = np.fromstring(f.read(), dtype=np.uint8)
		data = np.reshape(data, (nImages, nRows*nColumns))
		images = np.reshape(data, (nImages, nRows, nColumns))
	return data, images
def trainLabels(pathTrainLabels):
	with gzip.open(pathTrainLabels, "rb") as f:
		magic, nItems = struct.unpack('>ii', f.read(struct.calcsize('>ii')))
		labels = np.fromstring(f.read(), dtype=np.uint8)
		labels = np.reshape(labels, (nItems, ))
	return labels
def testImages(pathTestImages):
	with gzip.open(pathTestImages, "rb") as f:
		magic, nImages, nRows, nColumns = struct.unpack('>iiii', f.read(struct.calcsize('>iiii')))
		data = np.fromstring(f.read(), dtype=np.uint8)
		data = np.reshape(data, (nImages, nRows*nColumns))
		images = np.reshape(data, (nImages, nRows, nColumns))
	return data, images
def testLabels(pathTestLabels):
	with gzip.open(pathTestLabels, "rb") as f:
		magic, nItems = struct.unpack('>ii', f.read(struct.calcsize('>ii')))
		labels = np.fromstring(f.read(), dtype=np.uint8)
		labels = np.reshape(labels, (nItems, ))
	return labels
	
if __name__ == "__main__":
	print pathTrainImages
	print pathTrainLabels
	print pathTestImages
	print pathTestLabels

