# knn.py (question 2)
# Implements a k-nearest-neighbor classifier.
# Loops through all images in the test data set and uses the training data set and the
# k-nn algorithm to predict which class the image belongs to.
#---------------------------------------------------------------------------------------
# Il Shan Ng
# March 29, 2017

import numpy as np
import pandas as pd
import argparse

def read_dataset():
	"""
	Read in the training and test data sets and return them stored as numpy arrays.
	"""
	train_data = np.genfromtxt('train_data.txt', dtype=int, delimiter=',')
	train_labels = np.genfromtxt('train_labels.txt', dtype=int, delimiter=',')
	test_data = np.genfromtxt('test_data.txt', dtype=int, delimiter=',')
	test_labels = np.genfromtxt('test_labels.txt', dtype=int, delimiter=',')
	return train_data, train_labels, test_data, test_labels

def euclidean_distance(x1, x2):
	"""
	Returns the Euclidean distance between a pair of vectors.
	"""
	return np.sqrt(np.sum(np.square(x1-x2)))

def manhattan_distance(x1, x2):
	"""
	Returns the Manhattan distance between a pair of vectors.
	"""
	return np.sum(np.absolute(x1-x2))

def classify(train_data, train_labels, test_data, k, distance):
	"""
	Loops through all images in the test data set and classifies them using the training
	data set and the k-nn algorithm. Returns a vector of predicted labels for the test
	data set.
	"""
	# define distance metric
	if distance == 'euclidean':
		compute_distance = euclidean_distance
	elif distance == 'manhattan':
		compute_distance = manhattan_distance
	# initialize empty array to store predicted labels
	pred_labels = np.array([], dtype=int)
	# for each image in the test data set
	for test_image in test_data:
		# initialize empty array to store distances
		distances = np.array([])
		# iterate through all training instances
		for n in range(train_data.shape[0]):
			# compute the specified distance and store in array
			distances =  np.append(distances, compute_distance(test_image, train_data[n, ]))
		# return indices of k closest training instances
		# if multiple instances have the same distance, choose the first occurence
		indices = np.argsort(distances)[:k]
		# create count of each training label
		counts = np.bincount(train_labels[indices])
		# find majority label, tie-breaking by choosing a random label
		label = np.random.choice(np.flatnonzero(counts == counts.max()))
		# store majority label in array of predicted labels
		pred_labels = np.append(pred_labels, label)
	return pred_labels

def confusion_matrix(test_labels, pred_labels):
	"""
	Returns a confusion matrix based on the true and predicted labels for the test
	data set.
	"""
	# store the unique classes in this problem
	classes = np.unique(test_labels)
	# initialize empty confusion matrix
	con_matrix = np.zeros((np.size(classes), np.size(classes)), dtype=int)
	# for each test image, compare the predicted and true labels
	for n in range(np.size(test_labels)):
		# if the labels match up
		if test_labels[n] == pred_labels[n]:
			# find the index of the class
			i = np.where(classes==test_labels[n])
			# and increment the count of the corresponding diagonal entry in the confusion matrix
			con_matrix[i, i] += 1 
		# if the labels don't match up
		else:
			# find the index of the actual and predicted class
			i_actual = np.where(classes==test_labels[n])
			i_pred = np.where(classes==pred_labels[n])
			# and increment the count of the corresponding off-diagonal entry
			con_matrix[i_actual, i_pred] += 1
	return con_matrix

def accuracy(con_matrix):
	"""
	Takes in a confusion matrix and returns the accuracy of the classification.
	"""
	return np.trace(con_matrix)/np.sum(con_matrix)

def main():
	# obtain command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('k', type=int, 
		help='Integer number of closest neighbors to use when making a prediction')
	parser.add_argument('distance', type=str,
		help='String representing the distance metric to use (euclidean or manhattan)')
	flags, unparsed = parser.parse_known_args()

	# read in training and test data sets as numpy arrays
	train_data, train_labels, test_data, test_labels = read_dataset()
	# classify test data images and obtain predicted labels
	predicted_labels = classify(train_data, train_labels, test_data, flags.k, flags.distance)
	# write predicted labels to a text file for further analysis
	np.savetxt('pred_labels', predicted_labels, fmt='%i', delimiter=',')
	# compute confusion matrix
	con_matrix = confusion_matrix(test_labels, predicted_labels)
	# convert to pandas data frame to label rows and columns, then print
	con_mat_df = pd.DataFrame(con_matrix, index=['1','2','7'], columns=['1','2','7'])
	print('Confusion matrix: Predicted classes along horizontal axis. Actual classes along vertical axis.')
	print(con_mat_df)
	# print accuracy
	print('Accuracy:', accuracy(con_matrix))

if __name__ == '__main__':
	main()