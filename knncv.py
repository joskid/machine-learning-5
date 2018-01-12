# knn-cv.py (question 5)
# Implements a k-nearest-neighbor classifier and uses 5-fold cross validation on the
# training data to compute a confusion matrix and accuracy on just the training data.
#---------------------------------------------------------------------------------------
# Il Shan Ng
# March 29, 2017

import numpy as np
import pandas as pd
import knn
import argparse

def read_training_set():
	"""
	Reads in the training data set and randomly permutes the instances. Return the
	permuted training set in two arrays - one for the pixel values and one for the labels.
	"""
	# read in training data and labels
	train_data = np.genfromtxt('train_data.txt', dtype=int, delimiter=',')
	train_labels = np.array([np.genfromtxt('train_labels.txt', dtype=int, delimiter=',')])
	# concatenate pixel values and labels
	train_set = np.concatenate((train_data, train_labels.T), axis=1)
	# randomly permute rows of data set
	train_perm = np.random.permutation(train_set)
	# return permuted pixel values and labels in separate arrays
	return(train_perm[:, :np.shape(train_data)[1]], train_perm[:, np.shape(train_data)[1]])

def cross_validate(train_data, train_labels, k, distance, F=5, prints=True):
	"""
	Performs f-fold cross validation on the specified training set. Returns an array storing
	all cross-validation accuracies
	"""
	# number of training instances in each cross-validation subset
	C = train_data.shape[0]//F
	# initialize empty array to store cross-validation accuracies
	accuracy = np.zeros(F)
	# for each round of cross-validation
	for f in range(F):
		# create indices for the validation set
		validation_index = np.arange(f*C, (f+1)*C)
		# create indeices for the training set
		train_index = np.setdiff1d(np.arange(0, train_data.shape[0]), validation_index)
		# obtain predicted labels for the images in the validation set
		predicted_labels = knn.classify(train_data[train_index], train_labels[train_index],
			train_data[validation_index], k, distance)
		# compute confusion matrix for validation set
		con_matrix = knn.confusion_matrix(train_labels[validation_index], predicted_labels)
		# convert to pandas data frame to label rows and columns, then print
		# suppressed when performing cross-validation for multiple values of k
		if prints:
			con_mat_df = pd.DataFrame(con_matrix, index=['1','2','7'], columns=['1','2','7'])
			print('Cross-validation round', f+1)
			print('Confusion matrix: Predicted classes along horizontal axis. Actual classes along vertical axis.')
			print(con_mat_df)
		# compute and store cross-validation accuracy
		accuracy[f] = knn.accuracy(con_matrix)
	return accuracy

def wrapper(train_data, train_labels, distance, K=[1,3,5,7,9], F=5):
	"""
	Wrapper function that runs f-fold cross-validation for different values of k.
	Do this for k=1,3,5,7,9 by default. Returns a data frame storing the cross-validation
	accuracies for each value of k.
	"""
	# initialize empty array to store cross-validation accuracies for different k
	accuracies = np.empty(shape=(0,F), dtype=float)
	# for each value of k
	for k in K:
		# compute and store cross-validation accuracies
		accuracy = cross_validate(train_data, train_labels, k, distance, F, prints=False)
		accuracies = np.append(accuracies, np.array([accuracy]), axis=0)
		print('Completed cross-validation for k =', k)
	# convert to pandas data frame with labeled rows and columns
	accuracies_df = pd.DataFrame(accuracies, index=['k=1','k=3','k=5','k=7','k=9'], 
									columns=['f=1','f=2','f=3','f=4','f=5'])
	# add a new column for the average cross-validation accuracy
	accuracies_df['average'] = accuracies_df.mean(axis=1)
	return accuracies_df

def main():
	# obtain command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('k',
		help='Integer number of closest neighbors to use when making a prediction; \
		type "all" (without quotes) to run program for k=1,3,5,7,9')
	parser.add_argument('distance', type=str,
		help='String representing the distance metric to use (euclidean or manhattan)')
	flags, unparsed = parser.parse_known_args()

	# obtain permuted training data and labels
	train_data, train_labels = read_training_set()
	# if a value for k is specified
	if flags.k != 'all':
		# perform 5-fold cross validation for that k and obtain accuracies
		accuracy = cross_validate(train_data, train_labels, int(flags.k), flags.distance)
		print('Cross-validation accuracies are:', accuracy)
		print('Average accuracy is:', np.mean(accuracy))
	# if all values of k are to be tested
	elif flags.k == 'all':
		# call the wrapper function to perform 5-fold cross validation for each k
		accuracies_df = wrapper(train_data, train_labels, flags.distance)
		print(accuracies_df)
		# write data frame of accuracies to a text file
		accuracies_df.to_csv('accuracies_' + flags.distance, header=True, index=True)

if __name__ == '__main__':
	main()