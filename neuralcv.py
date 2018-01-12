# neuralcv.py
# CS321 Artificial Intelligence final project
# ------------------------------------------------------------------------
# Implements k-fold cross-validation for the neural network in neural.py.
# ------------------------------------------------------------------------
# Frank Yang and Il Shan Ng
# March 9, 2017

import neural
import numpy as np

def cross_validate(num_iters, train_data, batch_size, train_labels, neuralNet, crossEnt, L2Reg, \
							dropout, XavierHe, K):
	"""
	Function that implements k-fold cross-validation. Divides the training data into k subsets.
	Retain a single subset as the validation set, and pool the remaining k-1 subsets for use as 
	the training set. Repeat this k times to compute k cross-validation accuracies for use in
	parameter fine-tuning.
	"""
	# determine number of rows in validation set
	num_val = train_data.shape[0]//5
	# create array to store training accuracies
	accuracies = np.empty(K)
	# for each round of cross-valiation
	for k in range(K):
		# determine index of validation and training sets
		val_index = np.arange(k*num_val, (k+1)*num_val)
		train_index = np.setdiff1d(np.arange(0, train_data.shape[0]), val_index)
		# initialize weights according to specified method
		neuralNet.initialize_weights(XavierHe)
		# train neural network for specified number of iterations and obtain training time
		time = neuralNet.train(num_iters, train_data[train_index], batch_size, train_labels[train_index], crossEnt, \
										L2Reg, dropout, print_loss = False)
		# compute accuracy on validation set
		accuracies[k] = neuralNet.compute_accuracy(data = np.transpose(train_data[val_index]), labels = train_labels[val_index])
		print("Training accuracy for k=%i: %.4f" %(k+1, accuracies[k]))
		print("Training time for k=%i with %i iterations: %.4f seconds" %(k+1, num_iters, time))
	# return k training accuracies
	return(accuracies)

def main():
	# parse command line options from neural.py
	args = neural.user_options()

	# import training data and labels
	train_set, train_labels = neural.read_data(path = "digits_train_scaled.csv", rows = args.num_train)
	print("Successfully loaded training data with %i training images." % train_set.shape[0])
	
	# initialize neural network object with specified command line arguments
	neuralNet = neural.neural(num_features = train_set.shape[1], num_hidden = 100, num_classes = 10, \
		learn_rate = 0.001, reg_strength = 0.1, drop_prop = 0.1)

	# run k-fold cross-validation
	accuracies = cross_validate(num_iters = args.num_iters, train_data = train_set, batch_size = args.batch_size, \
		train_labels = train_labels, neuralNet = neuralNet, crossEnt = args.cross_entropy, L2Reg = args.L2_reg, 
		dropout = args.dropout, XavierHe = args.XavierHe_init, K = args.K)
	print("Average %i-fold cross-validation training accuracy: %.4f" % (args.K, np.mean(accuracies)))

	# import test data and labels
	test_set, test_labels = neural.read_data(path = "digits_test_scaled.csv", rows = args.num_test)
	# compute accuracy on test data
	test_accuracy = neuralNet.compute_accuracy(data = test_set.transpose(), labels = test_labels)
	print("Accuracy on %i test images: %.4f" % (test_set.shape[0], test_accuracy))

if __name__ == '__main__':
	main()