# neural.py
# CS321 Artificial Intelligence final project
# -----------------------------------------------------------------------------------
# Implements a feed-forward artificial neural network with 1 hidden layer to classify
# MNIST handwritten digits. The basic implementation uses ReLU activation functions, 
# a squared loss function, and mini-batch gradient descent for the parameter updates. 
# The advanced components include options to turn on L2 regularization, dropout, 
# Nesterov momentum, Xavier/He initialization of weights, and use a cross-entropy loss 
# function. Also implements gradient-checking to verify accuracy of parameter updates.
# -----------------------------------------------------------------------------------
# Frank Yang and Il Shan Ng
# March 9, 2017

import numpy as np
import time
import argparse

class neural():
	"""
	Class to store parameters related to basic neural network architecture.
	"""
	def __init__(self, num_features, num_hidden, num_classes, learn_rate, reg_strength, drop_prop):
		"""
		Constructor to create neural network with user-specified architecture.
		"""
		self.F = num_features 	 	  # number of features (784 pixels in this case)
		self.H = num_hidden   	 	  # number of hidden units
		self.C = num_classes   	 	  # number of classes (10 digits in this case)
		self.alpha = learn_rate  	  # learning rate for parameter update
		self.regStr = reg_strength    # L2 regularization strength 
		self.dropProp = drop_prop     # dropout proportion

	def initialize_weights(self, XavierHe):
		"""
		Function that initializes the two matrices of weights, with biases incorporated into
		these matrices.
		"""
		# if using Xavier/He initialization
		if XavierHe:
			# first matrix will have dimension H x (F+1)
			self.W1 = np.random.normal(size = (self.H, self.F+1), loc = 0, scale = np.sqrt(2/(self.H + self.F)))
			self.W2 = np.random.normal(size = (self.C, self.H+1), loc = 0, scale = np.sqrt(2/(self.C + self.H)))
			# second matrix will have dimension C x (H+1)
		else:
			self.W1 = np.random.uniform(size = (self.H, self.F+1), low = -0.5, high = 0.5)
			self.W2 = np.random.uniform(size = (self.C, self.H+1), low = -0.5, high = 0.5)

	def compute_scores(self, data, useDropout):
		"""
		Function that computes the output class scores through a series of matrix multiplications.
		Uses a ReLU activation function for the hidden layer units.
		"""
		# append row of ones (bias) to top of data set
		self.X_bias = np.append(np.ones((1, data.shape[1])), data, axis = 0)
		# compute scores entering hidden layer units and apply activation function
		self.A = np.maximum(0, np.dot(self.W1, self.X_bias))
		# append row of ones (bias) to top of matrix A
		self.A_bias = np.append(np.ones((1, data.shape[1])), self.A, axis = 0)
		# if using dropout, create random matrix to determine which weights are zeroed out
		if useDropout: 
			D = (np.random.rand(self.H+1, self.N) > self.dropProp)/(1 - self.dropProp)
		else: D = 1
		# compute scores entering output layer and apply activation function
		output_scores = np.maximum(0, np.dot(self.W2, self.A_bias*D).transpose())
		# return output class scores
		return(output_scores)

	def backpropagate(self, diff):
		"""
		Backpropagate gradients through the neural network to compute the analytical 
		gradients for both matrices of weights.
		"""
		# create indicator matrix to represent output of ReLU activation function in output layer
		G2 = np.zeros(shape = (self.H+1, self.C))
		A_diff = np.dot(self.A_bias, diff)
		G2[A_diff > 0] = 1
		# compute gradient with respect to weights matrix W2
		dW2 = np.transpose(A_diff)/self.N
		# create indicator matrix to represent output of ReLU activation function used in hidden layer
		G1 = np.zeros(shape = (self.H, self.N))
		G1[self.A > 0] = 1
		# compute gradients with respect to weights matrix W1
		dW1 = np.dot(np.transpose(np.dot(diff, self.W2[:, 1:]))*G1, self.X_bias.transpose())/self.N 
		return(dW1, dW2)

	def gradient_descent(self, dW1, dW2, L2Reg):
		"""
		Function that performs the parameter updates using mini-batch gradient descent.
		"""
		self.W1 -= self.alpha*dW1
		self.W2 -= self.alpha*dW2
		# if using L2 regularization
		if L2Reg:
			self.W1 -= self.alpha * self.regStr * self.W1
			self.W2 -= self.alpha * self.regStr * self.W2

	def train(self, num_iters, train_data, batch_size, train_labels, crossEnt, L2Reg, dropout, print_loss):
		"""
		Wrapper function that implements the training loop for the specified number of iterations.
		Returns the time required for training.
		"""
		# set number of training examples to equal batch size
		self.N = batch_size
		# start timer
		start_time = time.time()
		# begin training loop
		for i in range(num_iters):
			# randomly sample from data for mini-batch gradient descent
			batch_index = np.random.randint(low = 0, high = train_data.shape[0], size = batch_size)
			# recast vector of true labels into binary matrix
			labels_mat = np.zeros((batch_size, self.C))
			labels_mat[np.arange(batch_size), train_labels[batch_index]] = 1
			# obtain output class scores
			scores = self.compute_scores(data = np.transpose(train_data[batch_index]), useDropout = dropout)
			# if using cross entropy loss function
			if crossEnt:
				# compute difference matrix and mean cross entropy loss
				diff, loss = crossEnt_loss(scores, labels_mat)
			else:
				# compute difference matrix and mean squared loss
				diff, loss = squared_loss(scores, labels_mat)
			# if using L2 regularization, add in regularization loss
			if L2Reg: loss += self.L2Reg_loss()
			# print loss on every (num_iters/50)th iteration
			if print_loss & (i % (num_iters/20) == 0):
				print("Iteration %i: loss is %.4f" % (i, loss))
			# compute gradients w.r.t W1 and W2
			dW1, dW2 = self.backpropagate(diff)
			# perform parameter update using mini-batch gradient descent
			self.gradient_descent(dW1, dW2, L2Reg)
		# return elapsed training time
		return(time.time() - start_time)

	def compute_accuracy(self, data, labels):
		"""
		Function that computes the accuracy of the trained neural network on the given data set.
		"""
		# compute output class scores, turning dropout off
		scores = self.compute_scores(data, useDropout = False)
		# predict class for each data point
		pred_class = np.argmax(scores, axis = 1)
		# calculate number of correct predictions
		num_correct = np.sum(pred_class == labels)
		# return accuracy rate
		return(num_correct/data.shape[1])

	def L2Reg_loss(self):
		"""
		Returns the average L2 regularization loss over all training examples.
		"""
		return((0.5 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))))/self.N)

def read_data(path, rows):
	"""
	Function that reads in training/test data set and returns the features (normalized 
	pixel values) and ground truth labels as separate numpy arrays.
	"""
	data = np.genfromtxt(fname = path, skip_header = 1, delimiter = ",", max_rows = rows)
	return(data[ : , 0:-1], data[ : , -1].astype(dtype = int))

def squared_loss(scores, labels_mat):
	"""
	Function that takes in the output class scores from the neural network and the ground
	truth labels to compute the mean squared loss across all training examples. 
	"""
	# compute the difference matrix and mean squared loss
	diff = scores - labels_mat
	loss = 0.5*np.sum(np.square(diff))/scores.shape[0]
	# return the difference matrix and loss
	return(diff, loss) 

def crossEnt_loss(scores, labels_mat):
	"""
	Function that takes in the output class scores from the neural network and the ground
	truth labels to compute the mean cross entropy loss across all training examples. 
	"""
	# exponentiate matrix of class scores to get un-normalized probabilities
	P = np.exp(scores)
	# for each data point, sum up exponentiated scores over all classes
	Z = np.sum(P, axis = 1)
	# normalize to get class probabilities
	P = P/np.reshape(Z, newshape = (scores.shape[0], 1))
	# for each data point, store only probability of true class and collapse to vector
	P_true = np.sum(P * labels_mat, axis = 1)
	# compute cross-entropy loss
	loss = np.sum(-np.log(P_true))/scores.shape[0]
	# return difference matrix and loss
	return(P - labels_mat, loss)

def user_options():
	"""
	Function to parse and return command line options.
	"""
	parser = argparse.ArgumentParser(description = "Command line options")
	parser.add_argument("--numIters", dest = "num_iters", type = int, help = "Number of training iterations")
	parser.add_argument("--numTrain", dest = "num_train", type = int, help = "Number of training images")
	parser.add_argument("--numTest", dest = "num_test", type = int, help = "Number of test images")
	parser.add_argument("--batchSize", dest = "batch_size", type = int, help = "Batch size for mini-batch gradient descent")
	parser.add_argument("--crossEnt", dest = "cross_entropy", action = "store_true", help = "Use cross-entropy loss function")
	parser.add_argument("--L2Reg", dest = "L2_reg", action = "store_true", help = "Turn on L2 regularization")
	parser.add_argument("--XavierHe", dest = "XavierHe_init", action = "store_true", help = "Use Xavier/He initialization of weights")
	parser.add_argument("--dropout", dest = "dropout", action = "store_true", help = "Use dropout during training")
	parser.add_argument("--K", dest = "K", type = int, help = "Number of rounds of cross-validation")
	parser.set_defaults(numTrain = None, numTest = None, K = 1)
	return(parser.parse_args())

def main():

	# parse command line options
	args = user_options()

	# import training data and labels
	train_set, train_labels = read_data(path = "digits_train_scaled.csv", rows = args.num_train)
	print("Successfully loaded training data with %i training images" % train_set.shape[0])
	
	# initialize neural network object with specified command line arguments
	neuralNet = neural(num_features = train_set.shape[1], num_hidden = 100, num_classes = 10, learn_rate = 0.005, \
		reg_strength = 0.1, drop_prop = 0.1)
	# initialize weights according to specified method
	neuralNet.initialize_weights(XavierHe = args.XavierHe_init)
	# train neural network for specified number of iterations and obtain training time
	time = neuralNet.train(num_iters = args.num_iters, train_data = train_set, batch_size = args.batch_size, \
		train_labels = train_labels, crossEnt = args.cross_entropy, L2Reg = args.L2_reg, dropout = args.dropout)
	
	# compute accuracy of trained neural network on training data
	accuracy = neuralNet.compute_accuracy(data = train_set.transpose(), labels = train_labels)
	print("Training accuracy on %i training images: %.4f" % (train_set.shape[0], accuracy))
	print("Training time with %i iterations: %.2f seconds" % (args.num_iters, time))
	
	# import test data and labels
	test_set, test_labels = read_data(path = "digits_test_scaled.csv", rows = args.num_test)
	# compute accuracy on test data
	test_accuracy = neuralNet.compute_accuracy(data = test_set.transpose(), labels = test_labels)
	print("Accuracy on %i test images: %.4f" % (test_set.shape[0], test_accuracy))

if __name__ == '__main__':
	main()