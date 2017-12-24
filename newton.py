# Implements Newton's method for optimising the average empirical loss for logistic regression.
# Also known as Fisher scoring.
# 23 Dec 2017

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

# read in input and output files as numpy arrays
X_raw = np.genfromtxt('logistic_x.txt', dtype = float)
Y = np.genfromtxt('logistic_y.txt', dtype = float)

# add column of 1s to input as the bias
X = np.concatenate((np.ones((99, 1)), X_raw), axis = 1)

# store number of trainig examples
M = X.shape[0]

# initialise parameters to zero (2 features plus 1 bias)
theta = np.zeros(3)

# begin main training loop
for _ in range(10):
	# compute the vectorized output of the sigmoid function
	g = sp.expit(Y * np.sum(theta * X, axis = 1))
	# compute the gradient using all training examples
	grad = -(1/M) * np.sum((np.reshape((np.ones(M) - g) * Y, newshape = (99, 1))) * X, axis = 0)
	# compute the hessian using all training examples
	mat = np.empty((99, 0))
	for j in range(len(theta)):
		for k in range(len(theta)):
			mat = np.append(mat, X[:,[j]] * X[:,[k]], axis = 1)
	hess_flat = (1/M) * np.sum(np.reshape(g * (np.ones(M) - g), newshape = (99, 1)) * mat, axis = 0)
	hess = np.reshape(hess_flat, newshape = (len(theta), len(theta)))
	# perform parameter update
	theta = theta - np.dot(np.linalg.inv(hess), grad)
	# compute and print loss
	loss = (1/M) * np.sum(np.log(1 + np.exp(-Y * np.sum(theta * X, axis = 1))), axis = 0)
	print("Loss is %.8f" % loss)

# print trained parameters
print("Trained parameters are (bias last): ", theta)

# plot classification of training data
plt.plot(X[Y==1, 1], X[Y==1 ,2], "r_", label = "-1")
plt.plot(X[Y==-1, 1], X[Y==-1, 2], "b+", label = "+1")
plt.xlabel("x1"); plt.ylabel("x2")	
plt.legend()

# define function to plot decision boundary
def graph(formula, x_range):
	x = np.array(x_range)
	y = eval(formula)
	plt.plot(x, y, "g--")

# plot decision boundary and show complete plot
graph(formula = "(1/theta[2]) * (-theta[0] - theta[1] * x)", x_range = range(0, 9))
plt.title("Decision boundary for logistic regression")
plt.show()