# naiveBayes.py
# Implements a Naive Bayes classifier to predict whether students pass or fail the 
# Writing Portfolio based on a set of features. Uses leave-one-out cross validation
# to compute accuracy rates.
#---------------------------------------------------------------------------------------
# Il Shan Ng
# April 15, 2017

import numpy as np
from scipy import stats
import math
import argparse

def read_data(col_names):
	"""
	Reads in a csv file containing the data. Returns the feature values, project IDs and
	ground truth labels in separate arrays. The argument col_names is a vector that
	specifies the name of each feature.
	"""
	# read in data for features
	data = np.genfromtxt('writingportfolio.csv', delimiter=',', dtype=None, usecols=range(1,14), 
						 skip_header=1, names=col_names)
	# read in project IDs and true labels
	IDs = np.genfromtxt('writingportfolio.csv', delimiter=',', dtype=str, usecols=0, skip_header=1)
	labels = np.genfromtxt('writingportfolio.csv', delimiter=',', dtype=int, usecols=14, skip_header=1)
	# return feature values, project IDs and true labels separately
	return(data, IDs, labels)

def num_positive(labels):
	"""
	Takes in a vector of ground truth labels and returns the number of projects labelled
	as 'needs work'.
	"""
	return(np.sum(labels==1))

def discretize(data):
	"""
	Discretizes the abroad, AP, CS, english and science features by binning the continuous
	values into ranges.
	"""
	# bin number of abroad credits into low (0), medium (1) and high (2)
	data['abroad'][data['abroad']<=10] = 0
	data['abroad'][(data['abroad']>10)&(data['abroad']<=16)] = 1
	data['abroad'][data['abroad']>16] = 2
	# bin number of AP credits into low (0), medium (1) and high (2)
	data['AP'][data['AP']<=0] = 0
	data['AP'][(data['AP']>0)&(data['AP']<=24)] = 1
	data['AP'][data['AP']>24] = 2
	# bin number of CS credits into low (0), medium (1) and high (2)
	data['CS'][data['CS']==0] = 0
	data['CS'][(data['CS']>0)&(data['CS']<=18)] = 1
	data['CS'][data['CS']>18] = 2
	# bin number of english credits into low (0), medium (1), high (2) and very high (3)
	data['english'][data['english']<=3] = 0
	data['english'][(data['english']>3)&(data['english']<=6)] = 1
	data['english'][(data['english']>6)&(data['english']<=36)] = 2
	data['science'][data['science']>36] = 3
	# bin number of science credits into low (0), medium (1), high (2) and very high (3)
	data['science'][data['science']<=6] = 0
	data['science'][(data['science']>6)&(data['science']<=36)] = 1
	data['science'][(data['science']>36)&(data['science']<=80)] = 2
	data['science'][data['science']>80] = 3
	# bin number of writing credits into low(0), medium(1) and high(2)
	data['writing'][data['writing']<=10] = 0
	data['writing'][(data['writing']>10)&(data['writing']<=20)] = 1
	data['writing'][data['writing']>20] = 2
	# bin gpa into low(0), medium(1) and high(2)
	data['gpa'][data['gpa']<=3.2] = 0
	data['gpa'][(data['gpa']>3.2)&(data['gpa']<=3.5)] = 1
	data['gpa'][data['gpa']>3.5] = 2
	return(data)

def discrete_conditional(data, labels, feature, class_):
	"""
	Takes in an array of feature values along with the ground truth labels and computes the 
	counts for a feature with discrete values conditioned on the specified class. Returns a
	dictionary whose keys are the discrete feature values and values are the respective
	counts.
	"""
	# find indices of projects that were labelled class_
	indices = np.where(labels==class_)
	# obtain discrete feature values and conditional counts
	feature_values, counts = np.unique(data[feature][indices], return_counts=True)
	# return dictionary with feature values as the keys and counts as the value
	return(dict(zip(feature_values, counts)))

def continuous_conditional(data, labels, feature, class_, distribution):
	"""
	Takes in an array of feature values along with the ground truth labels and returns the 
	conditional probability P(feature|class) given the specified class. Assumes that the
	feature values are continuous. The argument "distribution" is a string that specifies
	the probability distribution to be used. 
	"""
	# find indices of projects that were labelled class_
	indices = np.where(labels==class_)
	# if using a normal distribution
	if distribution=='normal':
		# compute mean and standard deviation of feature value
		mean, sd = np.mean(data[feature][indices]), np.std(data[feature][indices])
		# return function that computes the height of the normal distribution given
		# these parameters
		def normal_prob(x):
			return stats.norm.pdf(x, loc=mean, scale=sd)
		return(normal_prob)
	# if using a negative binomial distribution
	elif distribution=="negativeBinomial":
		# compute parameters (size and probability) for the negative binomial distribution
		# here, I hard-coded the parameter estimates since only the feature "writing" will
		# use this distribution
		if class_==1:
			size, prob = 4.765, 0.2209
		else:
			size, prob = 4.865, 0.2526
		# return function that computes the height of the negative binomial distribution
		# given these parameter estimates
		def nbinom_prob(x):
			return stats.nbinom.pmf(x, n=size, p=prob)
		return(nbinom_prob)

def compute_distributions(data, labels, discrete_features, continuous_features):
	"""
	Obtains all the counts for discrete features and probability distributions for
	continuous features. Returns them as a list of dictionaries in the discrete case,
	or a list of probability density functions in the continuous case.
	"""
	# obtain counts for discrete features given needs work
	discrete_positive = [discrete_conditional(data, labels, feature, class_=1) 
								for feature in discrete_features]
	# obtain counts for discrete features given not needs work
	discrete_negative = [discrete_conditional(data, labels, feature, class_=-1) 
								for feature in discrete_features]
	# obtain probability distribution for continuous features given needs work
	continuous_positive = []
	for feature in continuous_features:
		if feature != 'writing':
			continuous_positive.append(
				continuous_conditional(data, labels, feature, class_=1, distribution='normal')
				)
		else:
			continuous_positive.append(
				continuous_conditional(data, labels, feature, class_=1, distribution='negativeBinomial')
				)	
	# obtain probability distribution for continuous features given not needs work
	continuous_negative = []
	for feature in continuous_features:
		if feature != 'writing':
			continuous_negative.append(
				continuous_conditional(data, labels, feature, class_=-1, distribution='normal')
				)
		else:
			continuous_negative.append(
				continuous_conditional(data, labels, feature, class_=-1, distribution='negativeBinomial')
				)	
	return (discrete_positive, discrete_negative, continuous_positive, continuous_negative)

def classify(data, labels, discrete_features, continuous_features, alpha):
	"""
	Performs leave-one-out cross-validation on the data by picking one instance to classify, 
	and using all the other instances as training data. Returns a vector of predicted labels.
	"""
	# compute total number of projects labelled "needs work"
	num_needsWork = num_positive(labels)
	# obtain all the necessary probability distributions
	dis_pos, dis_neg, cont_pos, cont_neg = compute_distributions(data, labels, 
												discrete_features, continuous_features)
	# initialize numpy array to store predicted labels
	predicted_labels = np.zeros(len(labels), dtype=int)

	# iterate over rows in data set
	for i in range(len(labels)):
		# compute number of training instances labelled positive (needs work)
		if labels[i]==1: num_pos = num_needsWork-1
		else: num_pos = num_needsWork

		# compute number of training instances labelled negative (not needs work)
		num_neg = len(labels)-1-num_pos

		# compute unconditional probability of positive class (needs work)
		P_needsWork = num_pos/(len(labels)-1)

		# initialize scores for positive class (needs work) and negative class (not needs work)
		score_pos, score_neg = math.log(P_needsWork), math.log(1-P_needsWork)
		
		# for each discrete feature
		for j in range(len(discrete_features)):
			# access feature value
			feature_value = data[discrete_features[j]][i]
			# obtain conditional count for that feature value given positive
			count_pos = dis_pos[j].get(feature_value, 0) 
			# if test case had true label positive, subtract 1 from this count
			if labels[i]==1: count_pos -= 1 
			# find conditional probability, applying Laplace smoothing
			prob_pos = (count_pos + alpha)/(num_pos + alpha*len(dis_pos[j]))
			# update score for positive class
			score_pos += math.log(prob_pos)
			# obtain conditional count for feature value given negative
			count_neg = dis_neg[j].get(feature_value, 0)
			# if test case had true label negative, subtract 1 from this count
			if labels[i]==-1: count_neg -= 1 
			# find conditional probability, applying Laplace smoothing
			prob_neg = (count_neg + alpha)/(num_neg + alpha*len(dis_neg[j]))
			# update score for negative class
			score_neg += math.log(prob_neg)

		# for each continuous feature
		for k in range(len(continuous_features)):
			# access feature value
			feature_value = data[continuous_features[k]][i]
			# obtain conditional probability for feature value given positive
			prob_pos = cont_pos[k](feature_value)
			# update score for positive class
			score_pos += math.log(prob_pos)
			# obtain conditional probability for feature value given negative
			prob_neg = cont_neg[k](feature_value)
			# update score for negative class
			score_neg += math.log(prob_neg)
		# classify current row
		if score_pos > score_neg: predicted_labels[i] = 1
		else: predicted_labels[i] = -1

	# return vector of predicted labels
	return(predicted_labels)
			
def confusion_matrix(true_labels, pred_labels):
	"""
	Returns a confusion matrix based on the true and predicted labels.
	"""
	# store the unique classes in this problem
	classes = np.unique(true_labels)
	# initialize empty confusion matrix
	con_matrix = np.zeros((np.size(classes), np.size(classes)), dtype=int)
	# for each instance, compare the predicted and true labels
	for n in range(np.size(true_labels)):
		# if the labels match up
		if true_labels[n] == pred_labels[n]:
			# find the index of the class
			i = np.where(classes==true_labels[n])
			# and increment the count of the corresponding diagonal entry in the confusion matrix
			con_matrix[i, i] += 1 
		# if the labels don't match up
		else:
			# find the index of the actual and predicted class
			i_actual = np.where(classes==true_labels[n])
			i_pred = np.where(classes==pred_labels[n])
			# and increment the count of the corresponding off-diagonal entry
			con_matrix[i_actual, i_pred] += 1
	return con_matrix

def compute_accuracy(con_matrix):
	"""
	Takes in a confusion matrix and returns the accuracy of the classification.
	"""
	return np.trace(con_matrix)/np.sum(con_matrix)

def classified_wrongly(true_labels, pred_labels, IDs):
	"""
	Returns the IDs of the projects which were classified wrongly.
	"""
	# obtain positions of wrong classifications
	wrong_pos = np.invert(true_labels==pred_labels)
	# return IDs of wrongly classified projects
	return(IDs[wrong_pos])

def false_negatives(true_labels, pred_labels, IDs):
	"""
	Returns the IDs of the projects which were false negatives.
	"""
	# obtain positions of false negatives
	false_neg = (true_labels==1) & (pred_labels==-1)
	return(IDs[false_neg])

def main():
	# obtain command line argument for smoothing factor alpha
	parser = argparse.ArgumentParser()
	parser.add_argument('alpha', type=float, 
		help='Enter smoothing factor alpha')
	flags, unparsed = parser.parse_known_args()

	# define names for features
	names = ['mn', 'int', 'birthyear', 'verbal', 'math', 'gpa', 'essays',
			 'abroad', 'AP', 'CS', 'english', 'science', 'writing']
	# obtain feature values, project IDs and ground truth labels
	data, IDs, labels = read_data(col_names=names)
	# discretize the data for number of abroad, AP, CS, english and science credits
	data = discretize(data)

	#######################################################################################
	# Edit these two variables to test out variations of the classifier.
	#######################################################################################
	# specify discrete and continuous features
	discrete_features = ['mn', 'int', 'birthyear', 'essays', 'abroad', 'AP', 
						 'CS', 'english', 'science']
	continuous_features = ['verbal', 'math', 'writing', 'gpa']
	#######################################################################################
	
	# obtain predicted labels
	predicted_labels = classify(data, labels, discrete_features, continuous_features, flags.alpha)
	# obtain confusion matrix
	con_mat = confusion_matrix(labels, predicted_labels)
	print('Confusion matrix:')
	print('      1   -1   \n   -----------\n 1 | %d | %d | \n-1 | %d | %d |\n   -----------' 
					%(con_mat[1,1], con_mat[1,0], con_mat[0,1], con_mat[0,0]))
	# compute accuracy
	print('Accuracy is: %.4f' % compute_accuracy(con_mat))
	# obtain IDs of projects that were classified wrongly
	print('IDs of projects that were classified wrongly:\n',
				classified_wrongly(labels, predicted_labels, IDs))
	# obtain IDs of projects that were false negatives
	print('IDs of projects that were false negatives:\n',
				false_negatives(labels, predicted_labels, IDs))

if __name__ == '__main__':
	main()