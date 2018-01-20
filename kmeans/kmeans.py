# kmeans.py
# Implements the k-means algorithm to cluster a subset of the MNIST database of
# handwritten digits. Experiments using different methods of centroid initialization.
#---------------------------------------------------------------------------------------
# Il Shan Ng
# May 6, 2017

import numpy as np
import scipy.spatial.distance as sp
import heapq as hq
import argparse

def read_data():
	"""
	Read in the data set to be clustered, along with the ground truth labels.
	Returns them stored in numpy arrays.
	"""
	data = np.genfromtxt('number_data.txt', dtype=int, delimiter=',', max_rows=10000)
	labels = np.genfromtxt('number_labels.txt', dtype=int, delimiter=',')
	# subtract 1 from each label to make them go from 0 to 9
	labels = labels-1
	print('Successfully read in data and labels.')
	return(data, labels)

def initialize_centroids(data, labels, N, K, method):
	"""
	Initializes K centroids for the clustering procedure, according to the user-
	specified method (random, k-means++, or cheating).
	"""
	# for debugging purposes
	# np.random.seed(1)
	# if random initialization chosen
	if method=='random':
		# generate K 1 by 784 vectors with entries randomly chosen between 0 and 255
		centroids = np.random.randint(255, size=(K, 784))
	# if other method of initialization chosen, use k-means++
	elif method=='other':
		# intialize empty numpy array to store K centroids
		centroids = np.empty(shape=(0,784), dtype=int)
		# choose first centroid uniformly at random among data points
		centroids = np.append(centroids, 
			np.array([data[np.random.randint(np.shape(data)[0])]]), axis=0)
		# loop until K centroids have been chosen
		while np.shape(centroids)[0]<K:
			# compute square of distance between each point and nearest centroid
			sq_distances = np.square(np.min(sp.cdist(data, centroids, 'euclidean'), axis=1))
			# create probability distribution by weighting each point proportionally to
			# the square of the computed distance
			prob = sq_distances/np.sum(sq_distances)
			# sample point according to this probability distribution
			sampled_point = np.random.choice(np.arange(N), p=prob)
			# use this sampled point as the next centroid
			centroids = np.append(centroids, np.array([data[sampled_point]]), axis=0)
	# if cheating
	elif method=='cheating':
		# initialize empty numpy array to store 10 centroids
		centroids = np.empty(shape=(0,784), dtype=int)
		# for each unique true label
		for k in range(10):
			# compute the mean of all data points with that label and use it as a centroid
			centroids = np.append(centroids, 
				np.array([np.mean(data[labels==k], axis=0)]), axis=0)
	print('Successfully initialized centroids.')
	return(centroids)

def compute_SSE(data, N, K, centroids, assignments):
	"""
	Computes the sum of squared error given the data set with the new assignment and 
	the current position of the centroids. Also returns the top K points that are
	furthest away from their assigned centroids.
	"""
	# initalize variable to store the computed SSE
	SSE = 0
	# initialize priority queue to store the furthest K points
	queue = []
	# iterate over all data points
	for n in range(N):
		# obtain assigned centroid
		centroid = assignments[n]
		# compute the square of Euclidean distance between data point and assigned centroid
		distance = sp.euclidean(data[n], centroids[centroid])**2
		# increment the SSE
		SSE += distance
		# push data point and distance into queue
		hq.heappush(queue, (distance, n))
		# if queue has more than K points, pop point that is least far away
		if len(queue)>K: hq.heappop(queue)
	# generate list of top K points that are furthest away from their assigned centroids
	furthest_points = [hq.heappop(queue)[1] for k in range(K)]
	# return computed SSE and list of furthest points
	return(SSE, furthest_points[::-1])

def cluster_data(data, N, K, centroids):
	"""
	Implements the main loop of the k-means algorithm. Assigns each data point to its
	closest centroid and recomputes centroids after new assignments, repeating until 
	convergence. Returns a vector storing the final assignment to each data point, as 
	well as the centroids of the final clusters.
	"""
	# initialize count for number of iterations and arbitrarily large SSE
	i = 1; prev_SSE = float('inf')
	# initialize list to store and write SSEs to text file
	SSE_list = []
	# iterate until convergence
	while True:
		# assign each data point to the closest centroid
		assignments = np.argmin(sp.cdist(data, centroids, 'euclidean'), axis=1)
		# compute the SSE and obtain the top K points that are furthest away from their
		# assigned clusters
		SSE, furthest_points = compute_SSE(data, N, K, centroids, assignments)
		print('SSE for iteration %i: %i' % (i, SSE))
		SSE_list.append(SSE)
		# iterate over all centroids
		for k in range(K):
			# if centroid has no data points assigned to it (i.e. cluster is empty)
			if np.sum(assignments==k)==0:
				# replace the centroid with furthest point
				centroids[k] = data[furthest_points.pop(0)]
			# otherwise
			else:
				# recompute centroid by averaging over all data points that were assigned 
				# to this centroid
				centroids[k] = np.mean(data[assignments==k], axis=0)
		# if SSE is no longer changing significantly, exit while loop
		if (SSE/prev_SSE)>0.9999: break
		# otherwise, increment iteration number and store SSE for previous iteration
		i += 1; prev_SSE = SSE
	# write SSEs to text file
	np.savetxt('SSE.txt', SSE_list, delimiter=',')
	return(assignments, centroids)

def main():
	# obtain command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('K', type=int, 
		help='The number of clusters K to use.')
	parser.add_argument('init', type=str,
		help='How initialization of the centroids should be done.')
	flags, unparsed = parser.parse_known_args()

	# check for invalid values of K
	if flags.K<1: print('K must be a positive integer.'); return
	# if "cheating" selected, overwrite user-specified value of K to 10
	if flags.init=='cheating': flags.K=10
	# read in data set and ground truth labels
	data, labels = read_data()
	# obtain initial centroids according to user specified method
	centroids = initialize_centroids(data, labels, np.shape(data)[0], flags.K, flags.init)
	# cluster data set using the k-means algorithm
	assignments, final_centroids = cluster_data(data, np.shape(data)[0], flags.K, centroids)
	# write final centroids to text file for further analysis
	np.savetxt('centroids.txt', final_centroids, delimiter=',')

if __name__ == '__main__':
	main()