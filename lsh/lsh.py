# lsh.py (question 2)
# Implements both locality sensitive hashing (LSH) and a brute-force nearest neighbor
# approach on a dataset of 39,861 e-mails.
#---------------------------------------------------------------------------------------
# Il Shan Ng
# April 7, 2017

import numpy as np
import heapq as hq
from collections import defaultdict
import sys
import random
import math
import time
import argparse

#####################################################################################
# specify path to and name of data file
file_name = 'docword.enron.txt'
#####################################################################################

def character_matrix(D):
	"""
	Reads in the first D documents of the data and creates a character matrix,
	implemented as a set of words for each document.
	"""
	# start timer
	start_time = time.time()
	# initialize list of D empty sets
	chr_mat = [set() for _ in range(D)]
	# open file containing data
	with open(file_name) as file:
		# skip first three lines
		for _ in range(3):
			next(file)
		while True:
			# read and split line on white space, casting fragments into integers
			line = [int(i) for i in str.split(file.readline())]
			# if reached end of data set, return list of sets
			if not line:
				return chr_mat
			# if document ID is within first D documents
			if line[0] <= D:
				# add word ID to the set for the current document
				chr_mat[line[0]-1].add(line[1])
			# otherwise, return list of sets
			else:
				return (chr_mat, time.time()-start_time)

def jaccard(doc1, doc2):
	"""
	Takes in two documents, in the form of two sets, and returns the Jaccard
	similarity between them.
	"""
	return (len(doc1 & doc2)/len(doc1 | doc2))

def signature_matrix(chr_mat, N, D):
	"""
	Takes in a character matrix and creates a signature matrix with N rows using
	the min-hash algorithm.
	"""
	# start timer
	start_time = time.time()
	# set random seed for debugging purposes
	random.seed(1)
	# compute number of words in the character matrix
	words = set.union(*chr_mat)
	W = len(words)
	# initialize empty list to store N randomly generated hash functions
	hash_functions = []
	# for each hash function
	for n in range(N):
		# generate random b between 0 and W-1
		b = random.randint(0, W-1)
		# generate random a between 0 and W-1, which is coprime with W
		gcd = 2
		while gcd != 1:
			a = random.randint(0, W-1)
			gcd = math.gcd(a, W)
		# add (a,b) tuple to list of hash functions
		hash_functions.append((a,b))
	# initialize signature matrix with inf in all entries
	sig_mat = np.empty(shape=(N, len(chr_mat)), dtype=int)
	sig_mat.fill(99999999)
	# for each row of the character matrix
	for w in range(W):
		# compute hash values for row number
		hash_values = np.array([(coeff[0]*w + coeff[1])%W for coeff in hash_functions])
		# obtain word ID corresponding to row
		wordID = words.pop()
		# for each document
		for d in range(D):
			# if the word ID is in the document
			if wordID in chr_mat[d]:
				# compare the existing column for the document to the vector of
				# computed hash values, and choose the element-wise minimum
				sig_mat[:,d] = np.minimum(sig_mat[:,d], hash_values)
	# return signature matrix along with time taken
	return (sig_mat, time.time() - start_time)

def estimated_jaccard(doc1, doc2):
	"""
	Takes in two documents, in the form of two columns of the signature matrix, and 
	returns the estimated Jaccard similarity between them.
	"""
	return sum(doc1 == doc2)/len(doc1)

def knn_brute_force(chr_mat, K, D):
	"""
	Takes in a character matrix, in the form of a list of sets, and finds the K closest
	neighbors to each document, measured by Jaccard similarity. Computes the average 
	Jaccard similarities for those K neighbors with respect to each document, and returns
	the average over all documents.
	"""
	# start timer
	start_time = time.time()
	# initialize numpy array to store average Jaccard similarities
	avg_jaccard_sim = np.empty(D, dtype=float)
	# for each document
	for d in range(D):
		# initialize empty priority queue to store K closest neighbors
		queue = []
		# loop over all other documents
		for j in list(range(d)) + list(range(d+1, D)):
			# compute the Jaccard similarity between the current pair and add to queue
			hq.heappush(queue, (jaccard(chr_mat[d], chr_mat[j]), j))
			# if more than K items in queue
			if len(queue) > K:
				# remove document with lowest Jaccard similarity
				hq.heappop(queue)
		# initialize numpy array to store Jaccard similarities
		jaccard_sim = np.empty(K, dtype=float)
		# iterate through all K neighbors in queue
		for k in range(K):
			# obtain Jaccard similarity and add it to the array
			jaccard_sim[k] = hq.heappop(queue)[0]
		# compute the average Jaccard similarity and add to array at top
		avg_jaccard_sim[d] = np.mean(jaccard_sim)
		# print progress to screen
		if D >= 10:
			if (d+1) % (D//10) == 0:
				print('Average Jaccard similarity for document ', d+1, ': ', 
					'%.4f' % avg_jaccard_sim[d], sep='')
				sys.stdout.flush()
	# return average of averages along with time taken
	return(np.mean(avg_jaccard_sim), time.time() - start_time)

def knn_lsh(sig_mat, K, D, R):
	"""
	Takes in a signature matrix, in the form of a numpy 2D array, and finds the K
	closest neighbors to each document using banding and locality sensitive hashing. 
	Computes the average estimated Jaccard similarities for those K neighbors with 
	respect to each document, and returns the average over all documents.
	"""
	# start timer
	start_time = time.time()
	# set random seed for debugging purposes
	random.seed(1)
	# determine number of bands (drop remainder rows that do not form a band)
	B = sig_mat.shape[0]//R
	# initialize empty list of dictionaries, one for each band
	dict_list = [defaultdict(set) for b in range(B)]
	# for each band
	for b in range(B):
		# iterate over all documents
		for d in range(D):
			# use the vector of R integers, converted to a tuple, as key
			key = tuple(sig_mat[b*R:(b+1)*R, d])
			# add key-documentID pair to corresponding dictionary
			dict_list[b][key].add(d)
	# initialize numpy array to store average estimated Jaccard similarities
	avg_jaccard_sim = np.empty(D, dtype=float)
	# initialize count for number of documents whose neighbors were padded at random
	num_padded = 0
	# for each document
	for d in range(D):
		# initialize empty set of candidates
		candidates = set()
		# iterate over all bands
		for b in range(B):
			# compute the key of the document for the current dictionary
			key = tuple(sig_mat[b*R:(b+1)*R, d])
			# add all document IDs in the corresponding bucket to set of candidates
			candidates = candidates.union(dict_list[b][key])
		# remove ID of current document from set of candidates
		candidates.remove(d)
		# initialize numpy array to store Jaccard similarities
		jaccard_sim = np.empty(K, dtype=float)
		# if number of candidates <= K for current document
		if len(candidates) < K:
			# sample randomly from the other non-candidate documents
			docs = random.sample((set(range(d))|set(range(d+1, D)))-candidates, K-len(candidates))
			# pad set of candidates with these document IDs and recast as list
			candidates = list(candidates.union(set(docs)))
			num_padded += 1
			# then iterate through all K neighbors
			for k in range(K):
				# and compute the estimated Jaccard similarity and add it to array
				neighbor = candidates[k]
				jaccard_sim[k] = estimated_jaccard(sig_mat[:,d], sig_mat[:,neighbor])
		# otherwise
		else:
			# initialize empty priority queue to store K closest neighbors
			queue = []
			# loop over all candidates
			for j in candidates:
				# compute the Jaccard similarity between the current pair and enqueue
				hq.heappush(queue, (estimated_jaccard(sig_mat[:,d], sig_mat[:,j]), j))
				# if more than K items in queue
				if len(queue) > K:
					# remove document with lowest Jaccard similarity
					hq.heappop(queue)
			# iterate through all K neighbors
			for k in range(K):
				# obtain Jaccard similarity and add it to the array
				jaccard_sim[k] = hq.heappop(queue)[0]
		# compute the average Jaccard similarity and add to array at top	
		avg_jaccard_sim[d] = np.mean(jaccard_sim)
	 	# print progress to screen
		if D >= 10:
			if (d+1) % (D//10) == 0:
		 		print('Average estimated Jaccard similarity for document ', d+1, ': ', 
		 			'%.4f' % avg_jaccard_sim[d], sep='')
		 		sys.stdout.flush()
	# return average of averages along with time taken and number of documents whose
	# neighbors were chosen randomly
	return(np.mean(avg_jaccard_sim), time.time() - start_time, num_padded)

def main():
	# # obtain command line arguments
	# parser = argparse.ArgumentParser()
	# parser.add_argument('--D', type=int, 
	# 	help='Integer number of documents to read into program')
	# parser.add_argument('--doc1', type=int, default=1,
	# 	help='Document ID of first document')
	# parser.add_argument('--doc2', type=int, default=2,
	# 	help='Document ID of second document')
	# parser.add_argument('--N', type=int, default=1000,
	# 	help='Number of rows of signature matrix')
	# parser.add_argument('--K', type=int,
	# 	help='Number of closest neighbors for the knn algorithm')
	# parser.add_argument('--R', type=int,
	# 	help='Number of rows in each band')
	# flags, unparsed = parser.parse_known_args()

	# get user input for number of documents D
	D = int(input('Input number of documents to read: '))
	# create character matrix stored as a list of sets
	chr_mat, time_taken0 = character_matrix(D)
	print('Created character matrix (%.4f seconds).' % float(time_taken0))
	# get user input for two documents whose Jaccard similarity is to be computed
	doc1 = int(input('Input first document ID: '))
	doc2 = int(input('Input second document ID: '))
	# compute Jaccard similarity between requested documents
	print('Actual Jaccard similarity for documents %s and %s: %.4f' % 
		(doc1, doc2, jaccard(chr_mat[doc1-1], chr_mat[doc2-1])))
	# get user input for number of rows of signature matrix N
	N = int(input('Input number of rows for the signature matrix: '))
	# create signature matrix stored as numpy 2D array
	sig_mat, time_taken1 = signature_matrix(chr_mat, N, D)
	print('Created signature matrix (%.4f seconds)' % float(time_taken1))
	# compute estimated Jaccard similarity between requested documents
	print('Estimated Jaccard similarity for documents %s and %s: %.4f' % 
		(doc1, doc2, estimated_jaccard(sig_mat[:,doc1-1], 
							sig_mat[:,doc2-1])))
	print('------------------------------------------------------------------------------------------------')
	# get user input for number of nearest neighbors K
	K = int(input('Input number of nearest neighbors: '))
	# compute average Jaccard similarity in data set using brute force
	avg_acc1, time_taken2 = knn_brute_force(chr_mat, K=K, D=D)
	print('Average Jaccard similarity across all documents using brute force: %.4f (%.4f seconds)' \
		% (avg_acc1, float(time_taken2)))
	# get user input for number of rows per band R
	R = int(input('Input number of rows in each band: '))
	# compute average estimated Jaccard similarity using LSH
	avg_acc2, time_taken3, num_random = knn_lsh(sig_mat, K=K, D=D, R=R)
	print('Average estimated Jaccard similarity across all documents using banding and LSH: %.4f (%.4f seconds)' \
		% (avg_acc2, float(time_taken3)))
	print('Number of documents whose neighbors were chosen at random:', num_random)

if __name__ == '__main__':
	main()