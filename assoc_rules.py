# assoc_rules.py
# Implements the Apriori algorithm to look for frequent itemsets and extract association 
# rules that pass a minimum support, minimum confidence and minimum interest threshold.
# ---------------------------------------------------------------------------------------
# Il Shan Ng
# April 28, 2017

import numpy as np
import collections as cl
import itertools as it
import heapq as hq
import argparse
import time

def get_transactions():
	"""
	Reads in the data file and returns the list of unique items and a list of transactions, 
	each stored as a set.
	"""
	# read in header from data file
	header = np.genfromtxt('BobRoss.txt', max_rows=1, delimiter=',', dtype=str)
	# ignore first four labels to obtain set of unique items
	items = header[4:]
	# read in data as numpy array, skipping header and first four columns
	data = np.genfromtxt('BobRoss.txt', skip_header=1, delimiter=',', 
						usecols=range(4, len(header)), dtype=int).astype(bool)
	# construct list of transactions
	transactions = [set(items[row]) for row in data]
	return(items, transactions)

def frequent_sizeOne(items, transactions, min_sup):
	"""
	Identifies the frequent itemsets of size one based on the specified minimum support.
	Returns the itemsets along with their counts in a dictionary.
	"""
	# initalize empty dictionary
	freq_items_sizeOne = cl.OrderedDict()
	# for each item
	for item in items:
		# initialize count for item
		count = 0
		# iterate over all transactions
		for transaction in transactions:
			# if item in transaction, increment count
			if item in transaction: count += 1
		# if count satisfies minimum support, add item and count to dictionary
		if count >= min_sup: freq_items_sizeOne[frozenset([item])] = count
	return(freq_items_sizeOne)

def generate_candidates(frequent_itemsets, k):
	"""
	Takes in a list of frequent itemsets of size k-1 and use it to generate a list
	of candidate itemsets of size k.
	"""
	# initialize empty list to store candidate itemsets of size k
	candidates_sizeK = []
	# look at all possible pairs of itemsets of size k-1
	for pair in list(it.combinations(frequent_itemsets, 2)):
		# if the pair matches on the first k-2 items, sorted lexicographically
		if sorted(pair[0])[:k-2] == sorted(pair[1])[:k-2]:
			# merge the pairs to form candidate itemset of size k
			candidate = set.union(set(pair[0]), set(pair[1]))
			# create boolean to indicate whether or not candidate is valid
			valid = True
			# for each item in this newly generated candidate
			for item in candidate:
				# create copy of candidate itemset and remove that item
				test_set = candidate.copy()
				test_set.remove(item)
				# then check for membership in list of itemsets of size k-1
				if test_set not in frequent_itemsets:
					# if not in list, make candidate invalid and exit for loop
					valid = False
					break
			# if candidate is valid, add to list of candidates of size k
			if valid: candidates_sizeK.append(frozenset(candidate))
	return(candidates_sizeK)

def apriori(transactions, min_sup, freq_sizeOne):
	"""
	Implements the Apriori algorithm to identify all frequent itemsets given a minimum
	support.
	"""
	# initialize list of dictionaries to store frequent itemsets of size 1 and their supports
	frequent_itemsets = [freq_sizeOne]
	print('Number of frequent itemsets of size 1:', len(freq_sizeOne))
	# initialize variable to store the current size of the itemsets
	k = 2
	# while there are frequent itemsets of size k-1
	while frequent_itemsets[(k-1)-1]:
		# isolate keys of frequent itemsets of size k-1 from dictionary
		frequent_itemsets_keys = [item for item in frequent_itemsets[(k-1)-1].keys()]
		# obtain candidate itemsets of size k and print number
		candidates = set(generate_candidates(frequent_itemsets_keys, k))
		# initialize counter to store support of candidate itemsets
		supports = cl.Counter()
		# for each transaction
		for transaction in transactions:
			# obtain all combinations of k items in that transaction
			combinations = set(frozenset(combination) for combination 
											in it.combinations(transaction, k))
			# find the candidate itemsets of size k that appear in the transaction
			candidates_subset = set.intersection(candidates, combinations)
			# iterate over all candidate itemsets that appear in the transaction
			for candidate in candidates_subset:
				# and increment the count for this candidate itemset 
				supports[candidate] += 1
		# iterate through counter and construct dictionary of frequent itemsets of size k
		freq_itemsets_sizeK = {item: supports[item] for item in supports.keys() 
											if supports[item] >= min_sup}
		# add to list of dictionaries and print out number
		frequent_itemsets.append(freq_itemsets_sizeK)
		print('Number of frequent itemsets of size %i: %i' % (k, len(frequent_itemsets[k-1])))
		# increment k
		k += 1
	return(frequent_itemsets)

def generate_rules(I, J, rules, frequent_itemsets, T, min_conf, min_int):
	"""
	Recursive function that takes in sets I and J and generates association rules that 
	have a minimum level of confidence and interest.
	"""
	# if set I has only 1 item, terminate recursion and return dictionary of rules
	if len(I)==1: return(rules)
	# for each item in I
	for item in I:
		# remove it from I and add it to J
		I_copy = I.copy(); I_copy.remove(item)
		J_copy = J.copy(); J_copy.add(item)
		# obtain the union of I and J
		union = set.union(I_copy, J_copy)
		# obtain support of I union J, I and J
		sup_union = frequent_itemsets[len(union)-1][frozenset(union)]
		sup_I = frequent_itemsets[len(I_copy)-1][frozenset(I_copy)]
		sup_J = frequent_itemsets[len(J_copy)-1][frozenset(J_copy)]
		# compute confidence for rule
		confidence = sup_union/sup_I
		# compute interest for rule
		interest = confidence-sup_J/T		
		# if confidence and interest exceed threshold
		if (confidence >= min_conf) & (abs(interest) >= min_int): 
			# add rule to dictionary
			rules[frozenset(I_copy), frozenset(J_copy)] = (confidence, interest)
		# if confidence exceeds threshold
		if (confidence >= min_conf):
			# recurse and find rules with larger sizes for J
			rules = generate_rules(I_copy, J_copy, rules, frequent_itemsets, 
											T, min_conf, min_int)
	return(rules)

def extract_rules(frequent_itemsets, T, min_conf, min_int):
	"""
	Extracts association rules that have a minimum level of confidence and interest, given
	all frequent itemsets.
	"""
	# initialize empty dictionary to store all association rules
	rules = {}
	# obtain maximum size of frequent itemsets
	max_K = len(frequent_itemsets)-1
	# for all frequent itemsets of each size
	for k in range(1, max_K+1):
		# iterate over each itemset of size k
		for (itemset, support) in frequent_itemsets[k-1].items():
			# call recursive function to generate all rules that pass threshold
			rules = generate_rules(set(itemset), set(), rules, frequent_itemsets, 
									T, min_conf, min_int)
	return(rules)

def top_ten(rules, conf, N=10):
	"""
	Takes in a dictionary of association rules and returns the top 10 rules ranked by 
	confidence or interest.
	"""
	# initialize empty priority queue to store top 10 rules
	queue = []
	# if ranking based on confidence, set position in the value tuple to be 0
	# if ranking based on interest, set position in the value tuple to be 1
	pos = 0 if conf else 1
	# iterate over all rules in dictionary
	for (key, value) in rules.items():
		# add rule and interest to priority queue
		hq.heappush(queue, (value[pos], key))
		# if length of queue exceeds 10, remove rule with lowest interest
		if len(queue) > N: hq.heappop(queue)
	# produce list of top ten from queue
	N = min(len(queue), N)
	top10 = [hq.heappop(queue) for i in range(N)]
	# return list sorted in reverse order
	return(top10[::-1])

def main():
	# obtain command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('min_sup', type=int, 
		help='Minimum level of support needed for a frequent itemset.')
	parser.add_argument('min_conf', type=float,
		help='Minimum level of confidence for a candidate association rule to be accepted.')
	parser.add_argument('min_int', type=float,
		help='Minimum level of interest for a candidate association rule to be accepted.')
	flags, unparsed = parser.parse_known_args()
	# check for non-valid values for min_sup
	if flags.min_sup<1:
		print("Minimum support must be a positive integer.")
		return
	# check for non-valid values for min_conf
	if flags.min_conf<0 or flags.min_conf>1:
		print("Minimum confidence must be a value between 0 and 1.")
		return
	# check for non-valid values for min_int
	if flags.min_int<0 or flags.min_int>1:
		print("Minimum interest must be a value between 0 and 1.")
		return

	# start timer
	start_time = time.time()
	# read in data to obtain list of unique items and list of transactions
	items, transactions = get_transactions()
	# obtain all frequent item sets of size one and print to screen
	freq_items_sizeOne = frequent_sizeOne(items, transactions, flags.min_sup)
	print('----------------------------------------------------------------------------------')
	print('Frequent itemsets of size 1 along with their supports:')
	for item in freq_items_sizeOne:
		print('%s: %i' % (set(item), freq_items_sizeOne[item]))
	print('----------------------------------------------------------------------------------')
	# obtain dictionary of all frequent itemsets using the apriori algorithm
	frequent_itemsets = apriori(transactions, flags.min_sup, dict(freq_items_sizeOne))
	# extract association rules that have minimum level of confidence
	rules = extract_rules(frequent_itemsets, len(transactions), flags.min_conf, 0)
	print('----------------------------------------------------------------------------------')
	print('Number of rules that pass confidence threshold:', len(rules))
	print('----------------------------------------------------------------------------------')
	# obtain top 10 association rules ranked by confidence and print
	top10 = top_ten(rules, conf=True)
	for (confidence, (I, J)) in top10:
		print('Rule: If %s, then %s' % (set(I), set(J)))
		print('Confidence for rule: %.16f' % (confidence))
	# extract association rules that have minimum level of interest
	rules2 = extract_rules(frequent_itemsets, len(transactions), 0, flags.min_int)
	print('----------------------------------------------------------------------------------')
	print('Number of rules that pass interest threshold:', len(rules2))
	print('----------------------------------------------------------------------------------')
	# obtain top 10 association rules ranked by confidence and print
	top10 = top_ten(rules2, conf=False)
	for (interest, (I, J)) in top10:
		print('Rule: If %s, then %s' % (set(I), set(J)))
		print('Interest for rule: %.16f' % (interest))
	print('----------------------------------------------------------------------------------')
	# print time taken for whole program
	print('Time taken for whole program: %.4f seconds' % float(time.time()-start_time))

if __name__ == '__main__':
	main()