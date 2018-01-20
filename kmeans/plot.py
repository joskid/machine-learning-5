# plot.py
# Plots the centroids of the final clusters produced by the kmeans.py program.
#-----------------------------------------------------------------------------------------
# Il Shan Ng
# May 6, 2017

import numpy as np
import matplotlib.pyplot as plt

def display_image(data):
	"""
	Displays a list of 784 gray-scale values as an image.
	"""
	data = np.array(data)
	data = np.reshape(data,(-1,28))
	plt.imshow(data)
	plt.show()
	
def main():
	# read in centroids for final clusters
	centroids = np.genfromtxt("centroids.txt", delimiter=',', dtype=int)
	# plot visualizations for all 10 centroids in a loop
	for row in centroids:
		display_image(row)

if __name__ == '__main__':
	main()