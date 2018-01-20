Place the data files "number_data" and "number_labels" in the same directory
as the kmeans.py program. Run the program using Python 3. There are two command
line arguments: a positive integer specifying the number of clusters K to use, 
and a string (random, other or cheating) specifying the initialization method
for the centroids. So, running the command
				
				python3 kmeans.py 10 random

will run the program with K=10 clusters and random initialization of centroids.
Note also the initialization method of "other" uses k-means++.