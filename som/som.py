# som.py
# Implements a self-organizing map to cluster countries based on world development
# indicators.
#---------------------------------------------------------------------------------------
# Yuan Shen Li and Il Shan Ng
# May 20, 2017

import numpy as np
import scipy.spatial.distance as sp
import argparse
import collections as cl
import pickle


class SOM():
    """
    Self-organizing map.
    """
    def __init__(self, J, K, inputs, init, dist_func, neighborhood_func, eps):
        """
	Store class variables.
	"""
        self.inputs = inputs
        self.J = J
        self.K = K
        self.init = init
        self.dist_func = dist_func
        self.neighborhood_func = neighborhood_func
        self.M = []
        self.L = cl.defaultdict(set)
        self.eps = eps

        # if Gaussian neighborhood function
        self.sigma_init = max(J,K)
        self.sigma = self.sigma_init
        self.sigma_decayfactor = 30

    def initialize(self):
        """
	Takes in six parameters
	    - the initialization method 'init'
	    - input data
	and sets M to a (J*K x F) array storing the initialized models.
	"""
        F = len(self.inputs[0])
        min_val = np.min(self.inputs)
        max_val = np.max(self.inputs)
        
        np.random.seed(1)
        if self.init=='random':
            # create 3D array storing initial models
            self.M = np.random.uniform(min_val, max_val, size=(self.J*self.K, F))
            self.M = np.array(self.M)

		    
    def competitive_step(self):
        """
        Implements the competitive step of SOM training. Takes in 
            - the input data set
        Sets dictionary L of list of "won" data points for each neuron
        """
        # create a distance matrix between inputs and models
        distance_matrix = sp.cdist(self.inputs, self.M, self.dist_func)

        # for each input, find the index of the winner model
        winner_list = np.nanargmin(distance_matrix, axis=1)

        # for each data point, append to L the entry:
        #     key = i, index of the winning node
        #     value = x, index of "won" data point
        for x in range(len(winner_list)):
            i = winner_list[x]
            self.L[i].add(x)

    def adaptive_step(self):
        """
        Implements the adaptive step of SOM training. Takes in
        Updates model M and returns error (change in model)
        """
        error = 0
        # compute topological distance matrix
        indices = [[i] for i in range(len(self.M))]
        if self.neighborhood_func == "gaussian":
            topological_distances = sp.cdist(indices,indices,self.gaussian)
        else:
            print("Error: unknown neighborhood function.")
            return

       # find length of L_j, mean(L_j)
        numWonPoints = [len(self.L[i]) for i in range(len(self.M))]
        meanpt = [np.array([0 for j in range(len(self.M[0]))]) for i in range(len(self.M))]
        for i in range(len(self.M)):
            if not numWonPoints[i] == 0:
                points = [self.inputs[indx] for indx in self.L[i]]
                meanpt[i] = np.mean(points,axis=0)
        
        for i in range(len(self.M)):
            m_old = self.M[i].copy()
            num = 0
            den = 0
            for j in range(len(self.M)):
                num += topological_distances[i][j] * numWonPoints[j] * meanpt[j]
                den += topological_distances[i][j] * numWonPoints[j]

            self.M[i] = num/den

            # update error (for convergence check)
            error += np.linalg.norm(m_old-self.M[i])
            
        return error

    def train(self):
        """
        Trains the SOM with input data
        Returns models M after training
        """
        iterations = 0
        error = float('inf')
        while error >= self.eps:
            iterations += 1
            print("Iteration", iterations," with error", error)
            print(np.sum(self.M))
            self.competitive_step()
            error = self.adaptive_step()
            # reduce sigma over iterations
            if self.neighborhood_func == "gaussian":
                self.sigma = self.sigma_init * np.exp(-1 * iterations/self.sigma_decayfactor)

        return self.M

    def gaussian(self, xl,yl):
        """
        Gaussian neigborhood function 
            - x,y, indices of models X and Y
            - sigma, spread of the gaussian kernel
            - K, height of the SOM
        Returns the computed topological distance
        """
        x = xl[0]; y = yl[0]
        x_j = int(x/self.K)
        x_k = x % self.K
        y_j = int(y/self.K)
        y_k = y % self.K

        sqdist = np.power(x_j-y_j,2) + np.power(x_k-y_k,2)

        return np.exp(-sqdist/(2*np.power(self.sigma,2)))

def read_data(filepath):
    """
    Read in the data set to be clustered, along with the ground truth labels.
    Returns them stored in numpy arrays.
    """
    data = np.genfromtxt(filepath, dtype=int, delimiter=',', max_rows=1000)
    return (None, data)

    
def normalize_features(indicators):
    """
    Normalizes the data for each indicator over all countries.
    """
    # subtract column-wise mean and divide by column-wise standard error
    return((indicators-np.nanmean(indicators, axis=0))/np.nanstd(indicators, axis=0))


def main():
    # obtain command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('J', type=int, 
                        help='Integer number representing the width of the map.')
    parser.add_argument('K', type=int, 
                        help='Integer number representing the height of the map.')
    parser.add_argument('init', type=str,
                        help='String representing the initialization method to use')
    parser.add_argument('dist_func', type=str,
                        help='String representing the distance function to use')
    parser.add_argument('neighborhood_func', type=str,
                        help='String representing the neighborhood function to use')
    parser.add_argument('epsilon', type=float,
                        help='Float representing convergence threshold')
    flags, unparsed = parser.parse_known_args()

    # read in dataset
    countries, indicators = read_data("number_data.txt")
    # normalize feature values
    # inputs = normalize_features(indicators)
    inputs = indicators
    # initialize SOM 
    som = SOM(J=flags.J, K=flags.K, inputs=inputs, init=flags.init, dist_func=flags.dist_func,
              neighborhood_func=flags.neighborhood_func, eps=flags.epsilon) #NOTE NEED TO ADD EPS option
    som.initialize()

    # train SOM
    final_model = som.train()


if __name__ == '__main__':
    main()