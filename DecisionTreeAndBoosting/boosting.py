import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		h = np.zeros(len(features))
		for clf, beta in zip(self.clfs_picked, self.betas):
		    h += beta * np.array(clf.predict(features))
		h = [-1 if hx <= 0 else 1 for hx in h]
		return h


class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################      
        
      # Step 1
		N = len(labels)
		D_t = np.full(N, 1/N)

		# Step 2
		for t in range(self.T): 
			eps_t = np.inf
			for clf in self.clfs:
			# step 3:
				h_pred = clf.predict(features)
				# step 4:
				eps = np.sum(D_t * (np.array(labels) != np.array(h_pred)))
				if eps < eps_t:
					h_t = clf
					eps_t = eps
					h_pred_t = h_pred
		                        
			self.clfs_picked.append(h_t)
		      
			# step 5:
			beta_t = 1/2 * np.log((1-eps_t)/eps_t)   
			self.betas.append(beta_t)

			# step 6:
			for n in range(N):
				if labels[n] == h_pred_t[n]:
					D_t[n] *= np.exp(-beta_t)
				else:
					D_t[n] *= np.exp(beta_t)

			# step 7:
			D_t /= np.sum(D_t)
		 
		
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	