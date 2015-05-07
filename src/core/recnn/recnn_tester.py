import numpy as np
import scipy as sp
from scipy.misc import logsumexp
from numpy import linalg as LA

import core.datasets.entity

class RecNNTester:
	"""docstring for RecNNTester"""
	def __init__(self, word_vec_dim, vocabulary_size, classify_category_num):
		self.word_vec_dim          = word_vec_dim
		self.vocabulary_size       = vocabulary_size
		self.classify_category_num = classify_category_num		
		
		self.activate_func 	     = lambda x : 1 / (1 + np.exp(-x))
		self.activate_prime_func = lambda fx: fx * (1 - fx)
		self.classify_func       = lambda x : np.exp(x - logsumexp(x))

		self.structure_lose_kappa  = 0.05

	def recursive_test(self, current_node):
		self.test_tree.all_node_num += 1
		if current_node.is_leaf():
			current_node.feature = self.word_dict[:, [current_node.feature_entry_id] ]
			self.test_tree.leaf_node_num += 1
		else:
			self.recursive_test(current_node.left)
			self.recursive_test(current_node.right)
			# calculate the parent's feature
			kis_feature = np.vstack( (current_node.left.feature, current_node.right.feature, 1) )
			current_node.feature = self.activate_func( self.weights_forward.dot(kis_feature) )		
		
		# calculate classification result
		classify_input = self.weights_classify.dot(np.vstack((current_node.feature,1)))
		classify_output = self.classify_func(classify_input)
		# check classification correctness
		true_argmax = np.argmax(current_node.category_vector)
		pred_argmax = np.argmax(classify_output)
		
		is_correct_grained = (true_argmax==pred_argmax).astype(int)
		is_correct_binary = ((true_argmax<2 and pred_argmax<2) or (true_argmax>2 and pred_argmax>2) or (true_argmax==2 and pred_argmax==2)).astype(int)

		self.test_tree.all_corr_grained += is_correct_grained
		self.test_tree.all_corr_binary += is_correct_binary

		if current_node.is_leaf():
			self.test_tree.leaf_corr_grained += is_correct_grained
			self.test_tree.leaf_corr_binary += is_correct_binary

		# aggregate information
		if current_node == self.test_tree.root:
			self.root_accuracy_grained += is_correct_grained
			self.leaf_accuracy_grained += self.test_tree.leaf_corr_grained / self.test_tree.leaf_node_num
			self.all_accuracy_grained  += self.test_tree.all_corr_grained / self.test_tree.all_node_num

			self.root_accuracy_binary += is_correct_binary
			self.leaf_accuracy_binary += self.test_tree.leaf_corr_binary / self.test_tree.leaf_node_num
			self.all_accuracy_binary  += self.test_tree.all_corr_binary / self.test_tree.all_node_num

	def setup_test_tree(self):		
		self.test_tree.all_node_num  = 0.0
		self.test_tree.leaf_node_num = 0.0

		self.test_tree.all_corr_binary  = 0.0
		self.test_tree.leaf_corr_binary = 0.0

		self.test_tree.all_corr_grained  = 0.0
		self.test_tree.leaf_corr_grained = 0.0

	def parse_params(self, params):
		# word_dict
		word_dict_end         = self.word_vec_dim * self.vocabulary_size
		self.word_dict        = params[:word_dict_end].reshape(self.word_vec_dim, self.vocabulary_size)
		# weights_score
		weights_score_end     = word_dict_end + self.word_vec_dim
		self.weights_score    = params[word_dict_end:weights_score_end].reshape(1, self.word_vec_dim)		
		# weights_forward
		weights_forward_end   = weights_score_end + self.word_vec_dim * (self.word_vec_dim*2+1)
		self.weights_forward  = params[weights_score_end:weights_forward_end].reshape(self.word_vec_dim, self.word_vec_dim*2+1)
		# weights_classify
		weights_classify_end  = weights_forward_end + self.classify_category_num * (self.word_vec_dim+1)
		self.weights_classify = params[weights_forward_end:weights_classify_end].reshape(self.classify_category_num, self.word_vec_dim+1)		
		
	def init_accuracy_records(self):
		self.root_accuracy_grained = 0.0
		self.leaf_accuracy_grained = 0.0
		self.all_accuracy_grained  = 0.0

		self.root_accuracy_binary = 0.0
		self.leaf_accuracy_binary = 0.0
		self.all_accuracy_binary  = 0.0

	def core_test(self, data, test_type):		
		if test_type == 0:
			self.test_tree = data.tree
			self.setup_test_tree()
			self.recursive_test(self.test_tree.root)			
		else:
			# TO DO 
			pass

	def calculate_accuracy(self, data_num):
		print ('%22s %12f' %('Root Grained Accuracy:', self.root_accuracy_grained/data_num))
		print ('%22s %12f' %('Leaf Grained Accuracy:', self.leaf_accuracy_grained/data_num))
		print ('%21s %12f' %('All Grained Accuracy:', self.all_accuracy_grained/data_num))

		print ('%22s %12f' %('Root Binary Accuracy:', self.root_accuracy_binary/data_num))
		print ('%22s %12f' %('Leaf Binary Accuracy:', self.leaf_accuracy_binary/data_num))
		print ('%22s %12f' %('All Binary Accuracy:', self.all_accuracy_binary/data_num))

	def test(self, params, testing_dataset, test_type = 0):
		self.parse_params(params)
		self.init_accuracy_records()
		
		data_num = len(testing_dataset)
		for data in testing_dataset:
			# feed-forward and calculate cost (loss)
			self.core_test(data, test_type)

		self.calculate_accuracy(data_num)		