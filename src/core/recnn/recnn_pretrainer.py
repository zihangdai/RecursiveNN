import numpy as np
import scipy as sp
from scipy.misc import logsumexp
from numpy import linalg as LA

class RecNNPreTrainer:
	def __init__(self, word_vec_dim, vocabulary_size, classify_category_num, regularization_lambda=0.01):
		self.word_vec_dim          = word_vec_dim
		self.vocabulary_size       = vocabulary_size
		self.classify_category_num = classify_category_num
		self.regularization_lambda = regularization_lambda

		self.activate_func       = lambda x : 1 / (1 + np.exp(-x))
		self.activate_prime_func = lambda fx: fx * (1 - fx)
		self.classify_func       = lambda x : np.exp(x) / (1e-6 + np.sum(np.exp(x), axis=0))
		# self.classify_func       = lambda x : np.exp(x - logsumexp(x))

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
		self.weights_classify = params[weights_forward_end:].reshape(self.classify_category_num, self.word_vec_dim+1)

		# print '%18s %10f \t %10f \t %10f' % ('word_dict:', np.max(self.word_dict), np.min(self.word_dict), np.mean(self.word_dict))
		# print '%18s %10f \t %10f \t %10f' % ('weights_score:', np.max(self.weights_score), np.min(self.weights_score), np.mean(self.weights_score))
		# print '%18s %10f \t %10f \t %10f' % ('weights_forward:', np.max(self.weights_forward), np.min(self.weights_forward), np.mean(self.weights_forward))
		# print '%18s %10f \t %10f \t %10f' % ('weights_classify:', np.max(self.weights_classify), np.min(self.weights_classify), np.mean(self.weights_classify))

	def init_loss_and_gradient(self):
		self.total_loss 		   = 0.0
		self.grad_word_dict        = np.zeros_like(self.word_dict)
		self.grad_weights_score    = np.zeros_like(self.weights_score)
		self.grad_weights_forward  = np.zeros_like(self.weights_forward)
		self.grad_weights_classify = np.zeros_like(self.weights_classify)

	def pre_train(self, params, pretrain_data, picked_data_index=[]):
		self.parse_params(params)
		self.init_loss_and_gradient()
		data_num = self.core_training(pretrain_data, picked_data_index)

		flatten_grad = np.hstack( (self.grad_word_dict.flatten(), self.grad_weights_score.flatten(), 
			self.grad_weights_forward.flatten(), self.grad_weights_classify.flatten()) )

		self.total_loss = self.total_loss/data_num + self.regularization_lambda/2 * params.dot(params)
		flatten_grad  = flatten_grad/data_num + self.regularization_lambda * params

		# return self.total_loss[0], flatten_grad
		return self.total_loss, flatten_grad

	def core_training(self, pretrain_data, picked_data_index=[]):
		
		#-------------------- Parse pretrain_data --------------------#		
		category_vector_left = pretrain_data["good_pair_categories"][0]
		category_vector_right = pretrain_data["good_pair_categories"][1]
		parent_category_vector = pretrain_data["good_pair_parents_categories"]
		

		good_pair = pretrain_data["good_pairs"].astype(int)
		bad_pair = pretrain_data["bad_pairs"].astype(int)

		#-------------------- Choose the picked data --------------------#
		if len(picked_data_index) > 0:
			category_vector_left = category_vector_left[:,picked_data_index]
			category_vector_right = category_vector_right[:,picked_data_index]
			parent_category_vector = parent_category_vector[:,picked_data_index]

			good_pair = good_pair[:,picked_data_index]
			bad_pair = bad_pair[:,picked_data_index]

		#-------------------- Feed to the Kids Nodes --------------------#
		pair_num = good_pair.shape[1]

		feature_good = np.ones( (2*self.word_vec_dim+1, pair_num) )
		feature_bad  = np.ones( (2*self.word_vec_dim+1, pair_num) )

		bias_feature = feature_good[-1]

		feature_good[:self.word_vec_dim, :]                    = self.word_dict[:, good_pair[0]]
		feature_good[self.word_vec_dim:self.word_vec_dim*2, :] = self.word_dict[:, good_pair[1]]

		feature_bad[:self.word_vec_dim, :]                    = self.word_dict[:, bad_pair[0]]
		feature_bad[self.word_vec_dim:self.word_vec_dim*2, :] = self.word_dict[:, bad_pair[1]]

		#-------------------- Softmax: Kids Nodes --------------------#
		feature_good_left  = np.vstack( (feature_good[:self.word_vec_dim], bias_feature) )
		feature_good_right = feature_good[self.word_vec_dim:]

		classify_input_left  = self.weights_classify.dot(feature_good_left)
		classify_input_right = self.weights_classify.dot(feature_good_right)

		classify_output_left  = self.classify_func(classify_input_left)
		classify_output_right = self.classify_func(classify_input_right)

		#+1e-10
		cross_entropy_error_left  = -np.sum(np.sum(category_vector_left * np.log(classify_output_left), axis=0))
		cross_entropy_error_right = -np.sum(np.sum(category_vector_right * np.log(classify_output_right), axis=0))
		
		classify_diff_left  = classify_output_left - category_vector_left
		classify_diff_right = classify_output_right - category_vector_right

		classify_delta_left  = self.weights_classify.T.dot(classify_diff_left)[:self.word_vec_dim]
		classify_delta_right = self.weights_classify.T.dot(classify_diff_right)[:self.word_vec_dim]

		# [c x n] \cdot [d x n].T				
		self.grad_weights_classify += classify_diff_left.dot(feature_good_left.T)
		self.grad_weights_classify += classify_diff_right.dot(feature_good_right.T)
		
		#-------------------- Feed to the Parents Nodes --------------------#
		parent_feature_good = self.activate_func( self.weights_forward.dot(feature_good) )
		parent_feature_bad  = self.activate_func( self.weights_forward.dot(feature_bad) )

		parent_score_good = self.weights_score.dot(parent_feature_good)
		parent_score_bad  = self.weights_score.dot(parent_feature_bad)

		# merge cost
		parent_merge_cost = 1 - parent_score_good + parent_score_bad
		negative_index = (parent_merge_cost<0)[0]
		print 'merge_cost_not_ignore: %12f, number of negative: %d' %(parent_merge_cost.sum(), negative_index.astype(int).sum())
		parent_merge_cost[:,negative_index] = 0
		parent_merge_cost = np.sum(parent_merge_cost)
		print 'merge_cost_ignore: %12f' %(parent_merge_cost)
		# Softmax: Parents Nodes
		# c x n
		parent_classify_input  = self.weights_classify.dot(np.vstack((parent_feature_good, bias_feature)))
		parent_classify_output = self.classify_func(parent_classify_input)
		parent_cross_entropy_error = -np.sum(np.sum(parent_category_vector * np.log(parent_classify_output+1e-10), axis=0))

		# Shape: [c x n]
		parent_classify_diff  = parent_classify_output - parent_category_vector

		# [c x d+1]		
		self.grad_weights_classify += parent_classify_diff.dot( np.vstack((parent_feature_good, bias_feature)).T )
		
		# Shape: [d x n] = [c x d].T \cdot [c x n] * 
		parent_classify_delta = self.weights_classify.T.dot(parent_classify_diff)[:self.word_vec_dim] * self.activate_prime_func(parent_feature_good)		

		#-------------------- Back-prop: Parents Nodes --------------------#
		# only punish non-negative scores
		parent_feature_good[:,negative_index] = 0
		parent_feature_bad[:,negative_index]  = 0

		feature_good[:,negative_index] = 0
		feature_bad[:,negative_index]  = 0

		# only parents' nodes are related to weights score
		self.grad_weights_score += -np.sum(parent_feature_good, axis=1) + np.sum(parent_feature_bad, axis=1)

		# Shape: [d x n]
		parent_delta_good = self.weights_score.T * self.activate_prime_func(parent_feature_good)
		parent_delta_bad  = self.weights_score.T * self.activate_prime_func(parent_feature_bad)

		parent_delta_good -= parent_classify_delta

		# only one layer of weights forward
		self.grad_weights_forward += -parent_delta_good.dot(feature_good.T) + parent_delta_bad.dot(feature_bad.T)

		#-------------------- Back-prop: Kids Nodes --------------------#		
		delta_from_parent_good = self.weights_forward.T[:2*self.word_vec_dim].dot(parent_delta_good)
		delta_from_parent_bad  = self.weights_forward.T[:2*self.word_vec_dim].dot(parent_delta_bad)

		for i in range(pair_num):
			self.grad_word_dict[:, bad_pair[0,i]]  += delta_from_parent_bad[:self.word_vec_dim,i]
			self.grad_word_dict[:, bad_pair[1,i]]  += delta_from_parent_bad[self.word_vec_dim:2*self.word_vec_dim,i]

			self.grad_word_dict[:, good_pair[0,i]] += -delta_from_parent_good[:self.word_vec_dim,i] + classify_delta_left[:,i]
			self.grad_word_dict[:, good_pair[1,i]] += -delta_from_parent_good[self.word_vec_dim:2*self.word_vec_dim,i] + classify_delta_right[:,i]

		total_classify_error = parent_cross_entropy_error + cross_entropy_error_right + cross_entropy_error_left
		self.total_loss += parent_merge_cost + total_classify_error

		print '%20s %12f' %('AVG Classify Error:', total_classify_error/pair_num)
		print '%20s %12f' %('AVG Good score:', np.sum(parent_score_good)/pair_num)
		print '%20s %12f' %('AVG Bad score:', np.sum(parent_score_bad)/pair_num)
		print '%20s %12f' %('AVG Merge Loss:', parent_merge_cost/pair_num)

		return pair_num
