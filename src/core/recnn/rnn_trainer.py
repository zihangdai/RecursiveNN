import sys
import math
import numpy as np
import scipy as sp
from scipy.misc import logsumexp
from numpy import linalg as LA

import core.datasets.entity

class RecNNTrainer:
	"""docstring for RecNNTrainer"""
	def __init__(self, word_vec_dim, vocabulary_size, classify_category_num, reg_lambda=0.001, lose_kappa=0.05):
		self.word_vec_dim          = word_vec_dim
		self.vocabulary_size       = vocabulary_size
		self.classify_category_num = classify_category_num
		
		self.corr_tree = None
		self.best_tree = None
		
		self.activate_func 	     = lambda x : 1 / (1 + np.exp(-x))
		self.activate_prime_func = lambda fx: fx * (1 - fx)
		self.classify_func       = lambda x : np.exp(x - logsumexp(x))		
		#self.classify_prime_func = params.classify_prime_func

		self.regularization_lambda = reg_lambda
		self.structure_lose_kappa  = lose_kappa

	
	def recursive_parse(self, current_node):
		if current_node.feature_entry_id:
			current_node.feature = self.word_dict[:, [current_node.feature_entry_id] ]						
			self.recnn_tree_nodes.append( current_node.clone() )
		else:
			self.recursive_parse(current_node.left)
			self.recursive_parse(current_node.right)
			# calculate the parent's feature
			kis_feature = np.vstack( (current_node.left.feature, current_node.right.feature, 1) )
			current_node.feature = self.activate_func( self.weights_forward.dot(kis_feature) )
			
			# calculate merge score
			score = self.weights_score.dot(current_node.feature)[0,0]
			self.corr_tree.total_score += score
			if math.isnan(current_node.feature[0]):
				print 'Nan feature:', self.weights_forward, kis_feature
		
		# common part for both lead nodes and non-terminal nodes
		# calculate classification result and cross_entropy_error
		classify_input = self.weights_classify.dot(np.vstack((current_node.feature,1)))
		current_node.classify_output = self.classify_func(classify_input)		
		cross_entropy_error = -np.sum(current_node.category_vector.reshape(-1,1) * np.log(current_node.classify_output+1e-10))
		if math.isnan(cross_entropy_error):
			print 'Nan CEE:', current_node.category_vector, np.log(current_node.classify_output+1e-10).flatten(), classify_input.flatten(), current_node.feature
		self.corr_tree.classify_cost += cross_entropy_error

	def parse_corr_tree_structure(self, data):
		# retrieve the correct tree from training data and set initial cost values
		self.corr_tree 			     = data.tree
		self.corr_tree.total_score   = 0.0
		self.corr_tree.classify_cost = 0.0

		# retrieve valid_span_range and feature_entry_ids from training data
		# valid_span_range: valid subspan tree structure
		# feature_entry_ids: input sentence entry ids
		self.valid_span_range  = data.range_node_map
		self.feature_entry_ids = data.feature_entry_ids
		self.recnn_tree_nodes    = []
		
		# begin to recursively parse the correct tree and meanwhile finish feed-forward process 		
		self.recursive_parse(self.corr_tree.root)

	def check_valid_span_func(span_range):
		return self.valid_span_range.has_key(span_range)

	def parse_params(self, params):
		# word_dict
		word_dict_end         = self.word_vec_dim * self.vocabulary_size
		self.word_dict        = params[:word_dict_end].reshape(self.word_vec_dim, self.vocabulary_size)
		# weights_score
		weights_score_end     = word_dict_end + self.word_vec_dim
		self.weights_score    = params[word_dict_end:weights_score_end].reshape(1, self.word_vec_dim)
		if LA.norm(self.weights_score) > 0:
			self.weights_score    = self.weights_score / LA.norm(self.weights_score)
		# weights_forward
		weights_forward_end   = weights_score_end + self.word_vec_dim * (self.word_vec_dim*2+1)
		self.weights_forward  = params[weights_score_end:weights_forward_end].reshape(self.word_vec_dim, self.word_vec_dim*2+1)
		# weights_classify
		self.weights_classify = params[weights_forward_end:].reshape(self.classify_category_num, self.word_vec_dim+1)		

		# print('%18s %12f \t %12f \t %12f' % ('word_dict:', np.max(self.word_dict), np.min(self.word_dict), np.mean(self.word_dict)))
		# print('%18s %12f \t %12f \t %12f' % ('weights_score:', np.max(self.weights_score), np.min(self.weights_score), np.mean(self.weights_score)))
		# print('%18s %12f \t %12f \t %12f' % ('weights_forward:', np.max(self.weights_forward), np.min(self.weights_forward), np.mean(self.weights_forward)))
		# print('%18s %12f \t %12f \t %12f' % ('weights_classify:', np.max(self.weights_classify), np.min(self.weights_classify), np.mean(self.weights_classify)))

	def init_loss_and_gradient(self):
		self.total_merge_loss      = 0.0
		self.total_classify_error  = 0.0		
		self.grad_word_dict        = np.zeros_like(self.word_dict)
		self.grad_weights_score    = np.zeros_like(self.weights_score)
		self.grad_weights_forward  = np.zeros_like(self.weights_forward)
		self.grad_weights_classify = np.zeros_like(self.weights_classify)
	
	def train(self, params, training_dataset):
		self.parse_params(params)
		self.init_loss_and_gradient()
		# for each data in training_dataset
		data_num = len(training_dataset)
		for data in training_dataset:
			# feed-forward and calculate cost (loss)			
			self.parse_corr_tree_structure(data)			
			ret = self.feed_forward_best_tree()
			if ret < 0:
				continue
			self.total_merge_loss     += self.best_tree.total_score - self.corr_tree.total_score
			self.total_classify_error += self.corr_tree.classify_cost
			# back-propagate and calculate gradient
			self.back_prop_corr_tree(self.corr_tree.root, 0.0)
			self.back_prop_best_tree(self.best_tree.root, 0.0)

		# flatten grad
		flatten_grad = np.hstack( (self.grad_word_dict.flatten(), self.grad_weights_score.flatten(), 
			self.grad_weights_forward.flatten(), self.grad_weights_classify.flatten()) )

		# average cost and gradient, and add regularization term
		regularization_cost = self.regularization_lambda/2 * params.dot(params)
		total_loss = self.total_merge_loss + self.total_classify_error
		avg_total_loss = total_loss/data_num + regularization_cost
		avg_flatten_grad  = flatten_grad/data_num + self.regularization_lambda * params

		print ('%20s %12f' %('AVG Classify Error:', self.total_classify_error/data_num))
		print ('%20s %12f' %('AVG Merge Loss:', self.total_merge_loss/data_num))
		print ('%20s %12f' %('AVG Total Loss:', avg_total_loss))

		return avg_total_loss, avg_flatten_grad
		
	def feed_forward_best_tree(self):
		if len(self.recnn_tree_nodes) <= 1:
			return -1
		# init score cost from merge decision
		total_score = 0.0

		# extract feature vector from self.word_dict by feature_entry_ids indexing
		input_feature  = self.word_dict[:, self.feature_entry_ids]

		# declair basic variables
		sentence_len   = input_feature.shape[1]
		total_node_num = 2 * sentence_len - 1
					
		# nodes_feature: word_vec_dim x total_node_num (pre-allocate all memory)
		# nodes_feature[:,i]: feature vector (column) of node i
		nodes_feature = np.zeros( (self.word_vec_dim, total_node_num) )
		nodes_feature[:, :sentence_len] = input_feature

		# nodes_span_range: list of tuples
		# nodes_span_range[i][0]: the left-end node index of the span range of node i
		# nodes_span_range[i][1]: the right-end node index of the span range of node i
		nodes_span_range = [(i,i) for i in range(total_node_num)]

		# pair_index: all potential pairs to be merged, 2 x dynamic_colmun_num
		# pair_index[0][i]: index of the left node of pair i 
		# pair_index[1][i]: index of the right node of pair i
		pair_index = -np.ones( (2, sentence_len-1) ).astype(int)
		pair_index[0] = np.arange(sentence_len-1)
		pair_index[1] = np.arange(1,sentence_len)
		
		# get structure loss for each potential parent node
		structure_loss = self.structure_lose_kappa * np.ones_like(pair_index[0])
		for i in range(pair_index.shape[1]):
			left_end_index  = nodes_span_range[pair_index[0,i]][0]
			right_end_index = nodes_span_range[pair_index[1,i]][1]
			node_span_range = (left_end_index, right_end_index)
			if self.valid_span_range.has_key(node_span_range):
				structure_loss[i] = 0

		# calculate new potential parents feature
		# parents feature will have the same column number as pair_index
		pair_feature_matrix = np.vstack( (nodes_feature[:,pair_index[0]], nodes_feature[:,pair_index[1]], np.ones_like(pair_index[0])) )
		parent_feature = self.activate_func( self.weights_forward.dot(pair_feature_matrix) )

		# calculate new potential parents score
		parent_score = self.weights_score.dot(parent_feature)		
		parent_score = parent_score + structure_loss

		for new_node_index in range(sentence_len, total_node_num):
			# get the max merge score and greedily choose the one as new parent node
			max_score = np.max(parent_score)
			max_score_index = np.argmax(parent_score)

			# update the total merge score
			total_score += max_score

			# update the feature of the new node 
			nodes_feature[:, new_node_index] = parent_feature[:, max_score_index]

			# extract kids index of the new chosen parent
			# and then update nodes_span_range for the new node
			kids_index = pair_index[:, max_score_index]
			# print pair_index, kids_index
			nodes_span_range[new_node_index] = (nodes_span_range[kids_index[0]][0], nodes_span_range[kids_index[1]][1])
			
			# allocate node instance for this new node
			# for the best tree, only left_kid, right_kid, and feature are needed
			new_tree_node = core.datasets.entity.RecursiveNNTreeNode(category_ids=None, feature_entry_id=None, 
				left_right_child=(self.recnn_tree_nodes[kids_index[0]], self.recnn_tree_nodes[kids_index[1]]))
			new_tree_node.feature = nodes_feature[:, [new_node_index] ]
			self.recnn_tree_nodes.append(new_tree_node)

			# if this is the last node, we can safely exit the for loop
			if new_node_index == total_node_num-1:
				break

			# delete the merged column
			to_delete_column_index = max_score_index
			new_pair_column_index  = [max_score_index - 1, max_score_index]
			
			pair_index = np.delete(pair_index, to_delete_column_index, 1)
			parent_feature = np.delete(parent_feature, to_delete_column_index, 1)
			parent_score = np.delete(parent_score, to_delete_column_index, 1)


			# update pair_index, prepare new_pair_index
			# the chosen(delete) node is the left-most node
			if new_pair_column_index[0] < 0:
				pair_index[0,new_pair_column_index[1]] = new_node_index
				new_pair_index = pair_index[:,new_pair_column_index[1]].reshape(-1,1)
				del new_pair_column_index[0]
			# the chosen(delete) node is the right-most node
			elif new_pair_column_index[1] >= pair_index.shape[1]:
				pair_index[1,new_pair_column_index[0]] = new_node_index
				new_pair_index = pair_index[:,new_pair_column_index[0]].reshape(-1,1)
				del new_pair_column_index[1]
			else:
				pair_index[1,new_pair_column_index[0]] = new_node_index
				pair_index[0,new_pair_column_index[1]] = new_node_index
				new_pair_index = pair_index[:,new_pair_column_index]
			
			# calculate the new structure loss
			new_structure_loss = self.structure_lose_kappa * np.ones_like(new_pair_index[0])
			for i in range(new_pair_index.shape[1]):
				left_end_index  = nodes_span_range[pair_index[0,i]][0]
				right_end_index = nodes_span_range[pair_index[1,i]][1]
				node_span_range = (left_end_index, right_end_index)
				if self.valid_span_range.has_key(node_span_range):
					new_structure_loss[i] = 0

			# calculate new potential parents feature and score
			pair_feature_matrix = np.vstack( (nodes_feature[:,new_pair_index[0]], nodes_feature[:,new_pair_index[1]], np.ones_like(new_pair_index[0])) )
			new_parent_feature  = self.activate_func( self.weights_forward.dot(pair_feature_matrix) )

			new_parent_score = self.weights_score.dot(new_parent_feature)
			new_parent_score = new_parent_score + new_structure_loss

			# update parent_feature, parent_score
			parent_feature[:,new_pair_column_index] = new_parent_feature
			parent_score[:,new_pair_column_index] = new_parent_score
		
		# create the best tree instance
		self.best_tree = core.datasets.entity.RecursiveNNTree(root=self.recnn_tree_nodes[-1])
		self.best_tree.total_score = total_score

		return 1

	# Recursive back-propagate for the Best Constructed Tree:
	# 1) For Delta of a Node: Condiser delta_from_score, delta_from_parent
	# 2) For Gradients: Condiser grad_weights_score, grad_weights_forward, grad_word_dict (only at the leaf node)
	
	def back_prop_best_tree(self, current_node, delta_from_parent):
		# extract stored data used to calculate delta and gradient of current node 
		# from current node
		current_feature = current_node.feature
		# from kids of current node
		left_kid  = current_node.left
		right_kid = current_node.right
		kids_pair_feature = np.vstack( (left_kid.feature, right_kid.feature, 1) )

		# calculate delta of current node
		delta_from_score   = self.weights_score.T * self.activate_prime_func(current_feature)
		delta_current_node = delta_from_parent + delta_from_score

		# gradient direct related to current node 
		self.grad_weights_score   += current_feature.T
		self.grad_weights_forward += delta_current_node.dot(kids_pair_feature.T)

		# preparent quantity back-propagate to kids
		weights_forward_dot_current_delta = self.weights_forward.T.dot(delta_current_node)
		weights_forward_dot_current_delta = weights_forward_dot_current_delta[:2*self.word_vec_dim].reshape(self.word_vec_dim, 2, order='F')
		
		kids_tuple  = (left_kid, right_kid)

		for index in range(2):
			kid_node  = kids_tuple[index]
			if kid_node.is_leaf():
				kid_delta = weights_forward_dot_current_delta[:,[index]]
				self.grad_word_dict[:, [kid_node.feature_entry_id]] += kid_delta
			else:
				kid_delta = weights_forward_dot_current_delta[:,[index]] * self.activate_prime_func(kid_node.feature)
				self.back_prop_best_tree(kid_node, kid_delta)
		
	# Recursive back-propagate for the Correct Tree:
	# 1) For Delta of a Node: Condiser delta_from_score, delta_from_classify, delta_from_parent
	# 2) For Gradients: Condiser grad_weights_score, grad_weights_forward, grad_weights_classify, grad_word_dict (only at the leaf node)
	
	def back_prop_corr_tree(self, current_node, delta_from_parent):
		# extract stored data used to calculate delta and gradient of current node 
		# from current node
		current_feature = current_node.feature
		current_classify_output = current_node.classify_output
		current_category_vector = current_node.category_vector.reshape(-1,1)
		current_classify_diff   = current_classify_output - current_category_vector
		# from kids of current node
		left_kid  = current_node.left
		right_kid = current_node.right
		# [2d+1 x 1]
		kids_pair_feature = np.vstack( (left_kid.feature, right_kid.feature, 1) )

		# calculate delta of current node
		# delta_current_node = delta_from_parent + delta_from_score + delta_from_classify
		# [d x 1] * [d x 1]
		delta_from_score    = self.weights_score.T * self.activate_prime_func(current_feature)
		# ([d+1 x c] \cdot [c x 1])[:d] * [d x 1]
		delta_from_classify = self.weights_classify.T.dot(current_classify_diff)[:self.word_vec_dim] * self.activate_prime_func(current_feature)
		# [d x 1] + [d x 1] + [d x 1]
		delta_current_node  = delta_from_parent + delta_from_score - delta_from_classify

		# gradient direct related to current node 
		# [1 x d]
		self.grad_weights_score    -= current_feature.T
		# [c x 1] \cdot [1 x d+1]
		self.grad_weights_classify += current_classify_diff.dot( np.hstack( ( current_feature.T, np.ones((1,1)) ) ) )
		# [d x 1] \cdot [1 x 2d+1]
		self.grad_weights_forward  -= delta_current_node.dot(kids_pair_feature.T)

		# preparent quantity back-propagate to kids
		weights_forward_dot_current_delta = self.weights_forward.T.dot(delta_current_node)
		weights_forward_dot_current_delta = weights_forward_dot_current_delta[:2*self.word_vec_dim].reshape(self.word_vec_dim, 2, order='F')
		
		kids_tuple  = (left_kid, right_kid)

		for index in range(2):
			kid_node  = kids_tuple[index]
			if kid_node.is_leaf():
				kid_delta = weights_forward_dot_current_delta[:,[index]]
				kid_node_feature = kid_node.feature
				kid_node_classify_output = kid_node.classify_output
				kid_node_category_vector = kid_node.category_vector.reshape(-1,1)
				kid_node_classify_diff   = kid_node_classify_output - kid_node_category_vector
				
				# ([d+1 x c] \cdot [c x 1])[:d] * [d x 1]
				kid_delta_from_classify = self.weights_classify.T.dot(kid_node_classify_diff)[:self.word_vec_dim]

				# [c x 1] \cdot [1 x d+1]
				self.grad_weights_classify += kid_node_classify_diff.dot( np.hstack( ( kid_node_feature.T, np.ones((1,1)) ) ) )
				self.grad_word_dict[:, [kid_node.feature_entry_id]] -= (kid_delta - kid_delta_from_classify)
			else:
				kid_delta = weights_forward_dot_current_delta[:,[index]] * self.activate_prime_func(kid_node.feature)
				self.back_prop_corr_tree(kid_node, kid_delta)
