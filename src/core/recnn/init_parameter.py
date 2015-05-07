import numpy as np

def init_RecNN_parameters(word_vec_dim, vocabulary_size, classify_category_num):
	# init lookup_dict
	init_range_dict = 1 / np.sqrt(word_vec_dim)
	word_dict = -init_range_dict + 2 * init_range_dict * np.random.rand(word_vec_dim, vocabulary_size)		

	# init score_weights
	init_range_score = 4 / word_vec_dim
	weights_score = init_range_score * np.random.rand(1, word_vec_dim)

	# init forward_weights
	init_range_forward = 1 / np.sqrt(2 * word_vec_dim)
	weights_forward = -init_range_forward + 2 * init_range_forward * np.random.rand(word_vec_dim, word_vec_dim*2+1)
	weights_forward[:,-1] = 0.0

	# sparsify forward weights
	zero_num = 35
	for i in range(weights_forward.shape[0]):
		zero_index = np.random.permutation(weights_forward.shape[1])[:zero_num]
		weights_forward[i, zero_index] = 0

	# init classify_weights
	init_range_classify = 1 / np.sqrt(word_vec_dim + classify_category_num)
	weights_classify = -init_range_classify + 2 * init_range_classify * np.random.rand(classify_category_num, word_vec_dim+1)
	weights_classify[:,-1] = 0.0

	# flatten params
	params = np.hstack( (word_dict.flatten(), weights_score.flatten(), weights_forward.flatten(), weights_classify.flatten()) )

	return params