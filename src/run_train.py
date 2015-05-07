import os.path
import numpy as np
import random
import math, sys
from scipy.optimize import fmin_l_bfgs_b

from core.datasets.entity import load_dataset
from core.recnn.recnn_trainer import RecNNTrainer

from core.recnn.init_parameter import init_RecNN_parameters

def main():
	word_vec_dim = 50
	vocabulary_size = 18281
	classify_category_num = 5

	pretrain_params_file = '../data/pretrained_params'
	train_params_file = '../data/trained_params'

	dataset = load_dataset('stanford_sentiment')['train'][1: 1000]

	trainer = RecNNTrainer(word_vec_dim=word_vec_dim, vocabulary_size=vocabulary_size, 
		classify_category_num=classify_category_num)

	params = init_RecNN_parameters(word_vec_dim=word_vec_dim, vocabulary_size=vocabulary_size, 
		classify_category_num=classify_category_num)

	result = fmin_l_bfgs_b(trainer.train, params, args=(dataset,), iprint=1, maxiter=1, factr=1e8)
	
	result[0].tofile(train_params_file)

if __name__ == '__main__':
	main()