import os.path
from scipy.optimize import fmin_l_bfgs_b

from core.datasets.entity import load_pretrain_dataset
from core.datasets.config import DATA_DIR
from core.recnn.init_parameter import init_RecNN_parameters
from core.recnn.recnn_pretrainer import RecNNPreTrainer

def main():
	word_vec_dim = 50
	vocabulary_size = 18281
	classify_category_num = 5

	pretrain_params_file = '../data/pretrained_params'

	pretrain_dataset = load_pretrain_dataset('stanford_sentiment')
	pretrainer = RecNNPreTrainer(word_vec_dim=word_vec_dim, vocabulary_size=vocabulary_size, 
		classify_category_num=classify_category_num)
	params0 = init_RecNN_parameters(word_vec_dim=word_vec_dim, vocabulary_size=vocabulary_size, 
		classify_category_num=classify_category_num)

	result = fmin_l_bfgs_b(pretrainer.pre_train, params0, args=(pretrain_dataset,), iprint=1, maxiter=10, factr=1e10)
	result[0].tofile(pretrain_params_file)

if __name__ == '__main__':
	main()
