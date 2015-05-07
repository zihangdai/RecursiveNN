import numpy as np
import sys
from core.datasets.entity import load_dataset
from core.recnn.recnn_tester import RecNNTester

def main():
	word_vec_dim = 50
	vocabulary_size = 18281
	classify_category_num = 5
	
	arguments = sys.argv[1:]
	if len(arguments) == 0:
		train_params_file = '../data/trained_params'
	else:
		train_params_file = arguments[0]
	
	dataset = load_dataset('stanford_sentiment')['test']
	tester = RecNNTester(word_vec_dim=word_vec_dim, vocabulary_size=vocabulary_size, 
		classify_category_num=classify_category_num)

	params2 = np.fromfile(train_params_file)

	tester.test(params2, dataset, 0)

if __name__ == '__main__':
	main()
