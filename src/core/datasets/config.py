import os

TASKS = [
	{
		"name": "stanford_sentiment", 
		"data_dir_train": "stanford_sentiment/train/", 
		"data_dir_test": "stanford_sentiment/test/", 
		# "data_dir": "",
		# "train_test_ratio": "",
		# train_test_ratio: a real number between 0~1, is the ratio of training set to the whole set. 
		# This is only used when a dataset only define data_dir without explicitly defining data_dir_train and data_dir_test. In 
		# this case, the whole dataset will be splitted ranomly according to this ratio
		"parsed_file_train": "stanford_sentiment_train.jsonl",
		"parsed_file_test": "stanford_sentiment_test.jsonl",
	}
]

PRETRAIN_VECTOR_MODEL = None
FEATURE_VECTOR_SAVEFILE = "feature_vector.pickle"
FEATURE_DIM = 100

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data/")