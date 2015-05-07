from __future__ import with_statement
import os.path

import numpy as np
import matplotlib.pyplot as plt

import tsne

VOCABULARY_SIZE = {
	'multi': 35738,
	'stan': 18281,
}
PARAMS_FILE = {
	'multi': 'trained_params_{0}_multi',
	'stan': 'trained_params_{0}_stan',
}
PARAMS_EMBEDDING_FILE = {
	'multi': 'trained_params_{0}_multi_2d.npy',
	'stan': 'trained_params_{0}_stan_2d.npy',
}
TERM_ID_MAP_FILE = {
	'multi': 'term_id_map_multi',
	'stan': 'term_id_map_stan',
	'penn': 'term_id_map_penn',
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data/")

def load_term_id_map(key):
	map_file = os.path.join(DATA_DIR, TERM_ID_MAP_FILE[key])
	term_id = {}
	with open(map_file) as data:
		for line in data:
			k, v = line.strip().split()
			term_id[k] = int(v)
	return term_id

def _get_params_file(key, word_vec_dim):
	file_path = PARAMS_FILE[key].format(word_vec_dim)
	file_path = os.path.join(DATA_DIR, file_path)
	return file_path

def _get_params_embedding_file(key, word_vec_dim):
	file_path = PARAMS_EMBEDDING_FILE[key].format(word_vec_dim)
	file_path = os.path.join(DATA_DIR, file_path)
	return file_path

def save_embedded_features(key, word_vec_dim=50):
	features = np.fromfile(_get_params_file(key, word_vec_dim))
	d, n = word_vec_dim, VOCABULARY_SIZE[key]
	features = features[:d*n].reshape(d, n).T
	embedding = tsne.tsne(features, no_dims=2, initial_dims=d)
	np.save(_get_params_embedding_file(key, word_vec_dim), embedding)

def plot_features(key, word_vec_dim=50):
	embedding = np.load(_get_params_embedding_file(key, word_vec_dim))
	plt.scatter(embedding[:, 0], embedding[:, 1], 20)
	plt.show()
