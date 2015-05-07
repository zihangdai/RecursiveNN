from mrnn.analysis.visualize import (
	load_term_id_map,
	save_embedded_features,
	plot_features,
)

def main():
	key = 'stan'
	word_vec_dim = 50

	term_id = load_term_id_map(key)
	save_embedded_features(key, word_vec_dim)
	plot_features(key, word_vec_dim)

if __name__ == '__main__':
	main()
