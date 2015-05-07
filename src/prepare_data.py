from core.datasets.parse_ptb import build_dataset

def main():
	"""
	This script collect all terms from different tasks, assign each of them an unique id, and construct grammar tree for each task
	"""
	build_dataset()

	# Test
	# trees = parse_file("test/data")
	# for tree in trees: tree.print_tree()

if __name__ == '__main__':
	main()
