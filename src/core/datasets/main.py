
from core.datasets import entity

def print_node(node, indent=""):
	if node.is_leaf():
		print "%s[%d] - %d" %(indent, node.category_id, node.feature_entry_id)
	else:
		print "%s[%d]" %(indent, node.category_id)
		for child in (node.left, node.right):
			print_node(child, indent + "  ")

def main():
	# This is just testing loading dataset from any arbitrary places
	for name in ("stanford_sentiment"):
		instances = entity.load_dataset(name)

		for idx, instance in enumerate(instances):
			assert instance.tree is not None
			assert instance.tree.root is not None

			def check_node(node):
				assert node.category_id is not None
				assert node.feature_entry_id or (node.left and node.right)

			instance.tree.apply_all(check_node)

			# if idx < 10: print_node(instance.tree.root)

	matrix = entity.get_feature_matrix()
	print matrix.data.shape

if __name__ == '__main__':
	main()