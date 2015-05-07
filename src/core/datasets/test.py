import unittest
import entity
import copy
from core.datasets import entity
from core.datasets.config import TASKS

class DatasetTestCase(unittest.TestCase):

	def test_load_dataset(self):
		zero_feature_entry_id_cnt = {"count": 0}
		for task in TASKS:
			instances = entity.load_dataset(task["name"])
			instances = instances["train"] + instances["test"]
			
			counts = []
			for idx, instance in enumerate(instances):
				assert instance.tree is not None
				assert instance.tree.root is not None

				counter = {"counter": 0}
				def check_node(node):
					assert node.category_ids is not None
					assert node.feature_entry_id is not None or (node.left and node.right)
					assert node.range
					# assert node.category_vector.shape[0] == 5
					counter["counter"] += 1
					if node.feature_entry_id is 0:
						zero_feature_entry_id_cnt["count"] += 1

				instance.tree.apply_all(check_node)

				counts.append(counter["counter"])
				# if idx < 10: print_node(instance.tree.root)

			print max(counts)
			print float(sum(counts)) / len(counts)
			print "zero_feature_entry_id_cnt:", zero_feature_entry_id_cnt["count"]

		feature_matrix = entity.get_feature_matrix()

	def test_clone(self):
		root = entity.RecursiveNNTreeNode(1, 2)
		left = entity.RecursiveNNTreeNode(1, 10)
		right = entity.RecursiveNNTreeNode(1, 12)

		root.left = left
		root.right = right

		cloned = root.clone()

		root.left.feature_entry_id = 100

		assert cloned.left.feature_entry_id == 10

	def test_load_pretrain(self):
		pretrain_num = None
		for task in TASKS:
			result = entity.load_pretrain_dataset(task["name"])
			# good pairs 		          : 2 x n matrix, each column is [left feature id; right feature id]
			# good pair categories        : [k x n, k x n] 
			# good pair parent categories : k x n
			# bad pairs 			      : 2 x n matrix, each column is [left feature id; right feature id]
			print result["good_pairs"].shape
			print result["good_pairs"][:, 0: 5]

			print result["good_pair_categories"][0].shape, result["good_pair_categories"][1].shape
			print result["good_pair_categories"][0][:, 0: 5]
			print result["good_pair_categories"][1][:, 0: 5]

			print result["good_pair_parents_categories"].shape
			print result["good_pair_parents_categories"][:, 0: 5]
			print result["bad_pairs"].shape
			print result["bad_pairs"][:, 0: 5]
			# assert len(good_pairs) == pretrain_num and len(good_pair_parents) == pretrain_num and len(bad_pairs) == pretrain_num


if __name__ == '__main__':
	unittest.main()
