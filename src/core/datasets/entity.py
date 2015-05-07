from numpy import ndarray, array
import os
try:
	import ujson as json
except ImportError:
	import json
import cPickle as pickle
import numpy
import random
from core.datasets.config import FEATURE_VECTOR_SAVEFILE, TASKS, DATA_DIR

class FeatureMatrix:

	def __init__(self):
		if not os.path.exists(DATA_DIR + "/" + FEATURE_VECTOR_SAVEFILE):
			raise Exception("Feature vector file not found. You should run parse_ptb to build data first")

		self.data = pickle.load(open(DATA_DIR + "/" + FEATURE_VECTOR_SAVEFILE, 'rb'))

class RecursiveNNTree:
	@staticmethod
	def fromdict(dictobj):
		return RecursiveNNTree(root=RecursiveNNTreeNode.fromdict(dictobj["root"]))

	def __init__(self, root):
		self.root = root

	def apply_all(self, apply_func, order="pre"):
		"""
		apply_func: a function takes a RecursiveNNTreeNode as its single argument, and will be applied to 
					every node in the tree in pre-order
		"""
		self.__apply_node(self.root, apply_func, order)

	def __apply_node(self, node, apply_func, order):
		if order == "pre":
			apply_func(node)
			if node.left:
				self.__apply_node(node.left, apply_func, order)
			if node.right:
				self.__apply_node(node.right, apply_func, order)
		elif order == "mid":
			if node.left:
				self.__apply_node(node.left, apply_func, order)
			apply_func(node)
			if node.right:
				self.__apply_node(node.right, apply_func, order)
		elif order == "post":
			if node.left:
				self.__apply_node(node.left, apply_func, order)
			if node.right:
				self.__apply_node(node.right, apply_func, order)
			apply_func(node)
		else:
			raise Exception("Incorrect order type")

	def dict(self):
		return {"root": self.root.dict()}

class RecursiveNNTreeNode:
	@staticmethod
	def fromdict(dictobj):
		left, right = None, None
		if dictobj["left"]: left = RecursiveNNTreeNode.fromdict(dictobj["left"])
		if dictobj["right"]: right = RecursiveNNTreeNode.fromdict(dictobj["right"])

		node = RecursiveNNTreeNode(dictobj["category_ids"], 
								   dictobj["feature_entry_id"], 
								   (left, right))
		node.category_vector = array(dictobj["category_vector"])

		return node

	def __init__(self, category_ids, feature_entry_id=None, left_right_child=(None, None), category_vector=None):
		"""
		category_id: int, class label
		feature_entry_id: only not None for leaf node, shows which term is it
		left_right_child: a tuple, only for non-leaf node
		There're some other fields which should be set afterwards
		"""
		self.category_ids = category_ids
		self.feature_entry_id = feature_entry_id
		self.left, self.right = left_right_child
		self.category_vector = category_vector

		# The follow fields are temporary, dynamically calculated
		self.feature = None # Should be a feature column vector
		self.range = (None, None)
		


	def is_leaf(self):
		return self.feature_entry_id is not None

	def left(self):
		return self.left

	def right(self):
		return self.right

	def child(self, idx):
		"""
		idx=1 -> left child, idx=2 -> right child
		"""
		if idx == 1: return self.left
		elif idx == 2: return self.right
		else: raise Exception("Index overflow")

	def dict(self):
		data = {
			"category_ids": self.category_ids,
			"feature_entry_id": self.feature_entry_id,
			"left": None,
			"right": None,
			"category_vector": list(self.category_vector),
		}
		if self.left is not None: data["left"] = self.left.dict()
		if self.right is not None: data["right"] = self.right.dict()
		return data

	def clone(self):
		cloned = RecursiveNNTreeNode(category_ids=self.category_ids, 
									 feature_entry_id=self.feature_entry_id)
		if self.left:
			cloned.left = self.left.clone()
		if self.right:
			cloned.right = self.right.clone()

		cloned.feature = self.feature
		return cloned
		


class RecursiveNNDataInstance:

	@staticmethod
	def fromdict(dictobj):
		return RecursiveNNDataInstance(RecursiveNNTree.fromdict(dictobj["tree"]))

	def __init__(self, tree):
		"""
		A data instance consists of:
		tree: the tree structure. each node contains category id, child
		"""
		self.tree = tree
		self.feature_entry_ids = [] # A list of feature ids, ordered from left to right
		self.range_node_map = {} # A hashmap from node range to node itself

		counter = {"counter": 0}
		def recursive_visit(node):
			# Collect feature id
			if node.is_leaf():
				self.feature_entry_ids.append(node.feature_entry_id)

			# Set range
			if node.is_leaf():
				node.range = (counter["counter"], counter["counter"])
				counter["counter"] += 1
			else:
				node.range = (node.left.range[0], node.right.range[1])

			self.range_node_map[node.range] = node

		self.tree.apply_all(recursive_visit, "post")

	def dict(self):
		"""
		return a json serializable object
		"""
		return {
			"tree": self.tree.dict()
		}



def load_dataset(name, max_num=None):
	"""
	Load a dataset with given name
	return {"train": [...], "test": [...]}
	"""
	print "Loading %s ..." %name,
	for task in TASKS:
		if task['name'] == name:
			parsed_file_train = os.path.join(DATA_DIR, task['parsed_file_train'])
			parsed_file_test  = os.path.join(DATA_DIR, task['parsed_file_test'])

			if not os.path.exists(parsed_file_train) or not os.path.exists(parsed_file_test):
				raise Exception("Dataset file not existed, you might need to build data set first")

			result = {"train": [], "test": []}

			for idx, line in enumerate(open(parsed_file_train)):
				if line[-1] == '\n': line = line[: -1]
				result["train"].append(json.loads(line))
				if max_num is not None and idx >= max_num:
					break

			for idx, line in enumerate(open(parsed_file_test)):
				if line[-1] == '\n': line = line[: -1]
				result["test"].append(json.loads(line))
				if max_num is not None and idx >= max_num:
					break

		 	# Convert from dict to class
		 	print "json file found, parsing...",

		 	result["train"]  = [RecursiveNNDataInstance.fromdict(instance) for instance in result["train"] ]
		 	result["test"]  = [RecursiveNNDataInstance.fromdict(instance) for instance in result["test"] ]
		 	print "done"

		 	return result

	raise Exception("Given data set name incorrect, not found in config. Please double check.")

def load_pretrain_dataset(name, pretrain_pair_num=None):
	"""
	Given the data set name, select <pretrain_pair_num> good pairs and <pretrain_pair_num> bad pairs
	as well as <pretrain_pair_num> parents of good pairs
	"""
	dataset = load_dataset(name)["train"]
	pairs_and_parent  = []
	leaves = []

	typecount = {}
	# Collect pairs
	for instance in dataset:
		def collect_pair(node):
			if node.is_leaf():
				leaves.append(node)
			elif node.left.is_leaf() and node.right.is_leaf():
				pairs_and_parent.append(((node.left, node.right), node))

			if not node.is_leaf() and (node.left.is_leaf() and node.right.is_leaf()):
				if len(node.category_ids) > 0:
					if typecount.has_key(node.category_ids[0]):
						typecount[node.category_ids[0]] += 1
					else:
						typecount[node.category_ids[0]] = 1

		instance.tree.apply_all(collect_pair)
	
	# SELECT GOOD PAIRS, RETURN ALL PAIRS
	pair_set = set([p[0] for p in pairs_and_parent])
	if pretrain_pair_num is None:
		pretrain_pair_num = len(pair_set)
	# random.shuffle(pairs_and_parent)

	good_pairs = [p[0] for p in pairs_and_parent]
	good_pair_parents = [p[1] for p in pairs_and_parent]

	# SELECT BAD PAIRS
	bad_pairs = []
	random.shuffle(leaves)

	for i1 in range(0, len(leaves) - 1, 2):
		i2 = i1 + 1
		node1, node2 = leaves[i1], leaves[i2]

		if (node1, node2) in pair_set:
			print "NOT A BAD PAIR"
			continue

		bad_pairs.append((node1, node2))
		if len(bad_pairs) >= pretrain_pair_num:
			break
			
	# Convert to an easy-to-use form
	# good pairs 		    : 2 x n matrix, each column is [left feature id; right feature id]
	# good pair categories  : [k x n, k x n] 
	# good pair parent categories: k x n
	# bad pairs 			: 2 x n matrix, each column is [left feature id; right feature id]
	classnum = len(leaves[0].category_vector)
	result = {}
	result["good_pairs"] = numpy.ndarray((2, len(good_pairs)))
	result["good_pair_categories"] = [numpy.ndarray((classnum, len(good_pairs))) for i in range(2)]
	result["good_pair_parents_categories"] = numpy.ndarray((classnum, len(good_pairs)))
	for idx, pair in enumerate(good_pairs):
		result["good_pairs"][:, idx] = [p.feature_entry_id for p in pair]
		result["good_pair_categories"][0][:, idx] = pair[0].category_vector
		result["good_pair_categories"][1][:, idx] = pair[1].category_vector
		result["good_pair_parents_categories"][:, idx] = good_pair_parents[idx].category_vector

	result["bad_pairs"] = numpy.ndarray((2, len(bad_pairs)))
	for idx, pair in enumerate(bad_pairs):
		result["bad_pairs"][:, idx] = [p.feature_entry_id for p in pair]

	return result

def get_feature_matrix():
	if not globals().has_key("__feature_matrix"):
		globals()["__feature_matrix"] = FeatureMatrix()
	return globals()["__feature_matrix"]
