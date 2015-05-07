import sys, os
import pickle
import numpy as np
import json, random

from core.datasets.config import TASKS, FEATURE_VECTOR_SAVEFILE, PRETRAIN_VECTOR_MODEL, FEATURE_DIM, DATA_DIR
from core.datasets.entity import RecursiveNNDataInstance, RecursiveNNTreeNode, RecursiveNNTree

def print_node(node, indent=""):
	if node.is_leaf():
		print "%s[%s] - %s" %(indent, node.type, node.term)
	else:
		print "%s[%s]" %(indent, node.type)
		for child in node.children:
			print_node(child, indent + "  ")

class GrammarTree:

	def __init__(self, root):
		self.root = root

	def print_tree(self):
		print_node(self.root)
		print ""

	def apply_all(self, func):
		self.apply_node(self.root, func, True)

	def apply_node(self, node, func, recursive=False, order="pre"):
		func(node)
		if recursive and not node.is_leaf():
			for child in node.children:
				self.apply_node(child, func, recursive)


class GrammarTreeNode:
	def __init__(self, type=None, term=None, children=None):
		self.term = term
		self.type = type
		self.children = children

		if self.term is not None:
			self.term = self.term.strip().replace('\n', '').replace('\t', '')

		self.term_id = None
		self.type_id = None

	def is_leaf(self):
		return self.term is not None

delimiters = " \n\t"

def match_single_node(string, start_pos):
	"""
	Parse a string from the given start_pos. It only expects a single node, in the format of '(TYPE TERM/NODE)'
	It return a tuple:
	(GrammarTreeNode, end_pos - which is the next position after the parsed node)
	"""
	while start_pos < len(string) and string[start_pos] in delimiters:
		start_pos += 1

	if start_pos >= len(string) or string[start_pos] == ')':
		return None, start_pos

	if string[start_pos] == '(':
		# Follow the pattern of ([type ]child1, child2)
		if string[start_pos + 1] == '(':
			node_type = None
			start_pos = start_pos + 1
		else:
			blank_idx = string.find(' ', start_pos)
			node_type = string[start_pos + 1: blank_idx]
			if not node_type: node_type = None

			start_pos = blank_idx + 1

		children, end_pos = match_nodes(string, start_pos)

		end_pos = end_pos + 1 # Use this right bracket

		assert len(children) > 0
		if len(children) == 0:
			print start_pos, string[start_pos: start_pos + 100]
			sys.exit(-1)

		# If only one child, this node can be compress and use the child directly
		if len(children) == 1:
			if children[0].is_leaf():
				children[0].type = node_type
			return children[0], end_pos

		# If there're more than two children, convert to binary tree manually
		# For intermedia nodes, assign type 'None', meaning they have no specific semantics
		if len(children) > 2:
			# Simple strategy: (1 2 3 4), we make it (((1 2) 3) 4)
			while len(children) > 2:
				[left, right] = children[: 2]
				merged_node = GrammarTreeNode(type=None, children=[left, right])
				children = [merged_node] + children[2: ]

		return GrammarTreeNode(type=node_type, children=children), end_pos

	else:
		end_pos = start_pos + 1
		while string[end_pos] not in "()": end_pos += 1
		term = string[start_pos: end_pos]
		return GrammarTreeNode(term=term), end_pos

def match_nodes(string, start_pos):
	"""
	Parse a string from the given start_pos. It expects to meet one or more complete nodes, in the format of '(TYPE TERM/NODE)'
	It return a tuple:
	(GrammarTreeNodes, end_pos - which is the next position after the parsed node)
	"""
	children = []
	while True:
		while start_pos < len(string) and string[start_pos] in delimiters:
			start_pos += 1

		child, end_pos = match_single_node(string, start_pos)
		if child is None:
			break

		children.append(child)
		start_pos = end_pos

	return children, start_pos

def parse_node(string, start_pos):
	return match_nodes(string, start_pos)

def parse_file(filename):
	"""
	Parse all trees in a given file
	"""
	filelines = list(open(filename))
	# Some files contain header, remove header lines
	idx = 0
	while filelines[idx].startswith("*x*"): idx += 1
	filelines = filelines[idx: ]
	filecontent = ''.join(filelines)

	sentense_nodes, pos = parse_node(filecontent, 0)

	if not pos == len(filecontent):
		print filename#, filecontent
		# for root in sentense_nodes: print_node(root)
		sys.exit(-1)

	trees = [GrammarTree(root) for root in sentense_nodes]

	# Legality check, every node must be either leaf or with two children
	def check(node):
		assert (node.is_leaf() and node.children is None ) or len(node.children) == 2

	for tree in trees: tree.apply_all(check)

	return trees

def collect_data_from_directory(data_dir):
	trees = []
	for dirname, subdirs, filenames in os.walk(data_dir):
		for filename in filenames:
			if not os.path.splitext(filename)[1] in (".prd", ".mrg", ):
				continue

			filefullpath = os.path.join(dirname, filename)
			print filefullpath
			trees += parse_file(filefullpath)
	return trees

def grammar_tree_to_recnn_tree(grammar_tree_node, classnum):
	"""
	Recursively convert a grammar tree to the data instance tree we want
	"""
	category_vector = np.zeros(classnum)
	for i in grammar_tree_node.category_ids: category_vector[i - 1] = 1.0 / len(grammar_tree_node.category_ids)
	recnn_node = RecursiveNNTreeNode(category_ids=grammar_tree_node.category_ids, category_vector=category_vector)

	if grammar_tree_node.is_leaf():
		recnn_node.feature_entry_id = grammar_tree_node.term_id
	else:
		assert len(grammar_tree_node.children) == 2
		recnn_node.left = grammar_tree_to_recnn_tree(grammar_tree_node.children[0], classnum=classnum)
		recnn_node.right = grammar_tree_to_recnn_tree(grammar_tree_node.children[1], classnum=classnum)

	return recnn_node

def build_dataset():
	"""
	Parse raw data and cached the processed result according to the path 
	specified by the config.py
	"""

	term_set = {"terms": {}, "counter": 1}
	task_specfic_trees = {}
	task_related_terms = {}

	def add_to_term_set(term):
		if not term_set["terms"].has_key(term):
			term_set["terms"][term] = term_set["counter"]
			term_set["counter"] += 1

	def apply_node(node, task_name):
		if node.is_leaf():
			term = node.term
			add_to_term_set(term)

			if term not in task_related_terms[task_name]:
				task_related_terms[task_name].add(term)

	print "Parsing grammar trees and collect terms"
	
	for task in TASKS:
		task_name = task['name']
		if task.has_key("data_dir_train") and task.has_key("data_dir_test"):
			data_dir_train = DATA_DIR + "/" + task["data_dir_train"]
			data_dir_test  = DATA_DIR + "/" + task["data_dir_test"]
			task_specfic_trees[task_name] = {
				"train" : collect_data_from_directory(data_dir_train),
				"test"  : collect_data_from_directory(data_dir_test),
			}
		else:
			assert task.has_key("data_dir") and task.has_key("train_test_ratio")
			
			data_dir = DATA_DIR + "/" + task['data_dir']
			trees = collect_data_from_directory(data_dir)
			# random.shuffle(trees)

			split_idx = int(len(trees) * task["train_test_ratio"])
			
			task_specfic_trees[task_name] = {
				"train" : trees[: split_idx],
				"test"  : trees[split_idx: ],
			}

		task_related_terms[task_name] = set()

		# Collect all terms
		for tree in task_specfic_trees[task_name]["train"]: # Only consider training set
			tree.apply_all(lambda node: apply_node(node, task_name))

	# Statistic
	for task_name, ts in task_related_terms.items():
		print "%s: %d terms" %(task_name, len(ts))
	common_count = 0
	for term in term_set["terms"].keys():
		iscommon = True
		for task_name, ts in task_related_terms.items():
			if term not in ts:
				iscommon = False
				break
		if iscommon:
			common_count += 1

	print "Common words: %d" %(common_count)


	print "Term collection and parsing done."
	# Write the term -> id map to a file
	ofs = open(DATA_DIR + "/term-id-map", 'w')
	for term, id in term_set["terms"].items():
		ofs.write("%s %d\n" %(term, id))
	ofs.close()

	# Construct the vector matrix, either by random initialization or load from a pretrain model
	if PRETRAIN_VECTOR_MODEL is not None:
		# Load..
		pass
	else:
		feature_vector_matrix = np.random.randn(term_set["counter"], FEATURE_DIM)

	pickle.dump(feature_vector_matrix, open(DATA_DIR + "/" + FEATURE_VECTOR_SAVEFILE, 'wb'))
	print "Finish saving feature_vector_matrix"

	# Update grammar tree, assign feature entry for leaf node, assign category id for all nodes
	print "Updating grammar tree and save to files respectively"
	for task in TASKS:
		trees_train = task_specfic_trees[task["name"]]["train"]
		trees_test  = task_specfic_trees[task["name"]]["test"]
		trees_all   = trees_train + trees_test

		# Collect all categories and assign an id for each
		categories = set()
		def add_category(node):
			if node.type is not None:
				possibletypes = node.type.split('|')
				
				for idx in range(len(possibletypes)):
					node_type = possibletypes[idx]					
					possibletypes[idx] = node_type

				possibletypes = [t for t in possibletypes if t is not None]
				node.category_ids = possibletypes
				for t in possibletypes:
					if t not in categories:
						categories.add(t)
			else:
				node.category_ids = []

		for tree in trees_all: tree.apply_all(add_category)

		# Assign an id to each category, None is always zero.
		categories = sorted(list(categories))
		category_id_map = {}
		for idx, category in enumerate(categories): category_id_map[category] = idx + 1
		category_id_map[None] = 0

		# Update all tree nodes' type id and leaf nodes' term id
		def update_node(node):
			node.category_ids = [category_id_map[t] for t in node.category_ids]
			if node.is_leaf(): 
				if term_set["terms"].has_key(node.term):
					node.term_id = term_set["terms"][node.term]
				else:
					node.term_id = 0 # The term is not in vocabulary

		for tree in trees_all: tree.apply_all(update_node)

		classnum = len(categories)

		# For debug
		fs = open(DATA_DIR + "/category_" + task["parsed_file_train"], 'w')
		fs.write(json.dumps(category_id_map, indent=4, sort_keys=True))
		fs.close()

		# Write to file
		instances_train = [RecursiveNNDataInstance(RecursiveNNTree(grammar_tree_to_recnn_tree(tree.root, classnum))).dict() for tree in trees_train]
		fs = open(DATA_DIR + "/" + task["parsed_file_train"], 'w')
		for instance in instances_train:
			fs.write(json.dumps(instance))
			fs.write("\n")
		fs.close()

		instances_test  = [RecursiveNNDataInstance(RecursiveNNTree(grammar_tree_to_recnn_tree(tree.root, classnum))).dict() for tree in trees_test]
		fs = open(DATA_DIR + "/" + task["parsed_file_test"], 'w')
		for instance in instances_test:
			fs.write(json.dumps(instance))
			fs.write("\n")
		fs.close()

		
		print "Task %s done" %(task["name"])
