import numpy as np
import pandas as pd

class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)

class Decision_Node:
	def __init__(self,question,true_branch,false_branch):
		self.question = question
		self.true_branch = true_branch
		self.false_branch = false_branch

train_data = [
	['green',3,'apple'],
	['red',1,'grape'],
	['red',2,'grape'],
	['yellow',3,'lemon']
]

header = ['color','diameter','label']

def uniq_val(rows,col):
	return set([row[col] for row in rows])

# print(uniq_val(train_data,1))

def class_counts(rows):
	counts = {}
	for row in rows:
		label = row[-1]
		if(label not in counts):
			counts[label] = 0
		counts[label] += 1
	return counts

# print(class_counts(train_data))

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

class Question:
	def __init__(self,column,val):
		self.column = column
		self.val = val
	def match(self,ex):
		ex_val = ex[self.column]
		if(is_numeric(ex_val)):
			return ex_val >= self.val
		else:
			return ex_val == self.val

# q = Question(1, 4)
# example = train_data[0]
# print(example)
# print(q.match(example))


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

# print(partition(train_data,Question(0,'red')))

def gini(rows):
	counts = class_counts(rows)
	impurity = 1
	for label in counts:
		prob_of_label = counts[label] / float(len(rows))
		print(label,prob_of_label)
		impurity = impurity - prob_of_label ** 2
	return impurity


# lit_mix = [
# 	['apple'],
# 	['pulao']
# ]

# print(gini(lit_mix))

def info_gain(left,right,current_uncertainty):
	p = float(len(left)) / (len(left) + len(right))
	q = 1 - p
	return current_uncertainty - p * gini(left) - q * gini(right)

# unc = gini(train_data)
# true_rows,false_rows = partition(train_data,Question(0,'green'))
# true_rows,false_rows = partition(train_data,Question(0,'red'))
# print(info_gain(true_rows,false_rows,unc))

def find_best_split(rows):
	best_gain = 0
	best_question = None
	current_uncertainty = gini(rows)
	n_features = len(rows[0]) - 1
	for col in range(n_features):
		values = set([row[col] for row in rows])
		for val in values:
			question = Question(col,val)
			true_rows,false_rows = partition(rows,question)

			if(len(true_rows) == 0 or len(false_rows) == 0):
				continue
			gain = info_gain(true_rows,false_rows,current_uncertainty)
			if(gain > best_gain):
				best_gain,best_question = gain,question

	return best_gain,best_question

# best_gain,best_question = find_best_split(train_data)
# print(best_gain,best_question)

def buildTree(rows):
	gain,question = find_best_split(rows)
	if gain == 0:
		return Leaf(rows)
	true_rows,false_rows = partition(rows,question)

	true_branch = buildTree(true_rows)
	false_branch = buildTree(false_rows)

	return Decision_Node(question,true_branch,false_branch)

def printTree(node,spacing=''):
	if(isinstance(node,Leaf)):
		print(spacing + 'Predict',node.predictions)
		return
	print(spacing + str(node.question))
	print (spacing + '--> True:')
	printTree(node.true_branch, spacing + ' ')
	print (spacing + '--> False:')
	printTree(node.false_branch, spacing + ' ')

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


my_tree = buildTree(train_data)
printTree(my_tree)
testing_data = [
    ['green', 3, 'apple'],
    ['yellow', 4, 'apple'],
    ['red', 2, 'grape'],
    ['red', 1, 'grape'],
    ['yellow', 3, 'lemon'],
]

for row in testing_data:
    print ("Actual: %s. Predicted: %s" %(row[-1], classify(row, my_tree)))