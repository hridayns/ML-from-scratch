import numpy as np
import pandas as pd

class Leaf:
	def __init__(self,prediction):
		self.prediction = prediction

class DecisionNode:
	def __init__(self,question,true_branch,false_branch):
		self.question = question
		self.true_branch = true_branch
		self.false_branch = false_branch

class Question:
	def __init__(self,col,val):
		self.col = col
		self.val = val
	def check(self,row):
		row_val = row[self.col]
		return row_val >= self.val
	def show(self):
		print('Is ',cols[self.col], ' >= ', self.val)

def class_counts(data):
	counts = {}
	if(data.ndim == 1):
		uniq_Lbs, uniq_cts = np.unique(data[-1], return_counts=True)
	else:
		uniq_Lbs, uniq_cts = np.unique(data[:,-1], return_counts=True)
	return uniq_Lbs,uniq_cts

def class_err(data):
	labels,counts = class_counts(data)
	if(len(labels) == 1):
		return 0
	return np.divide(np.min(counts),np.sum(counts))

def partition(data,question):
	true_rows = data[data[:,question.col] >= question.val]
	false_rows = data[data[:,question.col] < question.val]
	return true_rows,false_rows


def find_best_split(data):
	leastErr = 1
	bestQuestion = None
	n = data.shape[1] - 1
	m = data.shape[0]
	for col in range(n):
		d = np.sort(data[:,col],axis = 0)
		if(d.shape[0] == 1):
			values = d
		else:
			first_vals = d[0:len(d)-1]
			second_vals = d[1:len(d)]
			values = np.divide(np.add(first_vals,second_vals),2)
		for val in values:
			question = Question(col,val)

			true_rows,false_rows = partition(data,question)
			
			if(true_rows.shape[0] == 0 or false_rows.shape[0] == 0):
				continue

			true_err = class_err(true_rows)
			false_err = class_err(false_rows)
			err = true_err + false_err

			if(err < leastErr):
				leastErr = err
				bestQuestion = question
	return bestQuestion,leastErr

def buildTree(data):
	labels,counts = class_counts(data)
	p = labels[np.argmax(counts)]
	question,err = find_best_split(data)
	if err == 0 or question is None:
		return Leaf(p)
	true_rows,false_rows = partition(data,question)
	true_branch = buildTree(true_rows)
	false_branch = buildTree(false_rows)

	return DecisionNode(question,true_branch,false_branch)

def printTree(node,spacing=''):
	if(isinstance(node,Leaf)):
		print(spacing + 'Predict',node.prediction)
		return
	node.question.show()
	print (spacing + '--> True:')
	printTree(node.true_branch, spacing + ' ')
	print (spacing + '--> False:')
	printTree(node.false_branch, spacing + ' ')

def classify(data, node):
    if isinstance(node, Leaf):
        return node.prediction
    if node.question.check(data):
        return classify(data, node.true_branch)
    else:
        return classify(data, node.false_branch)

cols = ['variance','skewness','kurtosis','entropy','class']

df = pd.read_csv('data_banknote_authentication.csv',header=None,names=cols)
df = df.sample(frac=1,random_state=43)

train_size = int(df.shape[0] * 0.7)
test_size = int(df.shape[0] * 0.3)

[train_data,test_data] = [df[:train_size],df[train_size:]]

sample_train = np.array(train_data)
sample_test = np.array(test_data)

my_tree = buildTree(sample_train)
count = 0

print('actual',' <---> ','predicted')
for row in sample_test:
	prediction = classify(row, my_tree)
	if(row[-1] == prediction):
		count = count + 1
		print(row[-1],' <---> ', prediction)
print('accuracy: ',count/sample_test.shape[0] * 100.0)
