import math
import numpy as np
import pandas as pd
import operator

def euclideanDistance(row1,row2,n):
	distance = 0
	for x in range(n):
		distance += pow((row1[x] - row2[x]),2)
	return math.sqrt(distance)

cols = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)','class']

dataset = pd.read_csv('iris.data',header=None,names=cols)

train_size = int(dataset.shape[0] * 0.7)
test_size = int(dataset.shape[0] * 0.3)

[train_data,test_data] = [dataset[:train_size],dataset[train_size:]]

m = train_data.shape[0]
n = train_data.shape[1]

predictions = []

k = 3

distances = []
for i in range(test_size):
	for x in range(m):
		dist = euclideanDistance(list(test_data.iloc[i]),list(train_data.iloc[x]),n-1)
		distances.append((list(train_data.iloc[x]),dist))

	distances.sort(key=operator.itemgetter(1))
	neighbors = []

	for x in range(k):
		neighbors.append(distances[x][0])

	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1

	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

	predictions.append(sortedVotes[0][0])

correct = 0

for x in range(test_size):
	if(test_data.iloc[x][-1] == predictions[x]):
		correct += 1

acc = correct/float(test_size)*100
print('accuracy: ', acc)