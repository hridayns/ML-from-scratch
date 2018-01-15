import pandas as pd
import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(-x))

def costFunc(X,theta,y):
	m = X.shape[0]
	grad = np.zeros(theta.shape)
	h = sigmoid(np.dot(X,theta))
	J = np.sum(np.multiply(y,np.log(h)) + np.multiply(np.subtract(np.ones(y.shape),y),np.log(np.subtract(np.ones(h.shape),h))))
	J = (-1/m) * J

	grad = np.dot(X.T,(np.subtract(h,y)))
	grad = np.multiply((1/m),grad)

	return [J,grad]

def gradientDesc(X,theta,y,alpha,iter_num):
	m = X.shape[0]
	J_vals = np.zeros((iter_num,1))

	for iter in range(iter_num):
		[J_vals[iter],tmp] = costFunc(X,theta,y)
		theta = np.subtract(theta,np.multiply(alpha/m,tmp))

	return [theta,J_vals]


def predict(X,theta):
	h = sigmoid(np.dot(X,theta))
	h[h >= 0.5] = 1
	h[h < 0.5] = 0
	return h

cols = ['No. of times pregnant','Plasma glucose concentration','Diastolic blood pressure (mm Hg)','Triceps skin fold thickness (mm)','Serum Insulin','BMI (kg/m^2)','Diabetes pedigree function','Age (yrs)','class']

df = pd.read_csv('pima-indians-diabetes.csv',header=None,names=cols)

#filling in missing values
for col in df.drop(['class'],1):
	mu = np.mean(np.array(df[col]))
	# df[col] = df[col].replace(0,mu)
	df[col] = df[col].dropna()

#feature normalization
for col in df.drop(['class'],1):
	mu = np.mean(np.array(df[col]))
	sd = np.std(np.array(df[col]))
	df[col] = np.divide(np.subtract(np.array(df[col]),mu),sd)

train_size = int(df.shape[0]*0.7)
test_size = int(df.shape[0]*0.3)

[train_data,test_data] = [df[:train_size],df[train_size:]]


X = np.array(train_data.drop(['class'],1))
X = np.append(np.ones((X.shape[0],1)),X,axis=1)
y = np.array(train_data[['class']])
X_test = np.array(test_data.drop(['class'],1))
X_test = np.append(np.ones((X_test.shape[0],1)),X_test,axis=1)
y_test = np.array(test_data[['class']])


initial_theta = np.zeros((X.shape[1],1))
predictions = np.zeros((y_test.shape[0],1))

alpha = 0.1
iter_num = 100

[Theta,J_values] = gradientDesc(X,initial_theta,y,alpha,iter_num)

predictions = predict(X_test,Theta)

acc = np.mean(predictions == y_test) * 100
print('Accuracy: ',acc)