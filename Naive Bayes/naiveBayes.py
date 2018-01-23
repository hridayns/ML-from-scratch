import pandas as pd
import numpy as np

def class_probs(data):
	m = data.shape[0]
	class_prbs = {}
	uniq_Lbs, uniq_cts = np.unique(data[:,-1], return_counts=True)
	uniq_prbs = np.divide(uniq_cts,m)
	for lb in range(len(uniq_Lbs)):
		class_prbs[lb] = uniq_prbs[lb]
	return class_prbs

def unique_classes(data):
	uniq_Lbs, uniq_cts = np.unique(data[:,-1], return_counts=True)
	return uniq_Lbs

def calc_gauss_prob(data,mu,sdev):
	x_minus_mean_by_std = np.divide( np.subtract( data , np.tile(mu,(data.shape[0],1)) ), np.tile(sdev,(data.shape[0],1)) )
	exponent = np.exp((-1/2) * np.power(x_minus_mean_by_std,2))
	gp = np.prod( np.divide( exponent,np.tile(sdev,(exponent.shape[0],1)) * np.sqrt(2*np.pi) ),axis=1,keepdims=True )
	return gp

def predict(sample_train,sample_test):
	summs = getSummaries(sample_train)
	class_prbs = class_probs(sample_train)
	X = sample_test[:,:-1]
	prbs = {}
	p = np.zeros((sample_test.shape[0],1))
	for cl in summs:
		p_of_X_given_cl = calc_gauss_prob(X,summs[cl]['mean'],summs[cl]['std'])
		p_of_cl = class_prbs[cl]
		ptemp = np.multiply(p_of_cl,p_of_X_given_cl)
		prbs[cl] = ptemp
	for i in range(sample_test.shape[0]):
		if(prbs[1][i] >= prbs[0][i]):
			p[i] = 1
		else:
			p[i] = 0
	return p


def getSummaries(data):
	classes = unique_classes(data)
	summs = {}
	for cl in classes:
		mu = np.mean(data[data[:,-1] == cl,:-1],axis=0)
		sdv = np.std(data[data[:,-1] == cl,:-1],axis=0)
		summs[cl] = {'mean': mu,'std': sdv}
	return summs

cols = ['No. of times pregnant','Plasma glucose concentration','Diastolic blood pressure (mm Hg)','Triceps skin fold thickness (mm)','Serum Insulin','BMI (kg/m^2)','Diabetes pedigree function','Age (yrs)','class']

df = pd.read_csv('pima-indians-diabetes.csv',header=None,names=cols)

# filling in missing values
for col in df.drop(['class'],1):
	mu = np.mean(np.array(df[col]))
	df[col] = df[col].replace(0,mu)
	# df[col] = df[col].dropna()

#feature normalization
for col in df.drop(['class'],1):
	mu = np.mean(np.array(df[col]))
	sd = np.std(np.array(df[col]))
	df[col] = np.divide(np.subtract(np.array(df[col]),mu),sd)

train_size = int(df.shape[0]*0.7)
test_size = int(df.shape[0]*0.3)

[train_data,test_data] = [df[:train_size],df[train_size:]]

sample_train = np.array(train_data)
sample_test = np.array(test_data)

predictions = predict(sample_train,sample_test)
print('Accuracy: ',np.mean(predictions.T == sample_test[:,-1]) * 100)
