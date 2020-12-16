# Code by Jani Toiminen, LinkedIn: https://www.linkedin.com/in/jani-toiminen-9306a53/ , send me a msg if you liked it
# If you want to test with your own data, search for word "was" in this code to find the variables that need to be tuned according to your data

import numpy as np
import tensorflow as tf
import csv
import datetime
import pandas as pd
import time
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

def to_integer(dt_time):
	return time.mktime(time.strptime(dt_time, '%d.%m.%Y'))

labels = \
np.array(["Food and drinks indoors", \
"Food and drinks outdoors", \
"Living", \
"Clothing", \
"Health", \
"Traffic", \
"Phone, internet, pay-TV etc.", \
"Daycare", \
"Insurance", \
"Home appliances", \
"Refreshment, leisure-time", \
"Own savings", \
"Loan expences", \
"Other expences", \
"Own transactions (neg)", \
"Taxes", \
"Credit card payments", \
"Salary", \
"Child benefit", \
"Other income", \
"Own transactions (pos)"])

train_set = pd.read_csv("D:\Jani\Projects\AccountAnalyzer\AccountTransactions.csv", sep = ';', encoding='iso8859_10',header=None, decimal=",",
						names = ('index', 'name', 'date1', 'date2', 'date3', 'amount', 'receiver_payer', 'account_number', 'transaction', 'message'),
						dtype={'amount': 'float', 'message': 'string', 'date1': 'string', 'date2': 'string', 'date3': 'string'},
						na_filter=False)

dict1 = train_set['receiver_payer'].to_dict()
cv = CountVectorizer(dict1.values())
count_vector=cv.fit_transform(dict1.values())

dict2 = train_set['account_number'].to_dict()
cv2 = CountVectorizer(dict2.values())
count_vector2=cv2.fit_transform(dict2.values())

dict3 = train_set['transaction'].to_dict()
cv3 = CountVectorizer(dict3.values())
count_vector3=cv3.fit_transform(dict3.values())

dict4 = train_set['message'].to_dict()
cv4 = CountVectorizer(dict4.values())
count_vector4=cv4.fit_transform(dict4.values())

mylen = np.vectorize(len)
mydateint = np.vectorize(to_integer)
Y = np.array([[train_set['index']]]).T.flatten()
float_times = mydateint(train_set['date1'])
min_float_time = min(float_times)
X = np.c_[np.array(train_set['amount']).T / max(train_set['amount']), \
		  (mydateint(train_set['date1']) - min_float_time) / max((mydateint(train_set['date1']) - min_float_time)), \
		  np.array(count_vector.toarray()), \
		  np.array(count_vector2.toarray()), \
		  np.array(count_vector3.toarray()), \
		  np.array(count_vector4.toarray())]#, \
# original length idea that didn't do much above 65% in accuracy
#		  mylen(train_set['receiver_payer']) / max(mylen(train_set['receiver_payer'])), \
#		  mylen(train_set['account_number']) / max(mylen(train_set['account_number'])), \
#		  mylen(train_set['transaction']) / max(mylen(train_set['transaction'])), \
#		  mylen(train_set['message']) / max(mylen(train_set['message']))]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

def model(hidden_layer_size1, hidden_layer_size2, hidden_layer_size3):#, hidden_layer_size4):
	model = Sequential()
	model.add(Dense(hidden_layer_size1, input_dim=13, activation='relu')) #was 1249
	model.add(Dense(hidden_layer_size2, activation='relu'))
	model.add(Dense(hidden_layer_size3, activation='relu'))
#	model.add(Dense(hidden_layer_size4, activation='relu'))
	model.add(Dense(3, activation='softmax')) #was 20
#	opt = tf.keras.optimizers.RMSprop(learning_rate=0.002) #RMSprop 0.002 ALTERNATIVE
	opt = tf.keras.optimizers.Adam(learning_rate=0.005) #Adam 0.005
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) #adam #squared_hinge
#	print(model.summary())
	return model

resultDict = []

es = 100 #start
ee = 100 #end
ed = 10 #delta
bs = 3 #start  #was 1628
be = 3 #end
bd = 9 #delta
ns = 2 #start  #10 was normally used
ne = 2 #end
nd = 5 #delta
h1s = 600 #start
h1e = 600 #end
h1d = 150 #delta
h2s = 150 #start
h2e = 150 #end
h2d = 200 #delta
h3s = 40 #start
h3e = 40 #end
h3d = 25 #delta
h4s = 20 #start
h4e = 20 #end
h4d = 2 #delta
total_rounds = ((ee - es) / ed + 1)  * ((be - bs) / bd + 1) * ((ne - ns) / nd + 1) * ((h1e - h1s) / h1d + 1) * ((h2e - h2s) / h2d + 1) * ((h3e - h3s) / h3d + 1) * ((h4e - h4s) / h4d + 1)

round = 0
total_time_elapsed = datetime.datetime.now() - datetime.datetime.now()
for epochs in range(es, ee + 1, ed):
	for batch_size in range(bs, be + 1, bd):
		for n_splits in range(ns, ne + 1, nd):
			for hidden_layer_size1 in range(h1s, h1e + 1, h1d):
				for hidden_layer_size2 in range(h2s, h2e + 1, h2d):
					for hidden_layer_size3 in range(h3s, h3e + 1, h3d):
#						for hidden_layer_size4 in range(h4s, h4e + 1, h4d):
						start_time = datetime.datetime.now()
						estimator = KerasClassifier(build_fn=model, hidden_layer_size1=hidden_layer_size1, hidden_layer_size2=hidden_layer_size2, hidden_layer_size3=hidden_layer_size3, epochs=epochs, batch_size=batch_size, verbose=2)
						kfold = KFold(n_splits=n_splits, shuffle=True)
						results = cross_val_score(estimator, X, dummy_y, cv=kfold)
						print(results)
						#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
						time_elapsed = datetime.datetime.now() - start_time
						#", hlayer3 = " + str(hidden_layer_size3) + ", hlayer4 = " + str(hidden_layer_size4) +
						resultStr = "{:.2%}".format(results.mean()) + ", hlayer1 = " + str(hidden_layer_size1) + ", hlayer2 = " + \
							str(hidden_layer_size2) + ", hlayer3 = " + str(hidden_layer_size3) + \
							", epochs=" + str(epochs) + \
							", batch_size=" + str(batch_size) + ", n_splits=" + str(n_splits) + " <- in " + str(time_elapsed)
						resultDict.append(resultStr)
						round += 1
						total_time_elapsed += time_elapsed
						progStr = "Progress = {:.2%} in total ".format(round / total_rounds) + str(total_time_elapsed) + \
							", estimated total duration = " + str(total_time_elapsed * total_rounds / round)
						print(progStr)

print()
for key in resultDict:
    print(key)
print("END")
