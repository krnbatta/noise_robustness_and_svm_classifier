
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
import csv
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.svm import SVC


# In[11]:

instances = 6400
file0=open("assign_5_data_0.txt",'r')
x0 = np.ndarray(shape=(instances,2), dtype=float, order='F')
y0 = np.ndarray(shape=(instances,1), dtype=float, order='F')
lineCount=0
for line in file0:
    if lineCount < instances and line is not "\n":
        perRowData=line.split(',')
        x0[lineCount][0] = float(perRowData[0])
        x0[lineCount][1] = float(perRowData[1])
        y0[lineCount][0] = float(perRowData[2])
        lineCount = lineCount + 1
    
file1=open("assign_5_data_10.txt",'r')
x1 = np.ndarray(shape=(instances,2), dtype=float, order='F')
y1 = np.ndarray(shape=(instances,1), dtype=float, order='F')
lineCount=0
for line in file1:
    if lineCount < instances and line is not "\n":
        perRowData=line.split(',')
        #print(perRowData)
        x1[lineCount][0] = float(perRowData[0])
        x1[lineCount][1] = float(perRowData[1])
        y1[lineCount][0] = float(perRowData[2])
        lineCount = lineCount + 1
    
file2=open("assign_5_data_20.txt",'r')
x2 = np.ndarray(shape=(instances,2), dtype=float, order='F')
y2 = np.ndarray(shape=(instances,1), dtype=float, order='F')
lineCount=0
for line in file2:
    if lineCount < instances and line is not "\n":
        perRowData=line.split(',')
        x2[lineCount][0] = float(perRowData[0])
        x2[lineCount][1] = float(perRowData[1])
        y2[lineCount][0] = float(perRowData[2])
        lineCount = lineCount + 1

file4=open("assign_5_data_40.txt",'r')
x4 = np.ndarray(shape=(instances,2), dtype=float, order='F')
y4 = np.ndarray(shape=(instances,1), dtype=float, order='F')
lineCount=0
for line in file4:
    if lineCount < instances and line is not "\n":
        perRowData=line.split(',')
        x4[lineCount][0] = float(perRowData[0])
        x4[lineCount][1] = float(perRowData[1])
        y4[lineCount][0] = float(perRowData[2])
        lineCount = lineCount + 1
    
file6=open("assign_5_data_60.txt",'r')
x6 = np.ndarray(shape=(instances,2), dtype=float, order='F')
y6 = np.ndarray(shape=(instances,1), dtype=float, order='F')
lineCount=0
for line in file6:
    if lineCount < instances and line is not "\n":
	#print(perRowData)
        perRowData=line.split(',')
        x6[lineCount][0] = float(perRowData[0])
        x6[lineCount][1] = float(perRowData[1])
        y6[lineCount][0] = float(perRowData[2])
        lineCount = lineCount + 1


# In[12]:

rand=np.random.choice(instances,instances,replace=False)
trainX0 = x0[rand][0:5120]
trainY0 = y0[rand][0:5120]
testX0 = x0[rand][5120:6400]
testY0 = y0[rand][5120:6400]
trainY0_nn = np_utils.to_categorical(trainY0)
testY0_nn = np_utils.to_categorical(testY0)

trainX1 = x1[rand][0:5120]
trainY1 = y1[rand][0:5120]
testX1 = x1[rand][5120:6400]
testY1 = y1[rand][5120:6400]
trainY1_nn = np_utils.to_categorical(trainY1)
testY1_nn = np_utils.to_categorical(testY1)

trainX2 = x2[rand][0:5120]
trainY2 = y2[rand][0:5120]
testX2 = x2[rand][5120:6400]
testY2 = y2[rand][5120:6400]
trainY2_nn = np_utils.to_categorical(trainY2)
testY2_nn = np_utils.to_categorical(testY2)

trainX4 = x4[rand][0:5120]
trainY4 = y4[rand][0:5120]
testX4 = x4[rand][5120:6400]
testY4 = y4[rand][5120:6400]
trainY4_nn = np_utils.to_categorical(trainY4)
testY4_nn = np_utils.to_categorical(testY4)

trainX6 = x6[rand][0:5120]
trainY6 = y6[rand][0:5120]
testX6 = x6[rand][5120:6400]
testY6 = y6[rand][5120:6400]
trainY6_nn = np_utils.to_categorical(trainY6)
testY6_nn = np_utils.to_categorical(testY6)


# In[ ]:

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=2, activation='sigmoid'))
	model.add(Dense(4, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# In[ ]:

model0 = baseline_model()
# Fit the model
model0.fit(trainX0, trainY0_nn, validation_data=(testX0, testY0_nn), batch_size=5120, verbose=2, epochs = 20000)
# Final evaluation of the model
scores0 = model0.evaluate(testX0, testY0_nn, verbose=0)


model1 = baseline_model()
# Fit the model
model1.fit(trainX1, trainY1_nn, validation_data=(testX1, testY1_nn), batch_size=5120, verbose=2, epochs = 20000)
# Final evaluation of the model
scores1 = model1.evaluate(testX1, testY1_nn, verbose=0)
scores1clean = model1.evaluate(testX0, testY0_nn, verbose=0)

model2 = baseline_model()
# Fit the model
model2.fit(trainX2, trainY2_nn, validation_data=(testX2, testY2_nn), batch_size=5120, verbose=2, epochs = 20000)
# Final evaluation of the model
scores2 = model2.evaluate(testX2, testY2_nn, verbose=0)
scores2clean = model2.evaluate(testX0, testY0_nn, verbose=0)

model4 = baseline_model()
# Fit the model
model4.fit(trainX4, trainY4_nn, validation_data=(testX4, testY4_nn), batch_size=5120, verbose=2, epochs = 20000)
# Final evaluation of the model
scores4 = model4.evaluate(testX4, testY4_nn, verbose=0)
scores4clean = model4.evaluate(testX0, testY0_nn, verbose=0)

model6 = baseline_model()
# Fit the model
model6.fit(trainX6, trainY6_nn, validation_data=(testX6, testY6_nn), batch_size=5120, verbose=2, epochs = 20000)
# Final evaluation of the model
scores6 = model6.evaluate(testX6, testY6_nn, verbose=0)
scores6clean = model6.evaluate(testX0, testY0_nn, verbose=0)

def baseline_model2():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=2, activation='sigmoid'))
	model.add(Dense(8, input_dim=8, activation='sigmoid'))
	model.add(Dense(4, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# In[ ]:

model01 = baseline_model2()
# Fit the model
model01.fit(trainX0, trainY0_nn, validation_data=(testX0, testY0_nn), batch_size=5120, verbose=2, epochs = 20000)
# Final evaluation of the model
scores01 = model01.evaluate(testX0, testY0_nn, verbose=0)


model11 = baseline_model2()
# Fit the model
model11.fit(trainX1, trainY1_nn, validation_data=(testX1, testY1_nn), batch_size=5120, verbose=2, epochs = 20000)
# Final evaluation of the model
scores11 = model11.evaluate(testX1, testY1_nn, verbose=0)
scores11clean = model11.evaluate(testX0, testY0_nn, verbose=0)

model21 = baseline_model2()
# Fit the model
model21.fit(trainX2, trainY2_nn, validation_data=(testX2, testY2_nn), batch_size=5120, verbose=2, epochs = 20000)
# Final evaluation of the model
scores21 = model21.evaluate(testX2, testY2_nn, verbose=0)
scores21clean = model21.evaluate(testX0, testY0_nn, verbose=0)

model41 = baseline_model2()
# Fit the model
model41.fit(trainX4, trainY4_nn, validation_data=(testX4, testY4_nn), batch_size=5120, verbose=2, epochs = 20000)
# Final evaluation of the model
scores41 = model41.evaluate(testX4, testY4_nn, verbose=0)
scores41clean = model41.evaluate(testX0, testY0_nn, verbose=0)


model61 = baseline_model2()
# Fit the model
model61.fit(trainX6, trainY6_nn, validation_data=(testX6, testY6_nn), batch_size=5120, verbose=2, epochs = 20000)
# Final evaluation of the model
scores61 = model61.evaluate(testX6, testY6_nn, verbose=0)
scores61clean = model61.evaluate(testX0, testY0_nn, verbose=0)

print("[0 NN1]Clean Test Accuracy: %.5f%%" % (scores0[1]))
print("[1 NN1]Noisy Test Accuracy: %.5f%%" % (scores1[1]))
print("[2 NN1]Noisy Test Accuracy: %.5f%%" % (scores2[1]))
print("[4 NN1]Noisy Test Accuracy: %.5f%%" % (scores4[1]))
print("[6 NN1]Noisy Test Accuracy: %.5f%%" % (scores6[1]))
print("[0 NN2]Clean Test Accuracy: %.5f%%" % (scores01[1]))
print("[1 NN2]Noisy Test Accuracy: %.5f%%" % (scores11[1]))
print("[2 NN2]Noisy Test Accuracy: %.5f%%" % (scores21[1]))
print("[4 NN2]Noisy Test Accuracy: %.5f%%" % (scores41[1]))
print("[6 NN2]Noisy Test Accuracy: %.5f%%" % (scores61[1]))

print("[1 NN1]Clean Test Accuracy: %.5f%%" % (scores1clean[1]))
print("[2 NN1]Clean Test Accuracy: %.5f%%" % (scores2clean[1]))
print("[4 NN1]Clean Test Accuracy: %.5f%%" % (scores4clean[1]))
print("[6 NN1]Clean Test Accuracy: %.5f%%" % (scores6clean[1]))
print("[1 NN2]Clean Test Accuracy: %.5f%%" % (scores11clean[1]))
print("[2 NN2]Clean Test Accuracy: %.5f%%" % (scores21clean[1]))
print("[4 NN2]Clean Test Accuracy: %.5f%%" % (scores41clean[1]))
print("[6 NN2]Clean Test Accuracy: %.5f%%" % (scores61clean[1]))

svc_rbf0 = SVC(kernel='rbf', C=1000)
svc_rbf0.fit(trainX0, trainY0)
print("[0-RBF]Train Accuracy: %.5f" % svc_rbf0.score(trainX0, trainY0))
print("[0-RBF]Test Accuracy: %.5f" % svc_rbf0.score(testX0, testY0))

svc_rbf1 = SVC(kernel='rbf', C=1000)
svc_rbf1.fit(trainX1, trainY1)
print("[1-RBF]Noisy Train Accuracy: %.5f" % svc_rbf1.score(trainX1, trainY1))
print("[1-RBF]Noisy Test Accuracy: %.5f" % svc_rbf1.score(testX1, testY1))
print("[1-RBF]Clean Test Accuracy: %.5f" % svc_rbf1.score(testX0, testY0))

svc_rbf2 = SVC(kernel='rbf', C=1000)
svc_rbf2.fit(trainX2, trainY2)
print("[2-RBF]Noisy Train Accuracy: %.5f" % svc_rbf2.score(trainX2, trainY2))
print("[2-RBF]Noisy Test Accuracy: %.5f" % svc_rbf2.score(testX2, testY2))
print("[2-RBF]Clean Test Accuracy: %.5f" % svc_rbf2.score(testX0, testY0))

svc_rbf4 = SVC(kernel='rbf', C=1000)
svc_rbf4.fit(trainX4, trainY4)
print("[4-RBF]Noisy Train Accuracy: %.5f" % svc_rbf4.score(trainX4, trainY4))
print("[4-RBF]Noisy Test Accuracy: %.5f" % svc_rbf4.score(testX4, testY4))
print("[4-RBF]Clean Test Accuracy: %.5f" % svc_rbf4.score(testX0, testY0))

svc_rbf6 = SVC(kernel='rbf', C=1000)
svc_rbf6.fit(trainX6, trainY6)
print("[6-RBF]Noisy Train Accuracy: %.5f" % svc_rbf6.score(trainX6, trainY6))
print("[6-RBF]Noisy Test Accuracy: %.5f" % svc_rbf6.score(testX6, testY6))
print("[6-RBF]Clean Test Accuracy: %.5f" % svc_rbf6.score(testX0, testY0))
