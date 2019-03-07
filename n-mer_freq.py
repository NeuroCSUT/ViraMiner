import numpy as np

from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score

from helper_with_N import *
import itertools
import argparse
import pandas as pd
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


################################
##Read in the parameter values## 
################################
parser = argparse.ArgumentParser()
parser.add_argument("--save_path", default = "models/kmer_FFnet") #model save name and location
parser.add_argument("--input_path", default = "data/cccccccccc") # data location
parser.add_argument("--nmer",type=int, default = 3) # which n-nmers we use
parser.add_argument("--NN",type=str2bool, default = False) # data location
parser.add_argument("--RF",type=str2bool, default = False) # data location
parser.add_argument("--LReg",type=str2bool, default = False) # data location
args = parser.parse_args()


#read in data
train_data = np.loadtxt(args.input_path+"_train.csv",delimiter="\t",dtype=np.uint8)
train_counts = train_data[:,:-1]
train_labels = train_data[:,-1]
del train_data

test_data =  np.loadtxt(args.input_path+"_test.csv",delimiter="\t",dtype=np.uint8)
test_counts = test_data[:,:-1]
test_labels = test_data[:,-1]
del test_data
print "train data", train_counts.shape, "      test data", test_counts.shape



if args.NN:
 from sklearn.neighbors import KNeighborsClassifier
 model = KNeighborsClassifier(n_neighbors=1)
 model.fit(train_counts,train_labels) #neigbour
 knn_preds = model.predict(test_counts)
 print "KNN confusion_matrix\n", confusion_matrix(test_labels, knn_preds)
 print "KNN accuracy", accuracy_score(test_labels, knn_preds)
 assert False #stop here 



###########
if args.RF:
 from sklearn.ensemble import RandomForestClassifier
 model = RandomForestClassifier(n_estimators=1000,n_jobs=4)
 print "starting to fit RF"
 model.fit(train_counts,train_labels) 
 print "done fitting RF"

 
 rf_preds_train = model.predict_proba(train_counts)
 print "Random Forest TRAIN ROC area under the curve \n", roc_auc_score(train_labels, rf_preds_train[:,1]) 

 rf_preds_test = model.predict_proba(test_counts)
 print "Random Forest TEST ROC area under the curve \n", roc_auc_score(test_labels, rf_preds_test[:,1]) 
 np.savetxt("rf_7mers_preds_test.txt", rf_preds_test[:,1], fmt="%.5f")
 np.savetxt("rf_7mers_labels_test.txt", test_labels, fmt="%d")
 assert False #stop here 

############
if args.LReg:
 from sklearn.linear_model import LogisticRegression
 model = LogisticRegression(fit_intercept=True, max_iter=1000,penalty="l1",C=0.01)
 print "Lreg input data", train_counts.shape
 model.fit(train_counts,train_labels)
 
 print "done fitting Log Reg"
 log_preds = model.predict(test_counts)
 print "LogRegr confusion_matrix\n", confusion_matrix(test_labels, log_preds)

 accuracy = model.score(train_counts,train_labels)
 print "overall train accuracy", accuracy

 accuracy = model.score(test_counts, test_labels)
 print "overall test accuracy", accuracy
 
 log_preds_vals = model.predict_proba(test_counts)
 print "LogRegr TEST ROC area under the curve \n", roc_auc_score(test_labels, log_preds_vals[:,1]) # AUC=0.70

 log_preds_vals = model.predict_proba(train_counts)
 print "LogRegr TRAIN ROC area under the curve \n", roc_auc_score(train_labels, log_preds_vals[:,1]) # AUC=0.70

 #prms = model.coef_
 #print prms , "\n and bias/intercept: ", model.intercept_
 #np.savetxt("log_weights_reference_2mers_500.txt",prms,delimiter=',')
 assert False #stop here


###############################
####Defining the model#########
###############################

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, concatenate, Dropout, Flatten, MaxPooling1D,Reshape
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.optimizers import Adam, Nadam, SGD,RMSprop,Adagrad,Adadelta
from keras.initializers import RandomUniform


model = Sequential()
if args.nmer==3:
 model.add(Dense(256,input_shape=(64,),activation="relu"))
elif args.nmer==1:
 model.add(Dense(256,input_shape=(4,),activation="relu"))
elif args.nmer==2:
 model.add(Dense(256,input_shape=(16,),activation="relu"))
elif args.nmer==4:
 model.add(Dense(256,input_shape=(256,),activation="relu"))
elif args.nmer==5:
 model.add(Dense(256,input_shape=(1024,),activation="relu"))
else:
 assert False
model.add(Dense(256,activation="relu"))
model.add(Dense(256,activation="relu"))

model.add(Dense(1,activation="sigmoid", use_bias=True))



model.compile(optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True,clipnorm=0.5), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

###############################
####Fitting the model##########
###############################


#saving the model
callbacks = [ModelCheckpoint(filepath=args.save_path+".hdf5", verbose=1, save_best_only=False)]
callbacks.append(LearningRateScheduler(char_lrate_decay))

# This fitting procedure feeds datapoints to the network one by one
model.fit(train_counts,train_labels,batch_size=100, epochs=200, callbacks=callbacks,verbose=1, validation_data=(test_counts,test_labels)) #steps per epoch is set by the smallest training set size


###############################
####Testing the model##########
###############################


print "TRAIN eval:", model.evaluate(train_counts,train_labels)
print "TEST eval:", model.evaluate(test_counts,test_labels)



print "##########################"
pred_probas = model.predict(test_counts)
print np.shape(pred_probas), type(pred_probas)
preds = pred_probas>0.5

print pred_probas[:10],preds[:10]
print "confusion_matrix\n", confusion_matrix(test_labels, np.array(preds,dtype=int))

print "ROC area under the curve \n", roc_auc_score(test_labels, pred_probas)

#np.savetxt("30N_cls_cnf_predictions.csv",pred_matrix[:,:3],delimiter=',', fmt=["%d","%d","%.3f"])

