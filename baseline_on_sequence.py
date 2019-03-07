import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score

from helper_with_N import *

import argparse
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


#Parameters of the data
sequence_length = 300


################################
##Read in the parameter values## 
################################
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", default = "data/cccccccccc") # data location

args = parser.parse_args()


###############################
#### Read in the data #########
###############################
train_data = pd.read_csv(args.input_path+"_train.csv",delimiter=",",names=["projects","seqs","label","id"],dtype={"projects":str, "seqs":str,"label":np.int,"id":str})
print "read the CSV WITH PANDAS\n"

train_onehot = DNA_to_onehot_dataset(train_data.seqs)
print "Option1: converted to onehot!!\n"
train_numbers = np.argmax(train_onehot,axis=2)
print "Option2: converted to numbers!!\n"
print train_data.seqs[0],type(train_data.seqs),"\n", train_onehot[0], "\n", train_numbers[0]


val_data = pd.read_csv(args.input_path+"_validation.csv",delimiter=",",names=["projects","seqs","label","id"],dtype={"projects":str, "seqs":str,"label":np.int,"id":str})
val_onehot = DNA_to_onehot_dataset(val_data.seqs)
val_numbers = np.argmax(val_onehot,axis=2)
print np.shape(val_data.seqs),np.shape(val_onehot), np.shape(val_numbers)

test_data = pd.read_csv(args.input_path+"_test.csv",delimiter=",",names=["projects","seqs","label","id"],dtype={"projects":str, "seqs":str,"label":np.int,"id":str})
test_onehot = DNA_to_onehot_dataset(test_data.seqs)
test_numbers = np.argmax(test_onehot,axis=2)
print np.shape(test_data.seqs),np.shape(test_onehot), np.shape(test_numbers)



###################################
### RANDOM FOREST on the numeric sequence
###################################

RF_model = RandomForestClassifier(n_estimators=1000, n_jobs=4)
print "\n \n starting to fit RF"
RF_model.fit(train_numbers,train_data.label) # maybe should use flattened onehot?
print "done fitting RF"


rf_preds_train = RF_model.predict_proba(train_numbers)
print "Random Forest TRAIN ROC area under the curve \n", roc_auc_score(train_data.label, rf_preds_train[:,1]) 
rf_preds_val = RF_model.predict_proba(val_numbers)
print "Random Forest VAL ROC area under the curve \n", roc_auc_score(val_data.label, rf_preds_val[:,1]) 
rf_preds_test = RF_model.predict_proba(test_numbers)
print "Random Forest TEST ROC area under the curve \n", roc_auc_score(test_data.label, rf_preds_test[:,1]) 

###################################
### RANDOM FOREST on flattened one-hot
###################################

# reusable stuff
flat_onehot_train = train_onehot.reshape((-1,sequence_length*5))
flat_onehot_val = val_onehot.reshape((-1,sequence_length*5))
flat_onehot_test = test_onehot.reshape((-1,sequence_length*5))
######

RF_model = RandomForestClassifier(n_estimators=1000, n_jobs=4)
print "\n ##########################\n starting to fit RF on onehot"
RF_model.fit(flat_onehot_train,train_data.label) # maybe should use flattened onehot?
print "done fitting OneHot RF"


rf_preds_train = RF_model.predict_proba(flat_onehot_train)
print "OneHot Random Forest TRAIN ROC area under the curve \n", roc_auc_score(train_data.label, rf_preds_train[:,1]) 

rf_preds_val = RF_model.predict_proba(flat_onehot_val)
print "OneHot Random Forest VAL ROC area under the curve \n", roc_auc_score(val_data.label, rf_preds_val[:,1]) 

rf_preds_test = RF_model.predict_proba(flat_onehot_test)
print "OneHot Random Forest TEST ROC area under the curve \n", roc_auc_score(test_data.label, rf_preds_test[:,1]) 

np.savetxt("rf_preds_on_onehot_sequence.txt",rf_preds_test[:,1],fmt="%.5f")
np.savetxt("rf_labels_on_onehot_sequence.txt",test_data.label,fmt="%.5f")


################################
# Logistic Regression - 
# we flatten the one-hot matrix, to get (5 x seq_len) vector, each consecutive 5 features represent a position in sequence and contain one 1 and 4 zeroes 
################################

from sklearn.linear_model import LogisticRegression
print "\n \n starting to fit Logistic regression"

model = LogisticRegression(fit_intercept=True, max_iter=1000)
model.fit(flat_onehot_train,train_data.label) 
print "done fitting Log Reg"

log_preds_train = model.predict_proba(flat_onehot_train)
print "LogRegr TRAIN ROC area under the curve \n", roc_auc_score(train_data.label, log_preds_train[:,1])
log_preds_val = model.predict_proba(flat_onehot_val)
print "LogRegr VAL ROC area under the curve \n", roc_auc_score(val_data.label, log_preds_val[:,1])
log_preds_test = model.predict_proba(flat_onehot_test)
print "LogRegr TEST ROC area under the curve \n", roc_auc_score(test_data.label, log_preds_test[:,1]) 


#prms = model.coef_

#print prms , "\n and bias/intercept: ", model.intercept_
#np.savetxt("logReg_weights_onSeq_1000projectsplit.txt",prms,delimiter=',')


###############################
### Defining the KNN ########
###############################


#oneNN_model = KNeighborsClassifier(n_neighbors=1,metric='hamming')

#with higher K we can do ROC as well (though jumpy)
#model10 = KNeighborsClassifier(n_neighbors=10,metric='hamming',weights="uniform")
#model10 = KNeighborsClassifier(n_neighbors=10,metric='hamming',weights="distance",n_jobs=4)

###############################
### Fitting the KNN #########
###############################

# hamming distance is 0 only if the same value is observed, otherwise 1
# we use sequences transformed into a sequence of numbers . For ex: AACGTN -> 001234

#print np.shape(train_numbers), np.shape(train_data.label)
#oneNN_model.fit(train_numbers,np.array(train_data.label)) 
#knn_preds = oneNN_model.predict(test_numbers)
#print np.shape(train_numbers), np.shape(test_numbers), np.shape(knn_preds)
#print "KNN confusion_matrix\n", confusion_matrix(test_data.label, knn_preds)
#accuracy = accuracy_score(knn_preds,test_data.label)
#print "KNN overall accuracy", accuracy
#print "No ROC score can be calculated for 1NN"

#model10.fit(train_numbers,train_data.label) # 5 neigbours

#KNN_probas_train = model10.predict_proba(train_numbers)
#print "10NN TEST ROC area under the curve \n", roc_auc_score(train_data.label, KNN_probas_train[:,1]) 

#KNN_probas_val = model10.predict_proba(val_numbers)
#print "10NN VAL ROC area under the curve \n", roc_auc_score(val_data.label, KNN_probas_val[:,1]) 

#KNN_probas_test = model10.predict_proba(test_numbers)
#print "10NN TEST ROC area under the curve \n", roc_auc_score(test_data.label, KNN_probas_test[:,1]) 

