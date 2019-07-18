import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv1D, concatenate, Dropout
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.optimizers import Adam, Nadam,SGD

from sklearn.metrics import confusion_matrix,roc_auc_score

from helper_with_N import *

import argparse
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


################################
##Read in the parameter values## 
################################
parser = argparse.ArgumentParser()
parser.add_argument("save_path") #model save name and location
parser.add_argument("--input_path", default = "data/NN_data_clustered") # data location

parser.add_argument("--dropout", type=float, default=0.1) # add dropout? if 0 then no dropout
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--lr_decay",type=str, default="None", choices=["None","decreasing"])
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=128)

args = parser.parse_args()



def lr_decay(epoch):
  initial_lrate=args.learning_rate
  drop=0.5
  epochs_drop=5.0
  lrate = initial_lrate * np.power(drop,int((1+epoch)/epochs_drop))
  lrate=np.max([initial_lrate/100,lrate])
  print "lr_decay called, new lr ", lrate
  return lrate

#################################################
# Automatically checking train and test set sizes!
#################################################
from subprocess import check_output

def wc(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])

train_set_size = wc(args.input_path+"_train.csv")
print "train_set_size,val_size, test_set_size",train_set_size

tr_steps_per_ep = int(train_set_size/args.batch_size)

print "##\n input data: \n ", args.input_path, "\n##"

#######################
# read in test and validation data, train data wil be read with a "generator"
val_seqs = []
val_labels = []
test_seqs=[]
test_labels=[]

f = open(args.input_path+"_validation.csv")
for line in f:
    line = line[:-1] #remove \n
    seq, lab = process_line(line)
    val_seqs.append(DNA_to_onehot(seq))
    val_labels.append(lab)
f.close()
val_set_size = len(val_labels)

f=open(args.input_path+"_test.csv")
for line in f:
  line = line[:-1]
  seq,lab= process_line(line)
  test_seqs.append(DNA_to_onehot(seq))
  test_labels.append(lab)
f.close()
test_set_size = len(test_labels)
  
# nr of steps for generator when validating/testing
val_steps_per_ep = int(val_set_size/args.batch_size)
te_steps_per_ep = int(test_set_size/args.batch_size)
  
val_seqs= np.array(val_seqs)  # put to numpy format
val_labels = np.array(val_labels)  # put to numpy format
test_seqs = np.array(test_seqs)  # put to numpy format
test_labels = np.array(test_labels)  # put to numpy format


###############################
####Defining the model#########
###############################


# Build a model with two branches
inputs = Input(shape=(300,5)) # "None" means any sequence length, "5" because we have "ATGCN"

#first branch - averaging
first_freq = Conv1D(1000,8, activation="relu")(inputs)
freq_pooling = GlobalAveragePooling1D()(first_freq)
drop_freq = Dropout(args.dropout)(freq_pooling)
fc_layer1 = Dense(1000, activation="relu", name="fc_layer1")(drop_freq)

#second branch - maximum
first_pattern = Conv1D(1200,11, activation="relu")(inputs)
pattern_pooling = GlobalMaxPooling1D()(first_pattern)
drop_pattern = Dropout(args.dropout)(pattern_pooling)
fc_layer2 = Dense(1200, activation="relu", name="fc_layer2")(drop_pattern)

#merge the branches
concatenation = concatenate([fc_layer1, fc_layer2])

#add fully connected layers
drop_fc = Dropout(args.dropout, name="drop_fc1")(concatenation)

final = Dense(1,activation="sigmoid")(drop_fc)


# model is defined by defining the inputs and outputs 
# (the graph that connects them is already created above)
model = Model(inputs, final)

model.compile(optimizer = Adam(lr=args.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

###############################
####Fitting the model##########
###############################

#saving the model
callbacks=[]
callbacks.append(ModelCheckpointAUROC(filepath=args.save_path+".hdf5", validation_data=(val_seqs,val_labels), verbose=1, save_best_only=True))
callbacks.append(EarlyStoppingAUROC(validation_data=(val_seqs,val_labels),patience=6, min_delta=0.001))
callbacks.append(roc_callback(((val_seqs,val_labels)))) 

if args.lr_decay == "decreasing":
  callbacks.append(LearningRateScheduler(lr_decay))

# This fitting procedure feeds datapoints to the network one by one
model.fit_generator(generate_batches_from_file(args.input_path+"_train.csv",args.batch_size),
        steps_per_epoch=tr_steps_per_ep, epochs=args.epochs, workers=1, use_multiprocessing=False, callbacks=callbacks,verbose=2,  validation_data=(val_seqs,val_labels)) #steps per epoch is set by the smallest training set size


###############################
####Testing the model##########
###############################
print "\n ########### Loading the saved model (i.e best model) ###############"
model = load_model(args.save_path+".hdf5")

print "##########################"

pred_probas = model.predict_generator(generate_batches_from_file(args.input_path+"_test.csv",args.batch_size), steps=te_steps_per_ep+1,workers=1, use_multiprocessing=False)
pred_probas = pred_probas[:len(test_labels),:]
print "TEST ROC area under the curve \n", roc_auc_score(test_labels, pred_probas)

pred_probas = model.predict_generator(generate_batches_from_file(args.input_path+"_validation.csv",args.batch_size), steps=val_steps_per_ep+1,workers=1, use_multiprocessing=False)
pred_probas = pred_probas[:len(val_labels),:]
print "VAL ROC area under the curve \n", roc_auc_score(val_labels, pred_probas)
