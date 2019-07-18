import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv1D, concatenate, Dropout
from keras.layers import GlobalAveragePooling1D
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam

from sklearn.metrics import roc_auc_score

from helper_with_N import *
import argparse

# You should also change hardcoded values in helper_with_N if you change this
sequence_length = 300

################################
##Read in the parameter values## 
################################
parser = argparse.ArgumentParser()
parser.add_argument("save_path")  # model save name and location
parser.add_argument("--input_path", default = "data/NN_data_clustered")  # data location
parser.add_argument("--filter_size", type=int, default=5)  # with this you can scan other filter sizes
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--layer_sizes", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--lr_decay", type=str, default="None", choices=["None","decreasing"])
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=128)

args = parser.parse_args()

#function to reduce LR
def lrate_decay(epoch):
	initial_lrate = args.learning_rate
	drop = 0.5
	epochs_drop = 5.0
	lrate = initial_lrate * np.power(drop, int((1+epoch)/epochs_drop))
	lrate = np.max([initial_lrate/100, lrate])
	print "lr decay called: new lr ",lrate
	return lrate


#################################################
# Automatically checking train set size without reading it in (might be too big to read)
#################################################
from subprocess import check_output

def wc(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])

train_set_size = wc(args.input_path+"_train.csv")
print "train_set_size: ",train_set_size

# nr of steps generator needs to make per epoch
tr_steps_per_ep = int(train_set_size/args.batch_size)

#######################
# read in test and validation data, train data will be read with a "generator"
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


# Build a model with average pooling
inputs = Input(shape=(sequence_length,5)) # "5" because we have "ATGCN"

first_freq = Conv1D(args.layer_sizes,args.filter_size, activation="relu")(inputs)
freq_pooling = GlobalAveragePooling1D()(first_freq) #returns one value per filter
drop_freq = Dropout(args.dropout)(freq_pooling)

fc_layer1 = Dense(args.layer_sizes, activation="relu", name="freq_fc_layer1")(drop_freq)
drop_fc = Dropout(args.dropout, name="drop_fc1")(fc_layer1)

final = Dense(1,activation="sigmoid")(drop_fc)


# model is defined by defining the inputs and outputs 
# (the graph that connects them is already created above)
model = Model(inputs, final)
model.compile(optimizer = Adam(lr=args.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

###############################
####Fitting the model##########
###############################

callbacks=[]
# custom callbacks are defined in helper_with_N
callbacks.append(ModelCheckpointAUROC(filepath=args.save_path+".hdf5", validation_data=(val_seqs,val_labels), verbose=1, save_best_only=True))
callbacks.append(EarlyStoppingAUROC(validation_data=(val_seqs,val_labels),patience=6, min_delta=0.001))
callbacks.append(roc_callback(((val_seqs,val_labels)))) 

if args.lr_decay == "decreasing":
  callbacks.append(LearningRateScheduler(lrate_decay))

# This fitting procedure read and feeds datapoints to the network batch by batch. 
model.fit_generator(generate_batches_from_file(args.input_path+"_train.csv",args.batch_size),
        steps_per_epoch=tr_steps_per_ep, epochs=args.epochs, workers=1, use_multiprocessing=False, callbacks=callbacks,verbose=2, validation_data=(val_seqs,val_labels))




###############################
####Testing the model##########
###############################

# training is done. The params at the end of training are not necessarily the best params. Best ones were saved by ModelCheckpoint.
print "\n ########### Loading the saved model (i.e best model) ###############"
model = load_model(args.save_path+".hdf5")

print "##########################"

pred_probas = model.predict_generator(generate_batches_from_file(args.input_path+"_test.csv",args.batch_size), steps=te_steps_per_ep+1,workers=1, use_multiprocessing=False)
pred_probas = pred_probas[:len(test_labels),:]
print "TEST ROC area under the curve \n", roc_auc_score(test_labels, pred_probas)

pred_probas = model.predict_generator(generate_batches_from_file(args.input_path+"_validation.csv",args.batch_size), steps=val_steps_per_ep+1,workers=1, use_multiprocessing=False)
pred_probas = pred_probas[:len(val_labels),:]
print "VAL ROC area under the curve \n", roc_auc_score(val_labels, pred_probas)
