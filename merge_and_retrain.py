import numpy as np

from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense, concatenate, Dropout
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
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
parser.add_argument("--input_path", default = "data/fullset") # data location
parser.add_argument("--freq_model", default = "oct2018/best_frequency.hdf5")  # pretrained frequency model
parser.add_argument("--pattern_model", default = "oct2018/best_pattern.hdf5")  # pretrained pattern model
parser.add_argument("--finetuning",type=str, default="True", choices=["True","False"])

parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--lr_decay",type=str, default="None", choices=["None","decreasing"])
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=30) 


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
inputs = Input(shape=(sequence_length,5)) # "None" means any sequence length, "5" because we have "ATGCN"

# load pretrained models
freq_model = load_model(args.freq_model)
freq_model.name = "freq_model"
pattern_model = load_model(args.pattern_model)
pattern_model.name = "pat_model"


# THIS TURNS OFF TRAINING IN FIRST LAYERS
for layer in freq_model.layers:
    layer.trainable = False
for layer in pattern_model.layers:
    layer.trainable = False

freq_model_cropped = Model(freq_model.inputs, freq_model.layers[-3].output) # crop layers -1 and -2, i.e sigmoid and dropout 
pattern_model_cropped = Model(pattern_model.inputs, pattern_model.layers[-3].output)

freq_model_cropped.summary()
pattern_model_cropped.summary()

# Build a model with two branches
features1 = freq_model_cropped(inputs)
features2 = pattern_model_cropped(inputs)

#merge the branches
concatenation = concatenate([features1, features2])
drop_conc= Dropout(args.dropout)(concatenation)  # add a new dropout layer 
new_final = Dense(1,activation="sigmoid")(drop_conc)  # add a new output node


# model is defined by defining the inputs and outputs 
# (the graph that connects them is already created above)
model = Model(inputs, new_final)

model.compile(optimizer = Adam(lr=args.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

###############################
####Fitting the model##########
###############################

#### Learning the last layer from scratch ####
# Remember that the branches still have trainable=False

# Callbacks to save the model
callbacks=[]
# custom callbacks defined ni helper_with_N
callbacks.append(ModelCheckpointAUROC(filepath=args.save_path+"_beforeFT.hdf5", validation_data=(val_seqs,val_labels), verbose=1, save_best_only=True))
callbacks.append(EarlyStoppingAUROC(validation_data=(val_seqs,val_labels),patience=6, min_delta=0.001))
callbacks.append(roc_callback(((val_seqs,val_labels)))) #defined in helper_with_N

if args.lr_decay == "decreasing":
  callbacks.append(LearningRateScheduler(lr_decay))

# This fitting procedure reads datapoints from file on as-needed basis
model.fit_generator(generate_batches_from_file(args.input_path+"_train.csv",args.batch_size),
        steps_per_epoch=tr_steps_per_ep, epochs=args.epochs, workers=1, use_multiprocessing=False, callbacks=callbacks,verbose=2,  validation_data=(val_seqs,val_labels)) 


#### Fine-tuning the whole network ####
if args.finetuning=="True":
  # THIS TURNS BACK ON TRAINING IN FIRST LAYERS
  for layer in freq_model.layers:
    layer.trainable = True
  for layer in pattern_model.layers:
    layer.trainable = True

  # I guess I need to recompile after setting trainable TRUE?
  model.compile(optimizer = Adam(lr=args.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
  
  callbacks=[]
  callbacks.append(ModelCheckpointAUROC(filepath=args.save_path+"_afterFT.hdf5", validation_data=(val_seqs,val_labels), verbose=1, save_best_only=True))
  callbacks.append(EarlyStoppingAUROC(validation_data=(val_seqs,val_labels),patience=6, min_delta=0.001))
  callbacks.append(roc_callback(((val_seqs,val_labels))))
  if args.lr_decay=="decreasing":
    callbacks.append(LearningRateScheduler(lr_decay))

  # This fitting procedure reads datapoints from file on as-needed basis
  model.fit_generator(generate_batches_from_file(args.input_path+"_train.csv",args.batch_size),
        steps_per_epoch=tr_steps_per_ep, epochs=args.epochs, workers=1, use_multiprocessing=False, callbacks=callbacks,verbose=2,  validation_data=(val_seqs,val_labels)) 


###############################
####Testing the model##########
###############################

#We need to use the best model(that we saved), not the current weights. We reload the model
print "########### ViraMiner without fine-tuning ###############"
model = load_model(args.save_path+"_beforeFT.hdf5")

pred_probas = model.predict_generator(generate_batches_from_file(args.input_path+"_test.csv",args.batch_size), steps=te_steps_per_ep+1,workers=1, use_multiprocessing=False)
pred_probas = pred_probas[:len(test_labels),:]
print "TEST ROC area under the curve \n", roc_auc_score(test_labels, pred_probas)

pred_probas = model.predict_generator(generate_batches_from_file(args.input_path+"_validation.csv",args.batch_size), steps=val_steps_per_ep+1,workers=1, use_multiprocessing=False)
pred_probas = pred_probas[:len(val_labels),:]
print "VAL ROC area under the curve \n", roc_auc_score(val_labels, pred_probas)


print "\n\n ########### ViraMiner with fine-tuning ###############"
model = load_model(args.save_path+"_afterFT.hdf5")

pred_probas = model.predict_generator(generate_batches_from_file(args.input_path+"_test.csv",args.batch_size), steps=te_steps_per_ep+1,workers=1, use_multiprocessing=False)
pred_probas = pred_probas[:len(test_labels),:]
print "TEST ROC area under the curve \n", roc_auc_score(test_labels, pred_probas)

pred_probas = model.predict_generator(generate_batches_from_file(args.input_path+"_validation.csv",args.batch_size), steps=val_steps_per_ep+1,workers=1, use_multiprocessing=False)
pred_probas = pred_probas[:len(val_labels),:]
print "VAL ROC area under the curve \n", roc_auc_score(val_labels, pred_probas)



