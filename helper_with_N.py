import numpy as np
from sklearn.metrics import roc_auc_score
import keras
from keras.callbacks import Callback


PROJECT_NAMES=[]

# processes an entire dataset of DNA strings to onehot
def DNA_to_onehot_dataset(dataset):
  options_onehot = {'A': [1,0,0,0,0],'C' :[0,1,0,0,0], 'G':[0,0,1,0,0] ,'T':[0,0,0,1,0],'N':[0,0,0,0,1]}
  onehot_data = []
  for row in dataset:
    onehot_data.append(map(lambda e: options_onehot[e], row))
  onehot_data = np.array(onehot_data)
  print np.shape(onehot_data)
  return onehot_data 

# processes a DNA string to onehot
def DNA_to_onehot(dna_line):
  options_onehot = {'A': [1,0,0,0,0],'C' :[0,1,0,0,0], 'G':[0,0,1,0,0] ,'T':[0,0,0,1,0],'N':[0,0,0,0,1]}
  onehot_data = map(lambda e: options_onehot[e], dna_line)
  onehot_data = np.array(onehot_data)
  return onehot_data 

# takes in a line from dataset in format "seq_ID,sequence,label"
# removes seq_ID, but remembers unique project names
def process_line(line):
    pieces = line.split(",")
    seq = pieces[1]
    label = int(pieces[2])
    
    proj = pieces[0][pieces[0].find("_"):].rstrip("1234567890")
    if proj not in PROJECT_NAMES:
        PROJECT_NAMES.append(proj)
    return seq, label

# function to take in a dataset and predict for each line
def predict_one_by_one(model, seqs,labels):
    prediction_data=np.empty([len(seqs),4]) # true_class, pred_class, proba, seq_length 
    for i,seq in enumerate(seqs):
      p = model.predict(np.expand_dims(seq,axis=0))
      prediction_data[i,0] = labels[i]
      prediction_data[i,1] = int(p>0.5) 
      prediction_data[i,2] = p
      prediction_data[i,3] = len(seq)
    return prediction_data


# reads in sequences from file, expects them to be 300 bp long,
# yields a batch of them
def generate_batches_from_file(path,batch_size):
    total_counter=0
    for_counter = 0
    submitted_counter = 0
    last_seq = ""
    print "generator uses b_size: ", batch_size

    while 1:
        seqs=[]
        labels=[]
        batch_counter=0 # counts from 0 to batch_size
        
        print "opened file again"
        f = open(path)
        for_counter += 1

        for line in f:
            total_counter += 1
            line = line[:-1] #remove \n
            
            seq, lab = process_line(line) # we remove seq_ID
            assert len(seq)==300, "sequence length should be fixed to 300!, but is "+str(len(seq))+" at line"+str(total_counter)
            
            #just a double-check that we moved to next line
            assert not (seq == last_seq), str(seq)
            last_seq = seq
        
            seqs.append(DNA_to_onehot(seq))
            labels.append(lab)
            batch_counter+=1
            if batch_counter==batch_size:
                submitted_counter += batch_size

                yield (np.array(seqs), np.array(labels))
                batch_counter=0
                seqs=[]
                labels=[]
        print "Did all lines in file ",  submitted_counter, total_counter, for_counter
        print "Unique project names", PROJECT_NAMES

        seqs=[]
        labels=[]
        batch_counter=0
        f.close() # close the file so we can reopen it and loop again.

#############################################################
################## SPECIALIZED AUROC CALLBACKS ##############
#############################################################

# Callback to printout VALIDATION AUROC on each epoch end
class roc_callback(keras.callbacks.Callback):
    """# Arguments
       validation_data: validation data in format (val_x,val_y)
    """
    def __init__(self,validation_data):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]        
    
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):           
        
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)      
        
        print " AUROC on Validation: ", str(round(roc_val,4))
        if roc_val ==0.5: #happens if learning has crashed
          self.model.stop_training = True
          print "ATTENTION: Stopped learning process, becuase learning had in all probability crashed!"
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return   


# The following Callbakc allows to only save best models according to VAL AUROC
# Adapted from default ModelCheckpoint code
class ModelCheckpointAUROC(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        validation_data: (NEWLY ADDED for AUROC calculation - validation
            data in format (val_x,val_y)
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, validation_data, verbose=0,
                 save_best_only=False, save_weights_only=False,
                 period=1):
        super(ModelCheckpointAUROC, self).__init__()
        
        #added these two
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0


        # AUROC - the more the better
        self.monitor_op = np.greater
        self.best = -np.Inf


    def on_epoch_end(self, epoch, logs=None):
        #calculate AUROC 
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)

        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = roc_val
                if current is None:
                    warnings.warn('Can save best model. Current area_under_ROC is None.'
                                  'skipping.', RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, "AUROC", self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f (current: %0.5f)' %
                                  (epoch + 1, "AUROC", self.best, current))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


# The following Callback allows to stop learning if VAL AUROC is not improving
# Adapted from default EarlyStopping Code
class EarlyStoppingAUROC(Callback):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        validation_data: (NEWLY ADDED for AUROC calculation) validation
            data in format (val_x,val_y)        
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """
    # added validation data as input to the callback
    def __init__(self,validation_data,
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStoppingAUROC, self).__init__()

        self.x_val = validation_data[0]
        self.y_val = validation_data[1]  

        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        # Can hardcode these as metric is defined to be AUROC
        self.monitor_op = np.greater
        self.min_delta *= 1


    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # Added AUROC calculation
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)

        current = roc_val # everything below is unchanged
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


