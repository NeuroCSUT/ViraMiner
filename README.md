# Using pre-trained ViraMiner
See "Testing models" sections below, but in short:
1) format your data as "seq_id,sequence,label", where sequences are given in letters "ATCG...", labels are 1 or 0.
2) for using the best model - ViraMiner with pre-trained branches and without fine-tuning:
   python2 predict_only.py --input_file your_test_data.csv --model_path final_ViraMiner/final_ViraMiner_beforeFT.hdf5 > test_output.txt


# Overall workflow for results as described in the article "ViraMiner: Deep Learning for identifying viral genomes in human samples"

## 1) Generating datasets based on metagenomics experiments

1) DNA data for each metacenomic experiment separately is located at data/DNA_data/exp*
2) Dataset including all experiments, shuffled and 80/10/10 split is called fullset and has been provided in the folder
2.1) you can generate new datasets with 80/10/10 split, by using create_dataset.py. Replace what exp numbers to include/exclude in the top of the file. 
   MAKE SURE TO CHANGE OUTPUT NAME so you will not overwrite something.
2.2) you can generate train/val/test sets for leave-one-experiment-out by using create_LOO_set.py. 

---
## 2) Training end-to-end CNN models

1) The main dataset is located in data/DNA_data/fullset . The sets are named fullset_train.csv, fullset_validation.csv and fullset_test.csv
2) There are 3 end-to-end models to choose from (full ViraMiner is trained in steps): pattern_branch.py, frequency_branch.py and ViraMiner_end2end.py (ViraMiner trained end-to-end)
3) An example of how to train a model is given below. Notice for input_path you give dataset prefix only (without "_train.csv"). For other model types, just replace the python file you use.

python2 frequency_branch.py output_folder/output_model_name --input_path data/DNA_data/fullset --epochs 30 --filter_size 8 --layer_sizes 1000 --dropout 0.1 --learning_rate 0.001 --lr_decay None > output_folder/output_logfile.txt

4) In the output log file you have VAL AUROC in the last lines, which is the only metric we care about when selecting models. There is also TEST AUROC a few lines above, but this value should be looked at only for the best model. 

---
## 3) Training ViraMiner
1) Train a Pattern and a Frequency model as described in 3) just above. (or train many of them and select best ones)
2) To merge the two models as branches of ViraMiner and retrain the output layer, do (for example):
  python2 merge_and_retrain.py output_folder/output_model_name --input_path data/DNA_data/fullset **--pattern_model final_pattern/pattern_size1200_filter11_0.0001None_drop0.5.hdf5 --freq_model final_freq/freq_size1000_filter8_0.001None_drop0.1.hdf5 --finetuning True** --epochs 30 --dropout 0.1 --learning_rate 0.001 --lr_decay None > output_folder/output_logfile.txt
3) notice that with "finetuning True", also the best not-finetuned model is still saved separately.
   

---
## 4) Testing models

1) The main dataset is located at data/DNA_data/fullset
2) You just need to tell the python script which model to load, load data and save the predictions (and true labels)
3) IMPORTANT: you need to specify the test_set CSV file, not just the dataset prefix as above ("fullset_test.csv", not "fullset")
4) Usage example 

python2 predict_only.py --input_file data/DNA_data/fullset_test.csv --model_path final_freq/freq_size1000_filter8_0.001None_drop0.1.hdf5 > test_testing.txt

5) This will create two files - predictions and true labels to the same folder as your model

---
## 5) Running the K-mer models

1) data needs to be split into training and test files, named datasetname_train.csv and datasetname_test.csv.
   The data format is "feature1", "feature2", ......, "featureN,"label". Basically the n-mer counts and the virus/not-virus label, all separated 
   with commas. 

2) Run the n-mer_freq.py :

2.1) To train Random Forest model:
   python2 n-mer_freq.py --RF True --input_path data/datasetname > out_folder/output_log_filename.txt 

2.2) To train Logistic Regression model:

   python2 n-mer_freq.py --LReg True --input_path data/datasetname > out_folder/output_log_filename.txt 

2.3) To train Nearest Neighbour model (might be slow):

   python2 n-mer_freq.py --NN True --input_path data/datasetname > out_folder/output_log_filename.txt 


2.4) To train Feedforward network model:

   python2 n-mer_freq.py --nmer n_mer_length --input_path data/datasetname --save_path models/out_model_name > out_folder/output_log_filename.txt 

   where n_mer_length is the length (1,2,3,4 or 5)

---
## 6) Running the baselines on raw sequence

1) You canuse the main dataset as above, located in data/data_jan2019/0N_300/

2) To train a RandomForest run the code, for example:

   python2 baseline_on_sequences.py --input_path data/data_jan2019/0N_300/fullset > out_folder/baseline_on_seq_output.txt 

3) notice the code trains 1) RF on not-one-hot-encoded sequences, 2) RF on one-hot-encoded sequences and 3) logistic regression. Make sure you read the good line in output file.
3.1) you can also try to train k-NearestNeighbor model (uncomment last part), but it takes very very long time.







