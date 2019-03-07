import os

#exps_to_include=[]
exps = [0,8,9,10,11] # only serum

exps_to_exclude = [] # for 80/10/10 on serum 

for ex in exps_to_exclude:
  exps.remove(ex)

exps_string = ""
for ex in exps:
  exps_string += "exp"+str(ex)+"_* "

print exps_string
os.system("cat "+exps_string+" > temp.csv") 
os.system("shuf temp.csv > serum_set.csv")
os.system("rm temp.csv")

#cut into train val test
from subprocess import check_output

nr_of_lines = int(check_output(["wc", "-l", "serum_set.csv"]).split()[0])

#this will split it 80/10/10
os.system("split -l "+str(int(nr_of_lines*0.8))+" serum_set.csv temp")
os.system("mv tempaa serum_set_train.csv")
os.system("split -l "+str(int(nr_of_lines*0.1+0.5))+" tempab rest")
os.system("rm tempab")
os.system("mv restaa serum_set_validation.csv")
os.system("mv restab serum_set_test.csv")
