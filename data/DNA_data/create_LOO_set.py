import os

# all exps
exps=range(19) #all exps
filename_prefix = "serum_"

#uncomment next two lines for using only serum samples + LOO
#exps= [0,8,9,10,11] # serum
#filename_prefix = "only_serum_"


# remove one
exps_to_exclude = [8]
for ex in exps_to_exclude:
  exps.remove(ex)


exps_string = ""
for ex in exps:
  exps_string += "exp"+str(ex)+"_* "

print exps_string
os.system("cat "+exps_string+" > temp.csv") 
os.system("shuf temp.csv > "+filename_prefix+"_LOO_"+str(exps_to_exclude[0])+"_set.csv")
os.system("rm temp.csv")

#cut into train val test
from subprocess import check_output
nr_of_lines = int(check_output(["wc", "-l", filename_prefix+"_LOO_"+str(exps_to_exclude[0])+"_set.csv"]).split()[0])
print "OK",nr_of_lines

os.system("split -l "+str(int(nr_of_lines*0.9))+" "+filename_prefix+"_LOO_"+str(exps_to_exclude[0])+"_set.csv temp")
os.system("mv tempaa "+filename_prefix+"_LOO_"+str(exps_to_exclude[0])+"_set_train.csv")
os.system("mv tempab "+filename_prefix+"_LOO_"+str(exps_to_exclude[0])+"_set_validation.csv")

# test set is the left-out exp(s)
string_for_cat=""
for ex in exps_to_exclude:
 string_for_cat+= "exp"+str(ex)+"_* "

os.system("cat "+string_for_cat+" > "+filename_prefix+"_LOO_"+str(exps_to_exclude[0])+"_set_test.csv")
