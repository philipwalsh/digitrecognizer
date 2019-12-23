#################################################
# title   : [title]
# from    : [optional]
# file    : template.py
#         : philip walsh
#         : philipwalsh.ds@gmail.com
#         : [date]


# make sure to create an /excluded/ folder in the project root

import numpy as np
import pandas as pd
import os
#import matplotlib.pyplot as plt

from os.path import isfile, join

# save the script name, can be used when sending data frames to csv
# either for debugging prelim results or kaggle submissions
current_script = os.path.basename(__file__)
log_prefix = os.path.splitext(current_script)[0].replace('_','-')


bVerbose = False
my_test_size=0.1    # when ready to make the final sub
#my_test_size=0.30   # normal training eval

working_dir=os.getcwd()
excluded_dir = os.path.join(working_dir, 'excluded') # working_dir + '\excluded'

print('\n\n')
print('*****')
print('***** start of script: ', log_prefix)
print('*****')
print('\n\n')

if bVerbose:
    print('\nworking dir   :', working_dir)


import winsound
def alert_me(num_beeps):
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 500  # Set Duration To 1000 ms == 1 second

    for n in range(1,num_beeps+1):
        winsound.Beep(frequency, duration)





def sendtofile(outdir, filename, df, verbose=False):
    script_name = log_prefix + '_'
    out_file = os.path.join(outdir, script_name + filename)
    if verbose:
        print("saving file :", out_file)
    df.to_csv(out_file, index=False)
    return out_file







##
## MAIN SCRIPT START HERE
##

# load the data
train_data = pd.read_csv('excluded/train.csv', low_memory=False)
sub_data = pd.read_csv('excluded/test.csv', low_memory=False)
print(train_data.shape)
print(train_data.head())


#print('saving train_data ...', sendtofile(excluded_dir,'tran_data.csv',train_data))


print('\n\n')
print('*****')
print('***** end of script: ', log_prefix)
print('*****')
print('\n\n')

