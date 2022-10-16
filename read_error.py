import argparse
import os
import sys
import dill
import numpy as np
import time 
from utils import * 
from inference_functions import *
from functions import *

from numpyro.infer import Predictive
import pandas as pd
import pickle
# I want to take the average of the column inside the 
# mean_predictions_error_A_space_10
# mean_predictions_error_A_time_10
# std_predictions_error_A_space_10
# std_predictions_error_A_time_10

# for A and B and for 10
# read original data


## read output
data_model_names=['LGCP_Hawkes/']
inference_model_names=['Hawkes/']
i=0
j=0
data_folder='data_' + data_model_names[i]
print('\n Data simulated from ', data_model_names[i],'\n')

print('\n Data inferred from ', inference_model_names[j],'\n')

#model_names=['LGCP/','LGCP_Hawkes/','Hawkes/','Poisson/']
#model_names=['Hawkes']
Total_error=[];Total_errorB=[];Total_std_errorB=[];Total_std_error=[]
Total_error_space=[];Total_errorB_space=[];Total_std_errorB_space=[];Total_std_error_space=[]
Total_errorC_space=[];Total_std_errorC_space=[];
Total_errorD_space=[];Total_std_errorD_space=[];
TOTAL_combined_errorA=[]; TOTAL_combined_errorB=[]

n_simul=1
#for i,n in enumerate(model_names):
for n in np.arange(n_simul):
	model_folder='model_'+inference_model_names[j]
	filename='output/simulation_comparison/'+data_folder+model_folder
	ERROR = pd.read_pickle(filename+'ERROR_'+str(n)+'.pkl')
	if n%10==0:
		print('reading error for simulation ',n, np.mean(ERROR['EA_mean_t_10']))
		
	print(ERROR.keys())
	Total_error=np.concatenate((Total_error, ERROR['EA_mean_t_10']))
	Total_std_error=np.concatenate((Total_std_error, ERROR['EA_std_t_10']))

	Total_errorB=np.concatenate((Total_errorB, ERROR['EB_mean_t_10']))
	Total_std_errorB=np.concatenate((Total_std_errorB, ERROR['EB_std_t_10']))

	TOTAL_combined_errorA=np.concatenate((TOTAL_combined_errorA, ERROR['ErrorA_combined_10']))
	TOTAL_combined_errorB=np.concatenate((TOTAL_combined_errorB, ERROR['ErrorB_combined_10']))

	Total_error_space=np.concatenate((Total_error_space, ERROR['EA_mean_space_10']))
	Total_std_error_space=np.concatenate((Total_std_error, ERROR['EA_std_space_10']))

	Total_errorB_space=np.concatenate((Total_errorB_space, ERROR['EB_mean_space_10']))
	Total_std_errorB_space=np.concatenate((Total_std_errorB, ERROR['EB_std_space_10']))

print('TOTAL ERRORS \n')
print('For inference model', model_folder, 'mean total_error A time and space among all simulations is ',np.round(np.mean(TOTAL_combined_errorA),3))
print('For inference model', model_folder, 'st dev total_error A time and space among all simulations is ',np.round(np.std(TOTAL_combined_errorA),3))

print('For inference model', model_folder, 'mean total_error B time and space among all simulations is ',np.round(np.mean(TOTAL_combined_errorB),3))
print('For inference model', model_folder, 'st dev total_error B time and space among all simulations is ',np.round(np.std(TOTAL_combined_errorB),3))

#print('For inference model', model_folder, 'Error is ',np.mean(ERROR['EA_mean_t_10']))
	#df3 = pd.read_pickle(filename+'ERROR')




