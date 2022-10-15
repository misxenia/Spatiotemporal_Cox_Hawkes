# For a given dataset and simulation number 
# this script simulates from the true generating model  n_pred times
# and then stores it to be used in comparison to the predicted events
# under the different models.
#
# you can run from terminal like this
# python run_simulate_future_events.py --dataset_name 'LGCP_only' --simulation_number "$k" --n_pred 200 --model_name 'LGCP' #> "output/D1/D1M1S$i.txt"
# or using the bash file

if __name__=='__main__':
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
	import h5py
	import pickle

	my_parser = argparse.ArgumentParser()
	my_parser.add_argument('--dataset_name', action='store', default='LGCP_Hawkes', type=str, required=True, help='simulated dataset')
	my_parser.add_argument('--n_pred', action='store', default='10', type=int, help='numer of sequences to simulate')
	my_parser.add_argument('--simulation_number', action='store', default=0, type=int, help='simulation series out of 100')

    #num_chains, thinning
	args = my_parser.parse_args()
    
    #### choose simulated dataset to run inference on
	data_name = args.dataset_name;

	simulation_number=args.simulation_number
	print('simulation_number', simulation_number)
    ## making sure have got correct file paths
    #script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    
	#dset_names=list(f1_get.keys())# gets the enumerated numbers 1:1000
    #### choose one of the 1000 simulations from that dataset
	n_pred = args.n_pred;
	print('n_pred', n_pred)
	#dset=f1_get[dset_names[simulation_number]]    
	#data_get = dset[:]
    #T=dset.attrs['T']
    
	numpyro.set_host_device_count(2)
        
	load_data=True

	## choose the number of dataset
	#i=simulation_number


	#data_name='LGCP-Hawkes'
	with open('data/'+data_name+'.pkl', 'rb') as file:
		print(data_name)
		output_dict = dill.load(file)
		simulated_output_Hawkes=output_dict['simulated_output_Hawkes'+str(simulation_number)]
		simulated_output_Hawkes_train_test=output_dict['simulated_output_Hawkes_train_test'+str(simulation_number)]
		simulated_output_Hawkes_background=output_dict['simulated_output_background '+str(simulation_number)]	
		args_train=output_dict['args_train']
		args=output_dict['args']
		data_name=output_dict['data_name']
		a_0_true=args['a_0'] #simulated_output_background['a_0'];print(a_0_true)
		n_obs=simulated_output_Hawkes['G_tot_t'].size;#print('n_obs',n_obs)
		rate_xy_events_true=np.exp(a_0_true)*np.ones(n_obs)
		b_0_true=args['b_0']#simulated_output_background['b_0'];print(b_0_true)
		alpha=args['alpha']
		beta=args['beta']



	# find generating data folder
	if data_name in ['LGCP_Hawkes','LGCP-Hawkes']:
		data_folder='data_LGCP_Hawkes/'
	elif data_name=='Hawkes':
		data_folder='data_Hawkes/'
	elif data_name=='LGCP_only':
		data_folder='data_LGCP_only/'
	elif data_name=='Poisson':
		data_folder='data_Poisson/'

	else:
		error('no data name found')

	filename='output/simulation_comparison/'+data_folder
	print('Saving output in', filename)


	print('Data generating name', data_name)

	n_train=simulated_output_Hawkes_train_test['G_tot_t_train'].size
	t_events_total=simulated_output_Hawkes_train_test['G_tot_t_train'][0]
	xy_events_total=np.array((simulated_output_Hawkes_train_test['G_tot_x_train'],simulated_output_Hawkes_train_test['G_tot_y_train'])).reshape(2,n_train)
	args_train['t_events']=t_events_total
	args_train['xy_events']=xy_events_total
	#when reading the data

	#TRUE PARAMETERS if LGCP or if LGCP Hawkes
	if args['background'] in ['LGCP','LGCP_Hawkes']:
		ft_true=simulated_output_Hawkes_background['f_t'].flatten()
		f_xy_true=simulated_output_Hawkes_background['f_xy'].flatten()
		rate_xy_true=simulated_output_Hawkes_background['rate_xy'].flatten()
		rate_t_true=simulated_output_Hawkes_background['rate_t'].flatten()
	


	args_test={}


	args_test['n_t']=80
	args_test['x_t']=np.arange(0,args['T'],1)
	args_test['T']=80
	args_test['T_test']=80
	args_test['n_xy']=args['n_xy']
	args_test['x_xy']=args['x_xy']
	
	past_times=t_events_total
	past_locs=xy_events_total
	n_test=n_obs-n_train;print('number of test points',n_test)
	N_new=n_test



	args_test['x_t']=np.arange(50,80,1)

	
	x_min, x_max, y_min, y_max=0,1,0,1
	T_test=80
	T_train=50


	TRUE={}
	TRUE['T']=np.zeros((n_pred,n_test));TRUE['X']=np.zeros((n_pred,n_test));TRUE['Y']=np.zeros((n_pred,n_test))

	nums=[10]

	TRUE_df=pd.DataFrame()
	for n_stop in enumerate(nums):
		TRUE_df=pd.DataFrame({'T'+str(n_stop): np.zeros(n_pred),'X'+str(n_stop): np.zeros(n_pred),'Y'+str(n_stop): np.zeros(n_pred)})
		

	rng_key, rng_key_predict = random.split(random.PRNGKey(1))
	gp_predictive = Predictive(spatiotemporal_GP, num_samples=n_pred)
	GP_prior_samples = gp_predictive(rng_key_predict, args['T'], args['x_t'], args['x_xy'], gp_kernel=exp_sq_kernel, jitter=1e-5, a_0=0, b_0=0,  var_t=1, length_t=10, var_xy=1, length_xy=.25)


	# SIMULATE MANY SEQUENCES FOR THE FUTURE EVENTS
	for j in range(0,n_pred):
	  if j%20 ==0:
	    print('Simulating predictions of the',j,'th sequence from the model', data_name, 'with', args['background_simulation'], 'background')
	    #simulate from the underlying process 

	  #	
	  # True generating model
	  # LGCP only (i.e. Cox)
	  if data_name=='LGCP_only':
	  	N_0=n_test
	  	ind_t_i, t_i, rate_t_i=rej_sampling_new(N_0, np.arange(args_train['T'], T_test,1), np.exp(a_0_true)*np.exp(GP_prior_samples['f_t'][j][args_train['T']:]), 30)
	  	N_0 = t_i.shape[0]
	  	ind_xy_i, xy_i, rate_xy_i=rej_sampling_new(N_0, args['x_xy'], GP_prior_samples['rate_xy'][j,:], args['n_xy']**2)
	  	ord=np.argsort(t_i)
	  	T_true_test=t_i[ord].flatten()
	  	X_true_test=xy_i[:,0][ord].flatten()
	  	Y_true_test=xy_i[:,1][ord].flatten()

	  # Hawkes-Cox
	  if data_name in ['LGCP_Hawkes','LGCP-Hawkes']:
	  	#print('data_name', data_name, args['background_simulation'])
	  	T_true_test, X_true_test, Y_true_test, T_pred_all_test, X_pred_all_test,Y_pred_all_test=simulate_spatiotemporal_hawkes_predictions(past_times, 
	    past_locs, N_new, args['x_min'], args['x_max'], args['y_min'], args['y_max'], np.exp(a_0_true), args['alpha'], args['beta'], args['sigmax_2'], GP_prior_samples['Itot_xy'][j],args['background_simulation'], np.array(GP_prior_samples['f_t'][j]))
	  	
	  # Hawkes
	  if data_name=='Hawkes':# if hawkes cosntant background
	  	#print(args['background_simulation'])
	  	T_true_test, X_true_test, Y_true_test, T_pred_all_test, X_pred_all_test,Y_pred_all_test=simulate_spatiotemporal_hawkes_predictions(past_times, 
	    past_locs, N_new, args['x_min'], args['x_max'], args['y_min'], args['y_max'], np.exp(a_0_true), args['alpha'], args['beta'], args['sigmax_2'], GP_prior_samples['Itot_xy'][j],args['background_simulation'], None)

	  # Poisson
	  if data_name=='Poisson':# if hawkes cosntant background
	  	#print('will simulate Poisson events now')
	  	args_new={}
	  	args_new['t_min']=50
	  	args_new['t_max']=80
	  	args_new['x_min']=0
	  	args_new['x_max']=1
	  	args_new['y_min']=0
	  	args_new['y_max']=1	
	  	args_new['a_0']=a_0_true
	  	args_new['b_0']=0
	  	args_new['t_events']=args_train['t_events']
	  	N_new=n_test
	  	args_new['n_test']=n_test;#print('n_test',n_test)
	  	T_true_test, X_true_test, Y_true_test=simulate_uniform_Poisson(args_new)
	  	#print('T_true', T_true_test)
	  
	  TRUE['T'][j]=T_true_test
	  #np.round(T_true_test)
	  TRUE['X'][j]=X_true_test
	  #np.round(X_true_test)
	  TRUE['Y'][j]=Y_true_test
	  #np.round(Y_true_test)
	  
	with open(filename+'true_events_'+str(simulation_number)+'.pkl', 'wb') as f:
		#print(TRUE['T'])
		pickle.dump(TRUE, f, protocol=pickle.HIGHEST_PROTOCOL)




















