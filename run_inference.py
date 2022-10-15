#####

# to run from terminal do
##
#python run_inference.py --dataset_name='LGCP_Hawkes' --simulation_number=0 --model_name='Hawkes' --num_samples=20 --num_warmup=0

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

	my_parser = argparse.ArgumentParser()
	my_parser.add_argument('--dataset_name', action='store', default='LGCP_Hawkes' ,type=str, required=True, help='simulated dataset')
	my_parser.add_argument('--simulation_number', action='store',default=0 , type=int, help='simulation series out of 100')
	my_parser.add_argument('--model_name', action='store',default='LGCP_Hawkes' , type=str, help='model name for inference')
	my_parser.add_argument('--num_samples', action='store', default=1000 , type=int, help='mcmc iterations')    
	my_parser.add_argument('--num_warmup', action='store', default=500 , type=int, help='mcmc warmup')    
	my_parser.add_argument('--num_chains', action='store', default=2 , type=int,help='mcmc num chains')
	my_parser.add_argument('--num_thinning', 


		action='store', default=2, type=int,help='mcmc num thinning')
	my_parser.add_argument('--max_tree_depth',action='store', default=20, type=int,help='max_tree_depth')
	my_parser.add_argument('--save_results', action='store', default=True, type=bool,help='save output of mcmc')

    #num_chains, thinning
	args = my_parser.parse_args()
    
    
    #### choose simulated dataset to run inference on
	data_name = args.dataset_name
	model_name = args.model_name
	save_me=args.save_results

	print('Data generating model ', data_name)
	print('Inference model is ', model_name)
    ## making sure have got correct file paths
    #script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    
	#dset_names=list(f1_get.keys())# gets the enumerated numbers 1:1000
    #### choose one of the 1000 simulations from that dataset
	simulation_number = args.simulation_number
	#dset=f1_get[dset_names[simulation_number]]    
	#data_get = dset[:]
    #T=dset.attrs['T']
    
	numpyro.set_host_device_count(2)
        
    #### choose model under which to run inference on
	num_samples = args.num_samples
	num_warmup = args.num_warmup
	num_chains = args.num_chains
	num_thinning = args.num_thinning
	max_tree_depth=args.max_tree_depth

	load_data=True

	## choose the number of dataset
	i=simulation_number

	if load_data:
	  with open('data/'+data_name+'.pkl', 'rb') as file:
	    output_dict = dill.load(file)
	    simulated_output_Hawkes=output_dict['simulated_output_Hawkes'+str(i)]
	    simulated_output_Hawkes_train_test=output_dict['simulated_output_Hawkes_train_test'+str(i)]
	    args_train=output_dict['args_train']
	    args=output_dict['args']
	    #data_name=output_dict['data_name']
	    a_0_true=args['a_0'] #simulated_output_background['a_0'];print(a_0_true)
	    n_obs=simulated_output_Hawkes['G_tot_t'].size
	    rate_xy_events_true=np.exp(a_0_true)*np.ones(n_obs)
	    b_0_true=args['b_0']#simulated_output_background['b_0'];print(b_0_true)


	if model_name=='LGCP_Hawkes':
		args_train['background']='LGCP'
	elif model_name=='LGCP':
		args_train['background']='LGCP_only'
	elif model_name=='Hawkes':
		args_train['background']='constant'
	else:
		args_train['background']='Poisson'


	args_train["hidden_dim_temporal"]= 50
	args_train["z_dim_temporal"]= 20
	## I load my 1D temporal trained decoder parameters to generate GPs with hyperparameters that make sense in this domain
	# Load 
	#fixed lengthscale=10, var=1, T50
	with open('decoders/decoder_1d_T80_fixed_ls10', 'rb') as file:
	    decoder_params = pickle.load(file)
	    print(len(decoder_params))

	args_train["decoder_params_temporal"] = decoder_params
	#args["indices"]=indices


	#@title
	if args_train['background'] not in ['constant','Poisson']:
	  # spatial VAE training
	  args_train["hidden_dim1_spatial"]= 35
	  args_train["hidden_dim2_spatial"]= 30
	  args_train["z_dim_spatial"]=10
	  n_xy=25


	#@title
	if args_train['background'] not in ['constant','Poisson']:
	  n=n_xy
	  #Load 2d spatial trained decoder
	  with open('./decoders/decoder_2d_n25_infer_hyperpars'.format(n), 'rb') as file:
	      decoder_params = pickle.load(file)

	  args_train["decoder_params_spatial"] = decoder_params
	  
	  #args_train["decoder_params_spatial"]=args["decoder_params_spatial"]


	# MCMC inference
	args_train["num_warmup"]= num_warmup
	args_train["num_samples"] = num_samples
	args_train["num_chains"] =num_chains
	args_train["thinning"] = num_thinning

	n_train=simulated_output_Hawkes_train_test['G_tot_t_train'].size

	#when reading the data
	t_events_total=simulated_output_Hawkes_train_test['G_tot_t_train'][0]
	xy_events_total=np.array((simulated_output_Hawkes_train_test['G_tot_x_train'],simulated_output_Hawkes_train_test['G_tot_y_train'])).reshape(2,n_train)

	args_train["t_events"]=t_events_total
	args_train["xy_events"]=xy_events_total

	if args_train['background'] not in ['constant', 'Poisson']:
	  #need to add the indices
	  #print('t events shape',t_events_total.shape)
	  #print('x_t size',args_train['x_t'].shape)
	  
	  indices_t=find_index(t_events_total, args_train['x_t'])
	  indices_xy=find_index(xy_events_total.transpose(), args_train['x_xy'])
	  args_train['indices_t']=indices_t
	  args_train['indices_xy']=indices_xy
	  

	rng_key, rng_key_predict = random.split(random.PRNGKey(3))
	rng_key, rng_key_post, rng_key_pred = random.split(rng_key, 3)
	
	print('Background of the inference model is', args_train['background'])

	if args_train['background']=='LGCP_only':
	  model_mcmc=spatiotemporal_LGCP_model
	elif args_train['background']=='Poisson':
	  model_mcmc=spatiotemporal_homogenous_poisson
	else:
	  model_mcmc=spatiotemporal_hawkes_model

	args_train['x_min']=args['x_min'];
	args_train['x_max']=args['x_max'];
	args_train['y_min']=args['y_min'];
	args_train['y_max']=args['y_max'];

	if args_train['background']=='Poisson':
	  args_train['a_0']=None
	  args_train['t_events']=t_events_total
	  args_train['xy_events']=xy_events_total
	  
	  args_train['b_0']=0
	  args_train['t_min']=0
	  args_train['t_max']=50


	#print('true time events', t_events_total)
	args_train['t_events']=t_events_total
	#print('true xy events',xy_events_total)
	args_train['xy_events']=xy_events_total

	# inference
	print('Run inference for simulation', simulation_number)
	mcmc = run_mcmc(rng_key_post, model_mcmc, args_train)
	mcmc_samples=mcmc.get_samples()
	print('Estimating', mcmc_samples.keys())


	#save_me=True taken from the args with default True
	data_folder='data_'+data_name+'/'
	model_folder='model_'+model_name+'/'

	folder_name='simulation_comparison/'

	if save_me:
		filename='output/'+folder_name+data_folder+model_folder	
		print('Saving results in ', filename)
		output = {}
		output['model']=model_mcmc
		#output_dict['guide']=guide
		output['samples']=mcmc.get_samples()
		output['mcmc']=mcmc
		output['args_train']=args_train
		with open(filename+'output'+str(simulation_number)+'.pkl', 'wb') as handle:
			dill.dump(output, handle)



