#
# For a given simulation, dataset name and inference model, 
# this script reads the <n_pred> true future (test) simulations from a file, and performs prediction on the test set
# using the estimated parameters.
# Then computed the error and stores them in a file.
#
# You can run this from terminal like below
# python run_output_analysis_loglik.py --dataset_name 'LGCP_only' --simulation_number 0 --n_pred 200 --model_name 'Poisson'
#
# or using the bash file
#
#
#
# When you output and save from poisson, and LGCP MAKE SURE THEY ARE ORDERED!!!!!
import argparse
import os
import sys
import dill
import numpy as np
import time 
from utils import * 
from inference_functions import *
from functions import *
import numpyro
import jax
from numpyro.infer import Predictive
import pandas as pd
#import h5py
import pickle

if __name__=='__main__':

	my_parser = argparse.ArgumentParser()
	my_parser.add_argument('--dataset_name', action='store', default='LGCP_Hawkes', type=str, required=True, help='simulated dataset')
	my_parser.add_argument('--simulation_number', action='store', default=0, type=int, help='simulation series out of 100')
	my_parser.add_argument('--model_name', action='store', default='LGCP', type=str, help='model name for inference')
	my_parser.add_argument('--n_pred', action='store', default=10, type=int, help='number of predicted sequences')
	my_parser.add_argument('--simulate_predictions', action='store', default='False', type=str, help='simulate predictions or not')
	
    #num_chains, thinning

	args = my_parser.parse_args()
    
    #### choose simulated dataset to run inference on
	data_name = args.dataset_name;
	model_name = args.model_name;
	n_pred = args.n_pred;
	if args.simulate_predictions=='False':
		simulate_predictions=False
	else:
		simulate_predictions=True
	print('simulate_predictions', simulate_predictions)

    ## making sure have got correct file paths
    #script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    
	#dset_names=list(f1_get.keys())# gets the enumerated numbers 1:1000
    #### choose one of the 1000 simulations from that dataset
	simulation_number = args.simulation_number; 
	print('\n Simulation_number', simulation_number,'\n')
	#dset=f1_get[dset_names[simulation_number]]    
	#data_get = dset[:]
    #T=dset.attrs['T']
    
	numpyro.set_host_device_count(2)
        
	load_data=True

	## choose the number of dataset

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
		raise('no data name found')

	# find inference model folder
	if model_name in ['LGCP_Hawkes','LGCP-Hawkes']:
		model_folder='model_LGCP_Hawkes/'
	elif model_name in ['LGCP', 'LGCP_only']:
		model_folder='model_LGCP/'
	elif model_name=='Hawkes':
		model_folder='model_Hawkes/'
	elif model_name=='Poisson':
		model_folder='model_Poisson/'
	else:
		raise ('no model name found')


	print('Data generating name', data_name)
	print('Inference model name', model_name)

	filename='./output/simulation_comparison/'+data_folder+model_folder
	print('load output from', filename)

	with open(filename+'output'+str(simulation_number)+'.pkl', 'rb') as file:
		print(filename+'output'+str(simulation_number))
		output = dill.load(file)
		#model=output['model']
		mcmc_samples=output['samples']
		mcmc=output['mcmc']
		args_train=output['args_train']


	if model_name in ['LGCP_only', 'LGCP']:
		args_train['background']='LGCP_only';print('args train background',args_train['background'])
	elif model_name=='LGCP_Hawkes':
		args_train['background']='LGCP'; print('args train background',args_train['background'])
	elif model_name=='Hawkes':
		args_train['background']='constant'; print('args train background',args_train['background'])
	elif model_name=='Poisson':
		args_train['background']='Poisson'; print('args train background',args_train['background'])		
	else:
		raise('what is model name?')



	n_train=simulated_output_Hawkes_train_test['G_tot_t_train'].size
	t_events_total=simulated_output_Hawkes_train_test['G_tot_t_train'][0]
	xy_events_total=np.array((simulated_output_Hawkes_train_test['G_tot_x_train'],simulated_output_Hawkes_train_test['G_tot_y_train'])).reshape(2,n_train)
	args_train['t_events']=t_events_total
	args_train['xy_events']=xy_events_total
	#when reading the data

	save_me=True

	if 'a_0' in mcmc_samples.keys():
		fig, ax = plt.subplots(1, 2,figsize=(15,5))
		ax[0].plot(mcmc_samples['a_0'])
		ax[0].axhline(a_0_true+b_0_true,color='red')
		ax[0].set_ylim(([mcmc_samples['a_0'].min()-.5,mcmc_samples['a_0'].max()+.5]))
		ax[0].set_xlabel('a_0')
		ax[1].hist(mcmc_samples['a_0'],bins=150,density=True)
		ax[1].axvline(a_0_true+b_0_true,color='red')
		ax[1].set_xlim(([mcmc_samples['a_0'].min()-.5,mcmc_samples['a_0'].max()+.5]))

	if save_me:
		mypath='a_0.png'
		plt.savefig(filename+mypath)
	  
	fig, ax = plt.subplots(1, 2,figsize=(15,5))
	if 'alpha' in mcmc_samples.keys():
		ax[0].plot(mcmc_samples['alpha'])
		ax[0].axhline(alpha,color='red')
		ax[0].set_xlabel('alpha')
		#ax[0].axhline(alpha,color='red')
		ax[1].hist(mcmc_samples['alpha'],bins=150,density=True)
		ax[1].axvline(alpha,color='red')
	if save_me:
		mypath='alpha.png'
		plt.savefig(filename+mypath)


	fig, ax = plt.subplots(1, 2,figsize=(15,5))
	if 'sigmax_2' in mcmc_samples.keys():
		ax[0].plot(mcmc_samples['sigmax_2'])
		ax[0].axhline(args['sigmax_2'],color='red')
		ax[1].hist(mcmc_samples['sigmax_2'],bins=150,density=True)
		ax[1].axvline(args['sigmax_2'],color='red')
		ax[0].set_xlabel('sigma_x_2')

		if save_me:
			mypath='sigma2_x.png'
			plt.savefig(filename+mypath)


	fig, ax = plt.subplots(2, 2,figsize=(15,5))
	if 'sigmax_2' in mcmc_samples.keys():	  
		ax[0,0].plot(mcmc_samples['alpha'])
		ax[0,0].axhline(args['alpha'],color='red')
		ax[0,0].set_ylabel('alpha')

		ax[0,1].plot(mcmc_samples['beta'])
		ax[0,1].axhline(args['beta'],color='red')
		ax[0,1].set_ylabel('beta')

		ax[1,0].plot(mcmc_samples['a_0'])
		ax[1,0].axhline(args['a_0'],color='red')
		ax[1,0].set_ylabel('a_0')

		ax[1,1].plot(mcmc_samples['sigmax_2'])
		ax[1,1].axhline(args['sigmax_2'],color='red')
		ax[1,1].set_ylabel('sigma')

		if save_me:
			mypath='trace_plots.png'
			plt.savefig(filename+mypath)

	fig, ax = plt.subplots(2, 2,figsize=(15,5))

	if 'sigmax_2' in mcmc_samples.keys():
		ax[0,0].plot(mcmc_samples['sigmax_2'])
		ax[0,0].axhline(args['sigmax_2'],color='red')
		ax[0,1].hist(mcmc_samples['sigmax_2'],bins=150,density=True)
		ax[0,1].axvline(args['sigmax_2'],color='red')
	if 'sigmay_2' in mcmc_samples.keys():
		ax[1,0].plot(mcmc_samples['sigmay_2'])
		ax[1,0].axhline(args['sigmay_2'],color='red')
		ax[1,1].hist(mcmc_samples['sigmay_2'],bins=150,density=True)
		ax[1,1].axvline(args['sigmay_2'],color='red')

		if save_me:
			mypath='sigma2_x_y.png'
			plt.savefig(filename+mypath)

	fig, ax = plt.subplots(3, 1,figsize=(5,5))
	if 'alpha' in mcmc_samples.keys():

		ax[0].plot(mcmc_samples['a_0'],label='a_0')
		ax[0].axhline(a_0_true,color='red',label='a_0 true')
		ax[0].set_ylabel('a_0')

		ax[1].plot(mcmc_samples['alpha'],label='alpha')
		ax[1].axhline(alpha,color='red',)
		ax[1].set_ylabel('alpha')

		ax[2].plot(mcmc_samples['beta'])
		ax[2].axhline(beta,color='red',label='simulated value')
		ax[2].set_ylabel('beta')
		ax[2].set_xlabel('iterations')
		plt.subplots_adjust(left=0.1,
							bottom=0.1, 
							right=0.9, 
							top=0.9, 
							wspace=0.4, 
							hspace=0.4)
		plt.legend()
		if save_me:
			mypath='trace_plots_A.png'
			plt.savefig(filename+mypath)


	rng_key, rng_key_pred= random.split(random.PRNGKey(2))

	if args_train['background']=='LGCP_only':
		model_mcmc=spatiotemporal_LGCP_model
		print('Inference model has background',args_train['background'])
	elif args_train['background']=='Poisson':
		model_mcmc=spatiotemporal_homogenous_poisson
		print('Inference model has background',args_train['background'])
	else:
		model_mcmc=spatiotemporal_hawkes_model
		print('Inference model has background',args_train['background'])


	#model_mcmc=spatiotemporal_hawkes_model
	#print('args_train.keys',args_train.keys())
	#print('args_train events t',args_train['t_events'])
	
	predictive = Predictive(model_mcmc, mcmc_samples)
	predictions = predictive(rng_key_pred, args=args_train)

	if args_train['background'] not in ['constant', 'Poisson']:
		f_t_pred=predictions["f_t"]
		f_t_pred_mean=jnp.mean(f_t_pred, axis=0)
		f_t_hpdi = hpdi(f_t_pred, 0.9)

	  #f_t_hpdi.shape
	#f_t_pred_mean=jnp.mean(f_t_pred, axis=0)[0:T_train]
	#f_t_hpdi = hpdi(f_t_pred, 0.9)[0:T_train]

	##extract the last 500 samples and get the mean

	#np.m(500,)
	#n_total=750
	n_total = int(output['mcmc'].num_samples/output['mcmc'].thinning*output['mcmc'].num_chains)
	post_samples=500

	#simulated_output_Hawkes['G_tot_x'].size
	#print(simulated_output_Hawkes['G_tot_t'].size,'G_tot_t.size','\n\n')
	#print(simulated_output_Hawkes.keys(),'\n')
	a_0_post_mean=np.array(mcmc_samples['a_0'][n_total-post_samples:n_total].mean())
	a_0_post_samples=np.array(mcmc_samples['a_0'][n_total-post_samples:n_total])
	print('number of posterior samples', a_0_post_samples.size)

	#if args_train['background']=='Poisson':
	#	args_new={}
	#	args_new['t_min']=50
	#	args_new['t_max']=80
	#	args_new['a_0']=a_0_post_mean
	#	args_new['b_0']=0
	#	args_new['t_events']=None*np.zeros(0)
	#	args_new['xy_events']=None

	#	args_prior={}
	#	args_prior['a_0']=a_0_true
	#	args_prior['b_0']=0
	#	args_prior['t_events']=None
	#	args_prior['xy_events']=None
	#	args_prior['t_min']=50
	#	args_prior['t_max']=80
		#predictive = Predictive(model_mcmc, mcmc_samples)
		#predictions = predictive(rng_key_pred, args=args_new)


	#TRUE PARAMETERS if LGCP or if LGCP Hawkes
	if args['background'] in ['LGCP','LGCP_Hawkes']:
		ft_true=simulated_output_Hawkes_background['f_t'].flatten()
		f_xy_true=simulated_output_Hawkes_background['f_xy'].flatten()
		rate_xy_true=simulated_output_Hawkes_background['rate_xy'].flatten()
		rate_t_true=simulated_output_Hawkes_background['rate_t'].flatten()


	#post_samples=500
	#print('n_total',n_total,'\n\n')
	#print('post_samples', post_samples,'\n')
	
	a_0_post_mean=np.array(mcmc_samples['a_0'][n_total-post_samples:n_total].mean())
	a_0_post_samples=np.array(mcmc_samples['a_0'][n_total-post_samples:n_total])

	if args_train['background'] not in ['LGCP_only','Poisson']:
		alpha_post_mean=np.array(mcmc_samples['alpha'][n_total-post_samples:n_total].mean())
		alpha_post_samples=np.array(mcmc_samples['alpha'][n_total-post_samples:n_total])
		
		beta_post_mean=np.array(mcmc_samples['beta'][n_total-post_samples:n_total].mean())
		beta_post_samples=np.array(mcmc_samples['beta'][n_total-post_samples:n_total])

		sigma_x_2_post_mean=np.array(mcmc_samples['sigmax_2'][n_total-post_samples:n_total].mean())
		sigma_x_2_post_samples=np.array(mcmc_samples['sigmax_2'][n_total-post_samples:n_total])


	def normal_dist(mean,var,num_samples=args_train["z_dim_temporal"]):
		z_temporal=numpyro.sample("z_temporal",dist.Normal(mean, var).expand([num_samples]))

	normal_predictive = Predictive(normal_dist, num_samples=1)
	normal_predictive_samples = normal_predictive(rng_key, mean=.5, var=2)

	#z_temporal = numpyro.sample("z_temporal", dist.Normal(jnp.zeros(args["z_dim_temporal"]), jnp.ones(args["z_dim_temporal"])))
	z_temporal= normal_predictive(rng_key,jnp.zeros(args_train["z_dim_temporal"]), jnp.ones(args_train["z_dim_temporal"]))
	decoder_nn_temporal = vae_decoder_temporal(args_train["hidden_dim_temporal"], args["n_t"])  
	decoder_params = args_train["decoder_params_temporal"]
	v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal['z_temporal']))


	#plt.plot(np.mean(mcmc_samples['z_temporal'][100:],0))
	#plt.plot(z_temporal['z_temporal'][0])



	#def normal_dist(mean,var,num_samples=args_train["z_dim_temporal"]):
	#  z_temporal=numpyro.sample("z_temporal",dist.Normal(mean, var).expand([num_samples]))

	#normal_predictive = Predictive(normal_dist, num_samples=1)
	#normal_predictive_samples = normal_predictive(rng_key, mean=.5, var=2)

	#z_temporal = numpyro.sample("z_temporal", dist.Normal(jnp.zeros(args["z_dim_temporal"]), jnp.ones(args["z_dim_temporal"])))
	#z_temporal= normal_predictive(rng_key,jnp.zeros(args_train["z_dim_temporal"]), jnp.ones(args_train["z_dim_temporal"]))
	#decoder_nn_temporal = vae_decoder_temporal(args_train["hidden_dim_temporal"], args["n_t"])  
	#decoder_params = args_train["decoder_params_temporal"]
	#v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal['z_temporal']))
	args_test={}
	args_test['n_t']=80
	args_test['x_t']=np.arange(0,args['T'],1)
	args_test['T']=80
	args_test['T_test']=80
	
	args_test['n_xy']=args['n_xy']
	args_test['x_xy']=args['x_xy']
	args_test['a_0']=a_0_post_mean
	args_test['n_samples']=n_pred
	args_test['n_total']=n_total

	if args_train['background'] not in ['constant','Poisson']:
		args_test['hidden_dim_temporal']=args_train["hidden_dim_temporal"]
		args_test['z_dim_temporal']=args_train["z_dim_temporal"]
		args_test['decoder_params_temporal']=args_train["decoder_params_temporal"]
		args_test['z_dim_spatial']=args_train["z_dim_spatial"]
		args_test['hidden_dim1_spatial']=args_train["hidden_dim1_spatial"]
		args_test['hidden_dim2_spatial']=args_train["hidden_dim2_spatial"]
		args_test['decoder_params_spatial']=args_train["decoder_params_spatial"]
		args_test['a_0_post_mean']=a_0_post_mean

	def sample_GP(args,typ='posterior'):
		a_0=a_0_post_mean
		if typ=='prior':
			z_temporal= numpyro.sample("z_temporal",dist.Normal(jnp.zeros(args["z_dim_temporal"])), jnp.ones(args["z_dim_temporal"]))
		else:
			z_temporal=numpyro.deterministic("z_temporal",np.mean(mcmc_samples['z_temporal'][args['n_total']-args['n_samples']:args['n_total']],0))
		decoder_nn_temporal = vae_decoder_temporal(args["hidden_dim_temporal"], args["n_t"])  
		decoder_params = args["decoder_params_temporal"]
		v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal))
		f_t = numpyro.deterministic("f_t", v_t[0:args["n_t"]])
		rate_t = numpyro.deterministic("rate_t",jnp.exp(f_t+a_0))
		Itot_t=numpyro.deterministic("Itot_t", jnp.sum(rate_t)/args["n_t"]*args["T"])
		#Itot = numpyro.deterministic("Itot", v[len(x)])#Itot_t=jnp.trapz(mu_0*jnp.exp(f), back_t)
		#f_t_events=f_t[args["indices_t"]]

		# zero mean spatial gp
		b_0=0
		#numpyro.sample("b_0", dist.Normal(1,3))# this was 2,2
		if typ=='prior':
			z_spatial = numpyro.sample("z_spatial", dist.Normal(jnp.zeros(args["z_dim_spatial"]), jnp.ones(args["z_dim_spatial"])))
		else:
			z_spatial=numpyro.deterministic("z_spatial",np.mean(mcmc_samples['z_spatial'][args['n_total']-args['n_samples']:args['n_total']],0))

		decoder_nn = vae_decoder_spatial(args["hidden_dim2_spatial"], args["hidden_dim1_spatial"], args["n_xy"])  
		decoder_params = args["decoder_params_spatial"]
		f_xy = numpyro.deterministic("f_xy", decoder_nn[1](decoder_params, z_spatial))
		rate_xy = numpyro.deterministic("rate_xy",jnp.exp(f_xy+b_0))
		Itot_xy=numpyro.deterministic("Itot_xy", jnp.sum(rate_xy)/args["n_xy"]**2)
		#f_xy_events=f_xy[args["indices_xy"]]
		Itot_txy_back=numpyro.deterministic("Itot_txy_back",Itot_t*Itot_xy)#jnp.sum(mu_xyt*args['T']/args['n_t']/args['n']**2))


	print('number of predicred sequences', n_pred)

	if args_train['background'] not in ['constant','Poisson']:
		GP_predictive = Predictive(sample_GP, num_samples=n_pred)
		GP_predictive_samples = GP_predictive(rng_key, args_test, 'post')
		if save_me:
			mypath='GPt_post_test.png'
			plt.savefig(filename+mypath)

		f_xy_post_mean=GP_predictive_samples['f_xy'][0]
		#f_xy_post_di = hpdi(f_xy_post, 0.9)

		fig, ax = plt.subplots(1,1, figsize=(10, 5))
		_min, _max = np.amin(f_xy_post_mean), np.amax(f_xy_post_mean)
		n=25
		im = ax.imshow(f_xy_post_mean.reshape(n,n), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
		ax.title.set_text('Predictive test f_xy')
		fig.colorbar(im, ax=ax)
		if save_me:
			mypath='GPxy_post_test.png'
			plt.savefig(filename+mypath)


	if args_train['background'] not in ['constant','Poisson']:
		args_test['Itot_xy']=np.array(GP_predictive_samples['Itot_xy'])

	past_times=t_events_total
	#print('past times', past_times)
	past_locs=xy_events_total
	#print('past_locs', past_locs)

	n_test=n_obs-n_train; print('number of test points',n_test)
	N_new=n_test

	#print('n_total',n_total)
	#print('post_samples',post_samples)
	#print('a_0_post_samples',a_0_post_samples,'\n')# this is an empty array
	lambda_0_post_samples=np.exp(a_0_post_samples)
	lambda_0_post_mean=lambda_0_post_samples.mean()

	args_test['x_t']=np.arange(50,80,1)

	#
	# Simulate the predictions
	#

	post_mean=True ## always use this ie the posterior mean
	T_test=80
	T_train=50
	if args_train['background'] in ['LGCP_only', 'LGCP']:
		f_t_pred_mean=jnp.array(np.mean(GP_predictive_samples['f_t'][:,:],0));
		f_xy_pred_mean=np.mean(GP_predictive_samples['f_xy'],0);
		Itot_t_LGCP=jnp.sum(jnp.exp(f_t_pred_mean[T_train:]))/args["n_t"]*(T_test-T_train)
		Itot_xy_LGCP=jnp.sum(jnp.exp(f_xy_pred_mean))/args_train["n_xy"]**2


	x_min, x_max, y_min, y_max=0,1,0,1


	#n_pred=100
	nums=[10]
	print('simulate_predictions', simulate_predictions)
	with open('output/simulation_comparison/'+data_folder+'true_events_'+str(simulation_number)+'.pkl', 'rb') as handle:
		TRUE = pickle.load(handle)

	if simulate_predictions:

		PREDICTIONS={};

		#load true events

		PREDICTIONS['T']=np.zeros((n_pred,n_test))
		PREDICTIONS['X']=np.zeros((n_pred,n_test));
		PREDICTIONS['Y']=np.zeros((n_pred,n_test))

		rng_key, rng_key_predict = random.split(random.PRNGKey(1))
		gp_predictive = Predictive(spatiotemporal_GP, num_samples=n_pred)
		GP_prior_samples = gp_predictive(rng_key_predict, args['T'], args['x_t'], args['x_xy'], gp_kernel=exp_sq_kernel, jitter=1e-5, a_0=0, b_0=0,  var_t=1, length_t=10, var_xy=1, length_xy=.25)

		for j in range(0,n_pred):
			if j%20 ==0:
				print('Simulating predictions for the',j,'th sequence using the estimated parameters of the model', model_name)
		    	#simulate from the underlying process 

		  	## Inference model
			if args_train['background']=='LGCP':

				if post_mean:
					T_pred, X_pred, Y_pred, T_pred_all, X_pred_all, Y_pred_all=simulate_spatiotemporal_hawkes_predictions(past_times, 
		        past_locs, N_new, x_min, x_max, y_min, y_max,  lambda_0_post_mean, alpha_post_mean, beta_post_mean, sigma_x_2_post_mean, 
		         GP_predictive_samples['Itot_xy'][0], args_train['background'], f_t_pred_mean)   
				else:
					T_pred, X_pred, Y_pred, T_pred_all,X_pred_all,Y_pred_all=simulate_spatiotemporal_hawkes_predictions(past_times, 
		      past_locs, N_new, x_min, x_max, y_min, y_max,  lambda_0_post_samples[j], alpha_post_samples[j], beta_post_samples[j], sigma_x_2_post_samples[j], 
		       GP_predictive_samples['Itot_xy'], args_train['background'], np.array(GP_predictive_samples['f_t'][j])) 

			elif args_train['background']=='Poisson':
				print('will simulate events now')
				print(args_train['t_max'])
				#a_0_post_mean=np.array(mcmc_samples['a_0'][n_total-post_samples:n_total].mean())
				#a_0_post_samples=np.array(mcmc_samples['a_0'][n_total-post_samples:n_total])
					
				args_new={}
				args_new['t_min']=50
				args_new['t_max']=80
				args_new['a_0']=a_0_post_mean
				args_new['b_0']=0
				args_new['t_events']=args_train['t_events']
				args_new['n_test']=n_test;#print('n_test',n_test)
				args_new['x_min']=0
				args_new['x_max']=1
				args_new['y_min']=0
				args_new['y_max']=1	
				
				T_pred, X_pred, Y_pred=simulate_uniform_Poisson(args_new)
				#ST_background_predictive = Predictive(spatiotemporal_homogenous_poisson, num_samples=args["batch_size"])
				#simulated_output_background = ST_background_predictive(rng_key_predict, args_train)
				
				#print(simulated_output_background.keys())
				#print('n_test', n_test)
				#print(simulated_output_background['a_0'][0])
				#print(simulated_output_background['b_0'][0])
				
				#n_obs_back=simulated_output_background['N']
				#print('simulated_output_background[t_events]', simulated_output_background['t_events'])
				#print('simulated_output_background[xy_events]', simulated_output_background['xy_events'])

				#T_pred=np.sort(simulated_output_background['t_events'])[0,0:n_test]
				#X_pred=simulated_output_background['xy_events'][0,0:n_test,0]
				#Y_pred=simulated_output_background['xy_events'][0,:n_test,1]  
			
			elif args_train['background']=='constant':

				if post_mean:
					T_pred, X_pred, Y_pred, T_pred_all,X_pred_all,Y_pred_all=simulate_spatiotemporal_hawkes_predictions(past_times, 
					past_locs, N_new, x_min, x_max, y_min, y_max, lambda_0_post_mean, alpha_post_mean, beta_post_mean, sigma_x_2_post_mean, 0,
					args_train['background'])  
				else:
					T_pred, X_pred, Y_pred, T_pred_all,X_pred_all,Y_pred_all=simulate_spatiotemporal_hawkes_predictions(past_times, 
					past_locs, N_new, x_min, x_max, y_min, y_max, lambda_0_post_samples[j], alpha_post_samples[j], beta_post_samples[j], sigma_x_2_post_samples[j],0, 
					args_train['background'])  
		      
			elif args_train['background']=='LGCP_only':
				if post_mean:
					N_0=n_test;
					ind_t_i, t_i, rate_t_i=rej_sampling_new(N_0, np.arange(T_train, T_test, 1), lambda_0_post_mean*np.exp(f_t_pred_mean[T_train:]), 30)
					N_0 = t_i.shape[0]
					ind_xy_i, xy_i, rate_xy_i=rej_sampling_new(N_0, args['x_xy'], GP_predictive_samples['rate_xy'][j,:], args['n_xy']**2)
					ord=t_i.sort()
					T_pred=t_i[ord].flatten()
					X_pred=xy_i[:,0][ord].flatten()
					Y_pred=xy_i[:,1][ord].flatten()
					T_pred_all=np.concatenate((past_times,T_pred.flatten()))
					X_pred_all=np.concatenate((past_locs[0],X_pred.flatten()))
					Y_pred_all=np.concatenate((past_locs[1],Y_pred.flatten()))
				else:
					N_0=n_test
					ind_t_i, t_i, rate_t_i=rej_sampling_new(N_0, np.arange(T_train, T_test,1), lambda_0_post_samples[j]*np.exp(GP_predictive_samples['f_t'][j][args_train['T']:]), 30)
					N_0 = t_i.shape[0]
					ind_xy_i, xy_i, rate_xy_i=rej_sampling_new(N_0, args['x_xy'], GP_predictive_samples['rate_xy'][j,:], args['n_xy']**2)
					ord=t_i.sort()
					T_pred=t_i[ord].flatten();print('T_pred',T_pred)
					X_pred=xy_i[:,0][ord].flatten()
					Y_pred=xy_i[:,1][ord].flatten()
					T_pred_all=np.concatenate((past_times,T_pred.flatten()))
					X_pred_all=np.concatenate((past_locs[0],X_pred.flatten()))
					Y_pred_all=np.concatenate((past_locs[1],Y_pred.flatten()))
		  
		  #store the predicted and the true events
		  #print(T_pred.shape)
		  #print(X_pred.shape)
		  #print(PREDICTIONS['T'][j].shape)
			PREDICTIONS['T'][j]=T_pred#+np.random.normal(0,0.4)#np.round(T_pred)
			PREDICTIONS['X'][j]=X_pred#np.round(X_pred)
			PREDICTIONS['Y'][j]=Y_pred#np.round(Y_pred)

		  #TRUE['T'][j]=T_true_test#np.round(T_true_test)
		  #TRUE['X'][j]=X_true_test#np.round(X_true_test)
		  #TRUE['Y'][j]=Y_true_test#np.round(Y_true_test)	  
		
		with open(filename+'prediction_events'+'.txt', 'a') as f:						
			f.write(str(PREDICTIONS['T'])+'\n')
		
		with open(filename+'predictions_'+str(simulation_number)+'.pkl', 'wb') as f:
			pickle.dump(PREDICTIONS, f, protocol=pickle.HIGHEST_PROTOCOL)
	else:
		PREDICTIONS=pd.read_pickle(filename+'predictions_'+str(simulation_number)+'.pkl')
		
		#with open(filename+'predictions_'+str(simulation_number)+'.pkl', 'wb') as f:
		#	pickle.load(PREDICTIONS, f, protocol=pickle.HIGHEST_PROTOCOL)

	#
	# Evaluate the loglikelihood on the predicted data
	#


	#
	# Measure the error between the predicted and the true
	#
	#LOGLIK=np.zeros((nums,n_pred));
    
	#LOGLIK={};keys = [0, 1, 2];LOGLIK = {k:None for k in keys}#EA_mean_space=np.zeros(len(nums));
	#LOGLIK_std={};keys = [0, 1, 2];LOGLIK_std = {k:None for k in keys}#EA_std_space=np.zeros(len(nums));
	#n_pred=PREDICTIONS['T'].shape[0];
	print('n_pred',n_pred)
	LOGLIK=np.zeros(n_pred);
	LOGLIK_B=np.zeros(n_pred);
	use_post_samples=True
	use_post_mean=True
	num_samples=500
	for _,n_stop in enumerate(nums):
		FIT=pd.DataFrame({'loglik_'+str(n_stop): np.zeros(n_pred),'loglik_std_'+str(n_stop): np.zeros(n_pred),'loglikB_'+str(n_stop): np.zeros(n_pred),'loglikB_std_'+str(n_stop): np.zeros(n_pred)})

	n_stop=10


	if args_train['background']=='Poisson':
		T_test=80
		T_train=50
		#n_test_pred=np.random.poisson(a_0_post_mean*(T_test-T_train),n_pred)
		#print('n_test_pred',n_test_pred)

		if 	use_post_mean:
			loglik= a_0_post_mean*n_stop-jnp.exp(a_0_post_mean)*(T_test-T_train)
			LOGLIK_B=loglik
		if use_post_samples:
			#n_test_pred=np.random.poisson(a_0_post_samples*(T_test-T_train), n_pred)
			loglik= a_0_post_samples[0:n_pred]*n_stop-jnp.exp(a_0_post_samples[0:n_pred])*(T_test-T_train)
			LOGLIK=loglik

	if args_train['background']=='LGCP_only':
		x_t_test=np.arange(T_train,T_test,1)
		x_xy_test=args['x_xy']
		indices=np.arange(n_stop)
		#ind_t_i=np.zeros(30).astype(int)
		for j in range(n_pred):			
			T_pred=np.asarray(PREDICTIONS['T'][j])[indices]
			X_pred=PREDICTIONS['X'][j][indices]
			Y_pred=PREDICTIONS['Y'][j][indices]
			XY_pred=np.array((X_pred,Y_pred));#print('XY_pred.size',XY_pred.shape)
			ind_t_i=find_index_b(T_pred,x_t_test)	 
			ind_xy_i=find_index_b(XY_pred.transpose(), args['x_xy'])


			if use_post_mean:
				n_t_test=30
				#Itot_t_mean=jnp.sum(jnp.exp(f_t_pred_mean[T_train:]))/n_t_test*(T_test-T_train)
				#Itot_xy_mean=jnp.sum(jnp.exp(f_xy_pred_mean))/args_train["n_xy"]**2
				Itot_t_mean=jnp.sum(jnp.exp(f_t_pred_mean[T_train:]))
				Itot_xy_mean=jnp.sum(jnp.exp(f_xy_pred_mean))/args_train["n_xy"]**2				
				LOGLIK_B[j]=np.sum(f_t_pred_mean[T_train:][ind_t_i]+f_xy_pred_mean[ind_xy_i]+a_0_post_mean) - Itot_t_mean*Itot_xy_mean*lambda_0_post_mean


			if use_post_samples:
				ll=np.sum(GP_predictive_samples['f_t'][j,T_train:][ind_t_i]+GP_predictive_samples['f_xy'][j,:][ind_xy_i]+a_0_post_samples[j])
				Itot_t_LGCP=jnp.sum(jnp.exp(GP_predictive_samples['f_t'][j,T_train:]))/n_t_test*(T_test-T_train)
				Itot_xy_LGCP=jnp.sum(jnp.exp(GP_predictive_samples['f_xy'][j]))/args_train["n_xy"]**2	
				LOGLIK[j]=ll-Itot_xy_LGCP*Itot_t_LGCP*lambda_0_post_samples[j]



	if args_train['background']in ['constant','LGCP']: #hawkes case

		x_t_test=np.arange(T_train,T_test,1)
		x_xy_test=args['x_xy']
		ind_t_i=np.zeros(80).astype(int)
		for j in range(n_pred):			
			indices=np.arange(n_stop)
			T_pred=np.asarray(PREDICTIONS['T'][j])
			X_pred=PREDICTIONS['X'][j]
			Y_pred=PREDICTIONS['Y'][j]
			T_pred_all=np.concatenate((past_times,T_pred[indices].flatten()))
			X_pred_all=np.concatenate((past_locs[0],X_pred[indices].flatten()))
			Y_pred_all=np.concatenate((past_locs[1],Y_pred[indices].flatten()))
			XY_pred_all=np.array((X_pred_all,Y_pred_all));#print('XY_pred.size',XY_pred.shape)
			print(args['x_t'])
			ind_t_i=find_index_b(T_pred_all,args['x_t'])	 
			ind_xy_i=find_index_b(XY_pred_all.transpose(), args['x_xy'])


			args_test=args_train					
			args_test['x_min']=args['x_min']
			args_test['x_max']=args['x_max']
			args_test['y_min']=args['y_min']
			args_test['y_max']=args['y_max']
			args_test['n_t_train']=50
			args_test['n_t_test']=80
			args_test['T_test']=80
			args_test['T_train']=50
			args_test['n_stop']=n_stop
			args_test['background']=args_train['background']
			args_test['t_events']=T_pred_all
			args_test['xy_events']=np.array((X_pred_all,Y_pred_all))
			#print('background is', args_train['background'])

			#print('LGCP-HAWKES')
			if use_post_mean:				
				#args_test=args
				#args_test["z_dim_temporal"]=args_train["z_dim_temporal"]
				#args_test["hidden_dim_temporal"]=args_train["hidden_dim_temporal"]
				args_test['alpha']=alpha_post_mean
				args_test['beta']=beta_post_mean
				args_test['a_0']=np.log(lambda_0_post_mean)
				args_test['sigmax_2']=sigma_x_2_post_mean
				args_test['sigmay_2']=sigma_x_2_post_mean
				args_test['indices_t']=ind_t_i
				args_test['indices_xy']=ind_xy_i	
				if args_test['background']=='LGCP':
					args_test['f_t']=f_t_pred_mean
					args_test['f_xy']=f_xy_pred_mean								
				Hawkes_lik_predictive = Predictive(Hawkes_likelihood, num_samples=1)
				HK = Hawkes_lik_predictive(rng_key, args_test)
				#print(HK['loglik'])
				LOGLIK_B[j]=HK['loglik']	
			
			if use_post_samples:

				args_test['alpha']=alpha_post_samples[j]
				args_test['beta']=beta_post_samples[j]
				args_test['a_0']=np.log(lambda_0_post_samples[j])
				args_test['sigmax_2']=sigma_x_2_post_samples[j]
				args_test['sigmay_2']=sigma_x_2_post_samples[j]
				if args_test['background']=='LGCP':
					args_test['f_t']=GP_predictive_samples['f_t'][j,]	
					args_test['f_xy']=GP_predictive_samples['f_xy'][j]	
				Hawkes_lik_predictive = Predictive(Hawkes_likelihood, num_samples=1)
				HK = Hawkes_lik_predictive(rng_key, args_test)
				LOGLIK[j]=HK['loglik']					
		

	FIT['loglikB_'+str(n_stop)]=LOGLIK_B; #print('ErrorA_space', ErrorA_space) #EA_mean_space[ii]=np.round(np.mean(ErrorA_space),3);
	FIT['loglikB_std_'+str(n_stop)]=np.std(LOGLIK_B); #=np.round(np.std(ErrorA_space));

	FIT['loglik_'+str(n_stop)]=LOGLIK; #print('ErrorA_space', ErrorA_space) #EA_mean_space[ii]=np.round(np.mean(ErrorA_space),3);
	FIT['loglik_std_'+str(n_stop)]=np.std(LOGLIK); #=np.round(np.std(ErrorA_space));


	print('\nAVERAGE loglik using post mean',LOGLIK_B.mean())
	print('stdev of loglik using post mean', np.std(LOGLIK_B))

	print('\,AVERAGE loglik using samples',LOGLIK.mean())
	print('stdev of loglik using post samples', np.std(LOGLIK))

	FIT['loglikB_'+str(n_stop)]=LOGLIK_B; #print('ErrorA_space', ErrorA_space) #EA_mean_space[ii]=np.round(np.mean(ErrorA_space),3);
	FIT['loglikB_std_'+str(n_stop)]=np.std(LOGLIK_B); #=np.round(np.std(ErrorA_space));

	FIT['loglik_'+str(n_stop)]=LOGLIK; #print('ErrorA_space', ErrorA_space) #EA_mean_space[ii]=np.round(np.mean(ErrorA_space),3);
	FIT['loglik_std_'+str(n_stop)]=np.std(LOGLIK); #=np.round(np.std(ErrorA_space));



	save_error=False
	print('T test from original data sequence', simulated_output_Hawkes_train_test['G_tot_t_test'][0,:n_stop],'\n')
	print('T estimated', PREDICTIONS['T'][0,:n_stop],'\n')

	if save_error:
		print('Saving data in', filename+'ERROR_'+str(simulation_number)+'.pkl')
	
	save_loglik=True
	if save_loglik:
		print('Saving loglik in', filename+'LOGLIK_'+str(simulation_number)+'.pkl')
		FIT.to_pickle(filename+'LOGLIK_'+str(simulation_number)+'.pkl')  # where to save it, usually as a .pkl