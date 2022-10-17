# This script generates the events from a homogeneous poisson process
#
# Need to specify num_reps and the parameter a_0


# general libraries
import time
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

# JAX
import jax
import jax.numpy as jnp
from jax import random, lax, jit, ops
from jax.experimental import stax

from functools import partial

# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS, init_to_median, Predictive, RenyiELBO
from numpyro.diagnostics import hpdi
from random import randint

#my functions
from utils import *
from functions import *

#for saving
import dill


#
# Simulate
#

load_data=False
save_me=True

args={}
rng_key, rng_key_predict = random.split(random.PRNGKey(2))
args["rng_key"]=random.PRNGKey(1)
args["batch_size"]= 1

T=50+30
args['T']=T
args['a_0']=.7
args['b_0']=0

n_t=50+30
x_t = jnp.arange(0, T, T/n_t)
args[ "n_t"]=n_t
args["x_t"]=x_t

n_xy = 25
grid = jnp.arange(0, 1, 1/n_xy)
u, v = jnp.meshgrid(grid, grid)
x_xy = jnp.array([u.flatten(), v.flatten()]).transpose((1, 0))
args['x_xy']=x_xy
args["n_xy"]= n_xy
args["gp_kernel"]=exp_sq_kernel
args["batch_size"]= 1

args['t_min']=0
args['x_min']=0
args['x_max']=1
args['y_min']=0
args['y_max']=1
args['sigmay_2']=.2
args['sigmax_2']=.2
alpha = .5
beta = .7
args['alpha']=alpha
args['beta']=beta



data_name='Poisson'
args['background_simulation']='Poisson'


num_reps=100
for i in range(num_reps):

	if load_data:
	  print('Loading dataset', i)			  
	  with open('data/'+data_name+'.pkl', 'rb') as file:
	    output_dict = dill.load(file)
	    simulated_output_Hawkes=output_dict['simulated_output_Hawkes'+str(i)]
	    simulated_output_Hawkes_train_test=output_dict['simulated_output_Hawkes_train_test'+str(i)]
	    simulated_output_Hawkes_background=output_dict['simulated_output_background '+str(i)]
	    args_train=output_dict['args_train']
	    args=output_dict['args']
	    data_name=output_dict['data_name']
	    a_0_true=args['a_0'] #simulated_output_background['a_0'];print(a_0_true)
	    n_obs=simulated_output_Hawkes['G_tot_t'].size
	    rate_xy_events_true=np.exp(a_0_true)*np.ones(n_obs)
	    b_0_true=args['b_0']#simulated_output_background['b_0'];print(b_0_true)


	if not(load_data):
	  if i%10==0:
		  print('Simulating dataset ',i)
	  if args['background_simulation']=='Poisson':  
	    rng_key, rng_key_predict = random.split(random.PRNGKey(i))
	    ST_background_predictive = Predictive(generate_background_uniform_events, num_samples=args["batch_size"])
	    simulated_output_background = ST_background_predictive(rng_key_predict, T=args['T'], a_0=args['a_0'], b_0=args['b_0'])
	    n_obs_back=simulated_output_background['N']
	    t_events_background=np.sort(simulated_output_background['t_events'][0])
	    xy_events_background=simulated_output_background['xy_events'][0]
	    a_0_true=args['a_0'] #simulated_output_background['a_0'];print(a_0_true)
	    b_0_true=args['b_0']#simulated_output_background['b_0'];print(b_0_true)
	    rate_xy_events_true=np.exp(a_0_true+b_0_true)*np.ones(n_obs_back)

	simulated_output=simulated_output_background
	simulated_output_Hawkes=simulated_output

	T_train=50
	if not(load_data):
	  args_train={}
	  T_test=T_train+30
	  #T=
	  args_train['T']=T_train
	  args_train['a_0']=args['a_0']
	  args_train['b_0']=args['b_0']

	  n_t_train=50
	  x_t_train = jnp.arange(0, T_train, T_train/n_t_train)
	  args_train[ "n_t"]=n_t_train
	  args_train["x_t"]=x_t_train

	  args_train['x_xy']=x_xy
	  args_train["n_xy"]= n_xy
	  args_train["gp_kernel"]=exp_sq_kernel
	  args_train["batch_size"]= 1


	train_ind=simulated_output['t_events']<T_train
	n_train=train_ind.sum()
	n_obs=simulated_output['t_events'].size
	if i%10==0:
		print('n_train',n_train, 'n_obs', n_obs, 'n_test points', n_obs-n_train)


	simulated_output_Hawkes_train_test={}
	simulated_output_Hawkes['t_events']=t_events_background.reshape(1,n_obs);
	simulated_output_Hawkes['G_tot_t']=t_events_background.reshape(1,n_obs);

	simulated_output_Hawkes_train_test['G_tot_t_train']=simulated_output_Hawkes['t_events'][:,0:n_train]
	simulated_output_Hawkes_train_test['G_tot_t_test']=simulated_output_Hawkes['t_events'][:,n_train:n_obs]
	
	simulated_output_Hawkes_train_test['G_tot_x_train']=simulated_output_Hawkes['xy_events'][:,0:n_train,0]
	simulated_output_Hawkes_train_test['G_tot_x_test']=simulated_output_Hawkes['xy_events'][:,n_train:n_obs,0]

	simulated_output_Hawkes_train_test['G_tot_y_train']=simulated_output_Hawkes['xy_events'][:,0:n_train,1]
	simulated_output_Hawkes_train_test['G_tot_y_test']=simulated_output_Hawkes['xy_events'][:,n_train:n_obs,1]

	args['background']=args['background_simulation']
	if i%10==0:
		print('Data generating model is ', data_name)


	if not(load_data):
		if i==0:
			output_dict = {}
			output_dict['data_name']=data_name
			output_dict['args']=args
			output_dict['args_train']=args_train
		output_dict['simulated_output_background '+str(i)]=simulated_output_background
		output_dict['simulated_output_Hawkes'+str(i)]=simulated_output_Hawkes
		output_dict['simulated_output_Hawkes_train_test'+str(i)]=simulated_output_Hawkes_train_test
		with open('data/'+data_name+'.pkl', 'wb') as handle:
			dill.dump(output_dict, handle)

