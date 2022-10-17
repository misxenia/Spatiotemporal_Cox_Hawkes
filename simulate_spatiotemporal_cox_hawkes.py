##
# This can generate the events for a cox-hawkes process orda pure LGCP process
##
# To generate a Hawkes process with constant background use 
# data_name='Hawkes'             
# args['background_simulation']='constant' 

# To generate a Hawkes process with LGCP background use
# data_name='LGCP_Hawkes'          
# args['background_simulation']='LGCP' 

# To generate an LGCP process
# data_name='LGCP_only'        
# args['background_simulation']='LGCP_only' 

# Need to specify
# num_reps and the parameters of the model


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


from utils import *
from functions import *

import dill


load_data=False

args={}

rng_key, rng_key_predict = random.split(random.PRNGKey(2))
args["rng_key"]=random.PRNGKey(20)
args["batch_size"]= 1

T=50+30
args['T']=T
args['a_0']=.5
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
alpha = .6
beta = .7
args['alpha']=alpha
args['beta']=beta


data_name='LGCP_only'               ## LGCP_only or Hawkes or LGCP_Hawkes
args['background_simulation']='LGCP_only'  ## LGCP_only or constant  or LGCP

num_reps=100

for i in range(num_reps):  

  if load_data:
    print('Loading dataset', i) 
    with open('data/'+data_name+'.pkl', 'rb') as file:
      output_dict = dill.load(file)

      simulated_output_Hawkes=output_dict['simulated_output_Hawkes'+str(i)]
      simulated_output_Hawkes_train_test=output_dict['simulated_output_Hawkes_train_test'+str(i)]
      simulated_output_Hawkes.keys()
      args_train=output_dict['args_train']
      args=output_dict['args']
      data_name=output_dict['data_name']
      a_0_true=args['a_0'] #simulated_output_background['a_0'];print(a_0_true)
      n_obs=simulated_output_Hawkes['G_tot_t'].size
      rate_xy_events_true=np.exp(a_0_true)*np.ones(n_obs)
      b_0_true=args['b_0']#simulated_output_background['b_0'];print(b_0_true)

  if not(load_data):
    print('Simulating dataset ',i)
    if args['background_simulation']=='constant':  
      rng_key, rng_key_predict = random.split(random.PRNGKey(22))
      ST_background_predictive = Predictive(generate_background_uniform_events, num_samples=args["batch_size"])
      simulated_output_background = ST_background_predictive(rng_key_predict, T=args['T'], a_0=args['a_0'], b_0=args['b_0'])
      n_obs_back=simulated_output_background['N']
      t_events_background=np.sort(simulated_output_background['t_events'][0])
      xy_events_background=simulated_output_background['xy_events'][0]
      a_0_true=args['a_0'] #simulated_output_background['a_0'];print(a_0_true)
      rate_xy_events_true=np.exp(a_0_true)*np.ones(n_obs_back)
      b_0_true=args['b_0']#simulated_output_background['b_0'];print(b_0_true)


  if not(load_data):
    if args['background_simulation'] in ['LGCP','LGCP_only']:
      rng_key, rng_key_predict = random.split(random.PRNGKey(3))

      n_t=80
      T=80
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

      ST_LGCP_predictive = Predictive(spatiotemporal_LGCP, num_samples=args["batch_size"])
      simulated_output_background = ST_LGCP_predictive(rng_key_predict, args=args, x_t=x_t, x_xy=x_xy, T=T, a_0=args['a_0'], b_0=0,  var_t=1, length_t=10, var_xy=1, length_xy=.25)

      ## parameters 
      a_0_true=simulated_output_background['a_0'];print('a_0_true',a_0_true)
      b_0_true=simulated_output_background['b_0'];
      n_obs_back= simulated_output_background['t_i'][0].size
      print('There are',n_obs_back, 'background events')

      ## full size GP and rate
      ft_true=simulated_output_background['f_t'].reshape(n_t)
      rate_t_true=simulated_output_background['rate_t']
      f_xy_true=simulated_output_background['f_xy']
      rate_xy_true=simulated_output_background['rate_xy']

      #sampled events
      sorted_ind=np.argsort(simulated_output_background['t_i'][0])

      t_events_background=simulated_output_background['t_i'][0][sorted_ind]
      rate_t_events_true=simulated_output_background['rate_t_i'][0][sorted_ind]
      ind_back_t=simulated_output_background['ind_t_i'][0][sorted_ind]

      xy_events_background=simulated_output_background['xy_i'][0][sorted_ind]
      rate_xy_events_true=simulated_output_background['rate_xy_i'][0][sorted_ind]
      ind_back_xy=simulated_output_background['ind_xy_i'][0][sorted_ind]


  if not(load_data):
    if args['background_simulation']=='constant':  
      rng_key, rng_key_predict = random.split(random.PRNGKey(2))
      ST_background_predictive = Predictive(generate_background_uniform_events, num_samples=args["batch_size"])
      simulated_output_background = ST_background_predictive(rng_key_predict, T=args['T'], a_0=args['a_0'], b_0=args['b_0'])
      n_obs_back=simulated_output_background['N']
      t_events_background=np.sort(simulated_output_background['t_events'][0])
      xy_events_background=simulated_output_background['xy_events'][0]
      a_0_true=args['a_0'] #simulated_output_background['a_0'];print(a_0_true)
      rate_xy_events_true=np.exp(a_0_true)*np.ones(n_obs_back)
      b_0_true=args['b_0']#simulated_output_background['b_0'];print(b_0_true)


  if not(load_data):
    if args['background_simulation'] in ['LGCP','LGCP_only']:
      # could change the random seed here
      rng_key, rng_key_predict = random.split(random.PRNGKey(22))

      n_t=80
      T=80
      x_t = jnp.arange(0, T, T/n_t)
      args[ "n_t"]=n_t
      args["x_t"]=x_t

      n_xy = 25
      grid = jnp.arange(0, 1, 1/n_xy)
      u, v = jnp.meshgrid(grid, grid)
      x_xy = jnp.array([u.flatten(), v.flatten()]).transpose((1, 0))
      args['x_xy']=x_xy
      args["n_xy"]= n_xy
      length_xy=.5
      
      args["gp_kernel"]=exp_sq_kernel
      args["batch_size"]= 1

      ST_LGCP_predictive = Predictive(spatiotemporal_LGCP, num_samples=args["batch_size"])
      simulated_output_background = ST_LGCP_predictive(rng_key_predict, args, x_t=x_t, x_xy=x_xy, T=T, a_0=args['a_0'], b_0=0,  var_t=1, length_t=10, var_xy=1, length_xy=length_xy)

      ## parameters 
      a_0_true=simulated_output_background['a_0'];print(a_0_true)
      b_0_true=simulated_output_background['b_0'];
      n_obs_back= simulated_output_background['t_i'][0].size
      print('There are',n_obs_back, 'background events')

      ## full size GP and rate
      ft_true=simulated_output_background['f_t'].reshape(n_t)
      rate_t_true=simulated_output_background['rate_t']
      f_xy_true=simulated_output_background['f_xy']
      rate_xy_true=simulated_output_background['rate_xy']

      #sampled events
      sorted_ind=np.argsort(simulated_output_background['t_i'][0])

      t_events_background=simulated_output_background['t_i'][0][sorted_ind]
      rate_t_events_true=simulated_output_background['rate_t_i'][0][sorted_ind]
      ind_back_t=simulated_output_background['ind_t_i'][0][sorted_ind]

      xy_events_background=simulated_output_background['xy_i'][0][sorted_ind]
      rate_xy_events_true=simulated_output_background['rate_xy_i'][0][sorted_ind]
      ind_back_xy=simulated_output_background['ind_xy_i'][0][sorted_ind]


  if data_name in ['Hawkes', 'LGCP_Hawkes']:
    if not(load_data):
      rng_key, rng_key_predict = random.split(random.PRNGKey(22))
      ST_Hawkes_predictive = Predictive(generate_spatiotemporal_offspring, num_samples=args["batch_size"])
      simulated_output_Hawkes = ST_Hawkes_predictive(rng_key_predict, args, t_star=t_events_background, s_star=xy_events_background, alpha=alpha, beta=beta, method='remove_xy_outside_boundary')

    simulated_output=simulated_output_Hawkes
  elif data_name=='LGCP_only':
    simulated_output=simulated_output_background





  if not(load_data):
    args_train={}
    T_train=50
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


  if data_name in ['Hawkes', 'LGCP_Hawkes']:  
    train_ind=simulated_output['G_tot_t']<args_train['T']
    n_train=train_ind.sum()
    n_obs=simulated_output['G_tot_t'].size
  elif data_name =='LGCP_only':
    train_ind=simulated_output['t_i']<args_train['T']
    n_train=train_ind.sum()
    n_obs=simulated_output['t_i'].size


  print('n_train', n_train, 'n_obs', n_obs, 'n_test points', n_obs-n_train)
  
  




  if data_name =='LGCP_only':
    print('train-test split for', data_name)
    ord=np.argsort(simulated_output['t_i'][0]).astype(int)
    simulated_output_train_test={} 
    simulated_output['G_tot_t']=simulated_output['t_i'][0][ord].reshape(1,n_obs)
    simulated_output['G_tot_x']=simulated_output['xy_i'][:,:,0][0][ord].reshape(1,n_obs)
    simulated_output['G_tot_y']=simulated_output['xy_i'][:,:,1][0][ord].reshape(1,n_obs)
    simulated_output['ind_t_i']=simulated_output['ind_t_i'][0][ord].reshape(1,n_obs)
    simulated_output['ind_xy_i']=simulated_output['ind_xy_i'][0][ord].reshape(1,n_obs)
    simulated_output['rate_t_i']=simulated_output['rate_t_i'][0][ord].reshape(1,n_obs)
    simulated_output['rate_xy_i']=simulated_output['rate_xy_i'][0][ord].reshape(1,n_obs)  

    simulated_output_train_test['G_tot_t_train']=simulated_output['G_tot_t'][:,0:n_train]
    simulated_output_train_test['G_tot_t_test']=simulated_output['G_tot_t'][:,n_train:n_obs]

    simulated_output_train_test['G_tot_x_train']=simulated_output['G_tot_x'][:,0:n_train]
    simulated_output_train_test['G_tot_x_test']=simulated_output['G_tot_x'][:,n_train:n_obs]

    simulated_output_train_test['G_tot_y_train']=simulated_output['G_tot_y'][:,0:n_train]
    simulated_output_train_test['G_tot_y_test']=simulated_output['G_tot_y'][:,n_train:n_obs]

    simulated_output_train_test['ind_xy_i']=simulated_output['ind_xy_i'][:,0:n_train]
    simulated_output_train_test['ind_t_i']=simulated_output['ind_t_i'][:,0:n_train]
    simulated_output_Hawkes_train_test=simulated_output_train_test
    simulated_output_Hawkes=simulated_output

  elif data_name in ['LGCP_Hawkes','Hawkes']:
    print('train-test split for', data_name)
    simulated_output_Hawkes_train_test={}
    simulated_output_Hawkes_train_test['G_tot_t_train']=simulated_output_Hawkes['G_tot_t'][:,0:n_train]
    simulated_output_Hawkes_train_test['G_tot_t_test']=simulated_output_Hawkes['G_tot_t'][:,n_train:n_obs]

    simulated_output_Hawkes_train_test['G_tot_x_train']=simulated_output_Hawkes['G_tot_x'][:,0:n_train]
    simulated_output_Hawkes_train_test['G_tot_x_test']=simulated_output_Hawkes['G_tot_x'][:,n_train:n_obs]

    simulated_output_Hawkes_train_test['G_tot_y_train']=simulated_output_Hawkes['G_tot_y'][:,0:n_train]
    simulated_output_Hawkes_train_test['G_tot_y_test']=simulated_output_Hawkes['G_tot_y'][:,n_train:n_obs]

    simulated_output_Hawkes_train_test['index_array_train']=simulated_output_Hawkes['index_array'][:,0:n_train]


  args['background']=args['background_simulation']
  #data_name='LGCP_Hawkes'
  print(data_name)

  if not(load_data):
    print('saving')
    if i==0:
      output_dict = {}
      output_dict['data_name']=data_name
    #output_dict['t_events_total']=t_events_total
    #output_dict['xy_events_total']=xy_events_total
    #output_dict['xy_events_background']=xy_events_background
    #output_dict['t_events_background']=t_events_background
      output_dict['args']=args
      output_dict['args_train']=args_train
    output_dict['simulated_output_background '+str(i)]=simulated_output_background
    output_dict['simulated_output_Hawkes'+str(i)]=simulated_output_Hawkes
    output_dict['simulated_output_Hawkes_train_test'+str(i)]=simulated_output_Hawkes_train_test
    with open('data/'+data_name+'.pkl', 'wb') as handle:
        dill.dump(output_dict, handle)



