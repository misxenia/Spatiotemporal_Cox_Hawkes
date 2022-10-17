# 
# This script runs inference on gunshot 2013 data for the experiment int he paper
# 
# Need to specify the MCMC parameters and
# the model parameters (on the priors) 
# in the spatiotemporal hawkes function


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
import pandas as pd

from utils import * 
from inference_functions import *
from functions import *


df = pd.read_csv (r'data/gunfire_data_filtered_2006_2013.csv')
print (df)
index_2013=(pd.to_datetime(df['date']).dt.year==2013)#&(pd.to_datetime(df['date']).dt.month==4)
indices=index_2013
df['month']=pd.to_datetime(df['date']).dt.month
import dill
name='GUNFIRE_2013'
output_dict = {}
output_dict['data_name']='gunfire-2013'
output_dict['x']=df[indices]['X']
output_dict['Y']= df[indices]['Y']
with open('data/'+name+'.pkl', 'wb') as handle:
    dill.dump(output_dict, handle)



indices_train=indices
n_stop=24134
indices_train[n_stop:]=False

# arguments
args={}
args['T']=50 #####
args['t_min']=0
args['x_min']=0
args['x_max']=1
args['y_min']=0
args['y_max']=1
args['background']='LGCP'



t_events_total=((df['T'][indices_train]-df['T'][indices_train].min())).to_numpy()
t_events_total/=t_events_total.max()
t_events_total*=50
t_events_total

x_events_total=(df['X'][indices_train]-df['X'][indices_train].min()).to_numpy()
x_events_total/=x_events_total.max()
x_events_total

y_events_total=(df['Y'][indices_train]-df['Y'][indices_train].min()).to_numpy()
y_events_total/=y_events_total.max()
y_events_total

xy_events_total=np.array((x_events_total,y_events_total)).transpose()


if args['background']=='LGCP':
  rng_key, rng_key_predict = random.split(random.PRNGKey(10))

  n_t=50
  T=50
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


if args['background']=='LGCP':
  indices_t=find_index(t_events_total, x_t)
  indices_xy=find_index(xy_events_total, x_xy)

if args['background']=='LGCP':
  args['indices_t']=indices_t
  args['indices_xy']=indices_xy



# temporal VAE training arguments
args["hidden_dim_temporal"]= 35
args["z_dim_temporal"]= 11
args["T"]=T
# spatial VAE training arguments
args["hidden_dim1_spatial"]= 35
args["hidden_dim2_spatial"]= 30
args["z_dim_spatial"]=10
n_xy=25


## I load my 1D temporal trained decoder parameters to generate GPs with hyperparameters that make sense in this domain
# Load 
# fixed lengthscale=10, var=1, T50
with open('decoders/decoder_1d_T50_fixed_ls', 'rb') as file:
    decoder_params = pickle.load(file)
    print(len(decoder_params))

args["decoder_params_temporal"] = decoder_params
#args["indices"]=indices

#@title
n=n_xy
if args['background']=='LGCP':
  #Load 2d spatial trained decoder
  with open('./decoders/decoder_2d_n25_infer_hyperpars'.format(n), 'rb') as file:
      decoder_params = pickle.load(file)
      print(len(decoder_params))

  args["decoder_params_spatial"] = decoder_params


t_events=t_events_total;xy_events=xy_events_total



# MCMC inference
args["num_warmup"]= 500
args["num_samples"] = 1000
args["num_chains"] =1
args["thinning"] =1


args["t_events"]=t_events_total
args["xy_events"]=xy_events_total.transpose()
#args['background']='LGCP'
rng_key, rng_key_post, rng_key_pred = random.split(rng_key, 3)

# inference
mcmc = run_mcmc(rng_key_post, spatiotemporal_hawkes_model, args)
mcmc_samples=mcmc.get_samples()


import dill
output_dict = {}
output_dict['model']=spatiotemporal_hawkes_model
output_dict['samples']=mcmc.get_samples()
output_dict['mcmc']=mcmc
with open('output.pkl', 'wb') as handle:
    dill.dump(output_dict, handle)

