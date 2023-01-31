#
## This script  runs the gunshot data of 2012 with your choice of model 
# from Hawkes, LGCP, LGCP Hawkes

# Need to specify the following 
# Model parameters
# data_name='Gunfire'
# model_folder='model_LGCP_Hawkes/'

# args['background_simulation']='constant' among LGCP, LGCP_only, Poisson, constant
# depending on the above specify the model folder
# data_folder='gunfire_2012/'


# MCMC parameters
# post_samples=500 (make sure it is algined with the mcmc params of samples and warmup)
# num_warmup, thinning, samples, chains

# Prediction
# n_simul is the number of times to predict the future events


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
from inference_functions import *
from functions import *
from prediction_functions import *


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


load_data=True
data_name='Gunfire'
#args['background_simulation']='constant'


import pandas as pd
df = pd.read_csv (r'../data/gunfire_data_filtered_2006_2013.csv')
#print (df)

index_2012=(pd.to_datetime(df['date']).dt.year==2012)#&(pd.to_datetime(df['date']).dt.month==4)
#index_unique=~df['T'].duplicated()
indices=index_2012.copy()#&index_unique&index_holidays
df['month']=pd.to_datetime(df['date']).dt.month


import dill
name='GUNFIRE_2012'
output_dict = {}
output_dict['data_name']='gunfire-2012'
output_dict['x']=df[indices]['X']
output_dict['Y']= df[indices]['Y']
with open('../data/'+name+'.pkl', 'wb') as handle:
    dill.dump(output_dict, handle)



## SPLIT TRAIN TEST
n_all=np.sum(indices)
n_stop=21277
np.sum(indices[0:n_stop])
indices_train=indices.copy()
indices_train[n_stop:]=False
np.sum(indices_train)
indices_test=indices.copy()
indices_test[0:n_stop]=False
np.sum(indices_test)


args_train={}
T_train=50
T_test=T_train+30
#T=
args_train['T']=T_train

n_t_train=50
x_t_train = jnp.arange(0, T_train, T_train/n_t_train)
args_train[ "n_t"]=n_t_train
args_train["x_t"]=x_t_train
args_train['x_xy']=x_xy
args_train["n_xy"]= n_xy
args_train["gp_kernel"]=exp_sq_kernel
args_train["batch_size"]= 1

train_ind=indices_train
n_train=train_ind.sum()
n_obs=np.sum(indices)
print('n_train', n_train, 'n_obs', n_obs, 'n_test points', n_obs-n_train)

data_train_test={}
data_train_test['G_tot_t']=df[indices]['T']
data_train_test['G_tot_x']=df[indices]['X']
data_train_test['G_tot_y']=df[indices]['Y']

data_train_test['G_tot_t_train']=df[indices_train]['T']
data_train_test['G_tot_t_test']=df[indices_test]['T']

data_train_test['G_tot_x_train']=df[indices_train]['X']
data_train_test['G_tot_x_test']=df[indices_test]['X']

data_train_test['G_tot_y_train']=df[indices_train]['Y']
data_train_test['G_tot_y_test']=df[indices_test]['Y']

#args['background']=args['background_simulation']


#

#
#INFERENCE

args_train['background']='LGCP'
model_folder='model_LGCP_Hawkes/'

#args_train['background']='Poisson'
#model_folder='model_Poisson/'

#args_train['background']='constant'
#model_folder='model_Hawkes/'

#args_train['background']='LGCP_only'
#model_folder='model_LGCP/'



#@title
if args_train['background']!='constant':
  # spatial VAE training
  args_train["hidden_dim1_spatial"]= 35
  args_train["hidden_dim2_spatial"]= 30
  args_train["z_dim_spatial"]=10
  n_xy=25
#@title
if args_train['background']!='constant':
  # spatial VAE training

	args_train["hidden_dim_temporal"]= 50
	args_train["z_dim_temporal"]= 20
	with open('decoders/decoder_1d_T80_fixed_ls10', 'rb') as file:
	    decoder_params = pickle.load(file)

	args_train["decoder_params_temporal"] = decoder_params

#args["T"]=T
#args["n_t"]=T
#args['x_t']=jnp.linspace(0, args["T"], args["n_t"])
#hidden_dim = 50, z_dim = 20


#@title
if args_train['background']!='constant':
  n=n_xy
  #Load 2d spatial trained decoder
  with open('./decoders/decoder_2d_n25_infer_hyperpars'.format(n), 'rb') as file:
      decoder_params = pickle.load(file)

  args_train["decoder_params_spatial"] = decoder_params
  
  #args_train["decoder_params_spatial"]=args["decoder_params_spatial"]
t_events_total=df[indices_train]['T'].to_numpy()
x_events_total=df[indices_train]['X'].to_numpy()
y_events_total=df[indices_train]['Y'].to_numpy()



## MAP THE EVENTS IN the right domain to match the functions
min_t=df['T'][indices_train].min()
#print(min_t)
max_t=((df['T'][indices_train]-df['T'][indices_train].min())).max()
#print(max_t)

const_t=1/50*159.99097222222963+2532.87083333333
const_t=1/50*max_t+min_t
#print(const_t)

t_events_total=((df['T'][indices_train]-df['T'][indices_train].min())).to_numpy()
t_events_total/=t_events_total.max()
t_events_total*=50


min_x=df['X'][indices_train].min()
#print(min_x)
max_x=(df['X'][indices_train]-df['X'][indices_train].min()).to_numpy().max()
#print(max_x)

#const_x=1*11.883611320812063+0.521244924977036
const_x=1*max_x+min_x
const_x

x_events_total=(df['X'][indices_train]-df['X'][indices_train].min()).to_numpy()
x_events_total/=x_events_total.max()

min_y=df['Y'][indices_train].min()
print(min_y)
max_y=(df['Y'][indices_train]-df['Y'][indices_train].min()).to_numpy().max()
print(max_y)

#const_y=1*11.883611320812063+0.521244924977036
const_y=1*max_y+min_y
const_y

df['Y'][indices_train].min()

const_y=1*(df['Y'][indices_train]-df['Y'][indices_train].min()).to_numpy().max()/df['Y'][indices_train].min()
const_y

y_events_total=(df['Y'][indices_train]-df['Y'][indices_train].min()).to_numpy()
y_events_total/=y_events_total.max()

xy_events_total=np.array((x_events_total,y_events_total)).transpose()



#### MCMC inference
print(T_test)
print(T_train)

args_train["num_warmup"]= 100
args_train["num_samples"] =500
args_train["num_chains"] =1
args_train["thinning"] =1


args_train["t_events"]=t_events_total
args_train["xy_events"]=xy_events_total.transpose()


if args_train['background']!='constant':
  indices_t=find_index(t_events_total, args_train['x_t'])
  indices_xy=find_index(xy_events_total, args_train['x_xy'])
  args_train['indices_t']=indices_t
  args_train['indices_xy']=indices_xy


## which is the model you ran
rng_key, rng_key_post, rng_key_pred = random.split(rng_key, 3)
print(args_train['background'])
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


# inference
run_inference=True
if run_inference:
  tic = time.time()
  mcmc = run_mcmc(rng_key_post, model_mcmc, args_train)
  print('MCMC (numpyro) compiling time:', time.time() - tic, '\n')
  train_runtime=time.time()-tic
  mcmc_samples=mcmc.get_samples()


save_me=True
data_folder='gunfire_2012_new/'
print('model_folder',model_folder)
#model_folder='model_LGCP/'

if run_inference and save_me:
  filename='output/'+data_folder+model_folder
  import dill
  output_dict = {}
  output_dict['model']=spatiotemporal_hawkes_model
  #output_dict['guide']=guide
  output_dict['samples']=mcmc.get_samples()
  output_dict['mcmc']=mcmc
  with open(filename+'output.pkl', 'wb') as handle:
      dill.dump(output_dict, handle)

if not(run_inference):
  filename='output/'+data_folder+model_folder
  import dill
  print('output/'+data_folder+model_folder)
  with open(filename+'output2.pkl', 'rb') as handle:
    dill.load(output_dict, handle)
  output_dict = {}
  #output_dict['model']=spatiotemporal_hawkes_model
  #output_dict['guide']=guide
  mcmc_samples=output_dict['samples']
  mcmc=output_dict['mcmc']



fig, ax = plt.subplots(1, 2,figsize=(15,5))
if 'alpha' in mcmc_samples.keys():
  ax[0].plot(mcmc_samples['alpha'])
  ax[0].set_xlabel('alpha')
  #ax[0].axhline(alpha,color='red')
  ax[1].hist(mcmc_samples['alpha'],bins=150,density=True)
if save_me:
  mypath='alpha.png'
  plt.savefig(filename+mypath)


fig, ax = plt.subplots(1, 2,figsize=(15,5))
if 'beta' in mcmc_samples.keys():
  ax[0].plot(mcmc_samples['beta'])
  ax[0].set_xlabel('beta')
  #ax[0].axhline(alpha,color='red')
  ax[1].hist(mcmc_samples['beta'],bins=150,density=True)
if save_me:
  mypath='beta.png'
  plt.savefig(filename+mypath)


fig, ax = plt.subplots(2, 2,figsize=(15,5))
if 'sigmax_2' in mcmc_samples.keys():
  
  ax[0,0].plot(mcmc_samples['alpha'])
  ax[0,0].set_ylabel('alpha')

  ax[0,1].plot(mcmc_samples['beta'])
  ax[0,1].set_ylabel('beta')

  ax[1,0].plot(mcmc_samples['a_0'])
  ax[1,0].set_ylabel('a_0')

  ax[1,1].plot(np.sqrt(mcmc_samples['sigmax_2']))
  ax[1,1].set_ylabel('sigma')

  if save_me:
    mypath='trace_plots.png'
    plt.savefig(filename+mypath)

fig, ax = plt.subplots(3, 1,figsize=(5,5))
if 'alpha' in mcmc_samples.keys():

  ax[0].plot(mcmc_samples['a_0'],label='a_0')
  ax[0].set_ylabel('a_0')

  ax[1].plot(mcmc_samples['alpha'],label='alpha')
  ax[1].set_ylabel('alpha')

  ax[2].plot(mcmc_samples['beta'])
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


predictive = Predictive(model_mcmc, mcmc_samples)
predictions = predictive(rng_key_pred, args=args_train)


if args_train['background'] in ['LGCP','LGCP_only','LGCP_Hawkes']:
  f_t_pred=predictions["f_t"]
  f_t_pred_mean=jnp.mean(f_t_pred, axis=0)
  f_t_pred_mean.shape
  f_t_hpdi = hpdi(f_t_pred, 0.9)
  f_t_hpdi.shape
#f_t_pred_mean=jnp.mean(f_t_pred, axis=0)[0:T_train]
#f_t_hpdi = hpdi(f_t_pred, 0.9)[0:T_train]


post_samples=500
a_0_post_mean=np.array(mcmc_samples['a_0'][-post_samples:].mean())
a_0_post_samples=np.array(mcmc_samples['a_0'][-post_samples:])


if args_train['background'] in ['LGCP','LGCP_only']:
  #ft_true=np.zeros(args_train['n_t'])
  #f_xy_true=np.zeros(args_train['n_xy']**2)
  rate_t_pred=np.exp(predictions['f_t'])
  rate_t_pred_mean=jnp.mean(rate_t_pred, axis=0)
  rate_t_hpdi = hpdi(rate_t_pred, 0.9)

  f_t_pred=predictions["f_t"]
  f_t_pred_mean=jnp.mean(f_t_pred, axis=0)
  f_t_hpdi = hpdi(f_t_pred, 0.9)

  fig,ax=plt.subplots(1,3,figsize=(15,5))
  ax[0].scatter(x_t[indices_t], f_t_pred_mean[indices_t], color="red", label="estimated rate at observed times")
  ax[0].set_xlabel('Interval number')
  ax[0].plot(args_train['x_t'], f_t_pred_mean, color="green", label="mean estimated rate")
  ax[0].fill_between(args_train['x_t'], f_t_hpdi[0], f_t_hpdi[1], alpha=0.4, color="palegoldenrod", label="90%CI rate")
  ax[0].legend()


  ax[1].scatter(args_train['x_t'][indices_t], np.exp(f_t_pred_mean[indices_t]), color="red", label="estimated exp(f) at observed times")
  ax[1].set_xlabel('Interval number')
  ax[1].plot(args_train['x_t'], np.exp(f_t_pred_mean), color="green", label="mean estimated exp(f)")
  ax[1].fill_between(args_train['x_t'], np.exp(f_t_hpdi[0]), np.exp(f_t_hpdi[1]), alpha=0.4, color="palegoldenrod", label="90%CI rate")
  ax[1].legend()
  ax[2].scatter(args_train['x_t'][indices_t], np.exp(f_t_pred_mean[indices_t]+a_0_post_mean), color="red", label="estimated rate at observed times")
  ax[2].set_xlabel('Interval number')
  ax[2].plot(args_train['x_t'], np.exp(f_t_pred_mean+a_0_post_mean), color="green", label="mean estimated rate")
  ax[2].fill_between(args_train['x_t'], np.exp(f_t_hpdi[0]+a_0_post_mean), np.exp(f_t_hpdi[1]+a_0_post_mean), alpha=0.4, color="palegoldenrod", label="90%CI rate")
  ax[2].legend()

  if save_me:
    mypath='f_t.png'
    plt.savefig(filename+mypath)

    

if args_train['background'] in ['LGCP','LGCP_only']:

  rate_xy_pred=np.exp(predictions['f_xy'])
  rate_xy_pred_mean=jnp.mean(rate_xy_pred, axis=0)
  rate_xy_hpdi = hpdi(rate_xy_pred, 0.9)

  f_xy_pred=predictions["f_xy"]
  f_xy_pred_mean=jnp.mean(f_xy_pred, axis=0)
  f_xy_hpdi = hpdi(f_xy_pred, 0.9)

  fig, ax = plt.subplots(1,2, figsize=(10, 10))
  #fig.show()
  _min, _max = np.amin(f_xy_pred), np.amax(f_xy_pred)
  im = ax[0].imshow(f_xy_pred_mean.reshape(n,n), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
  ax[0].title.set_text('Estimated f_xy')
  #fig.show()
  rate_xy_pred_norm=rate_xy_pred_mean/np.sum(rate_xy_pred)
  _min, _max = np.amin(rate_xy_pred_norm), np.amax(rate_xy_pred_norm)
  im = ax[1].imshow(rate_xy_pred_norm.reshape(n,n), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
  ax[1].title.set_text('Estimated normalized rate_xy')

  #ax[0].plot(args_train['xy_events'][0],args_train['xy_events'][1],'x',color='red')


  if save_me:
    mypath='f_xy.png'
    plt.savefig(filename+mypath)



if args_train['background'] in ['LGCP','LGCP_only']:

  rate_xy_pred=np.exp(predictions['f_xy'])
  rate_xy_pred_mean=jnp.mean(rate_xy_pred, axis=0)
  rate_xy_hpdi = hpdi(rate_xy_pred, 0.9)

  f_xy_pred=predictions["f_xy"]
  f_xy_pred_mean=jnp.mean(f_xy_pred, axis=0)
  f_xy_hpdi = hpdi(f_xy_pred, 0.9)

  fig, ax = plt.subplots(2,2, figsize=(10, 10))
  #fig.show()
  _min, _max = np.amin(f_xy_pred), np.amax(f_xy_pred)
  im = ax[0,0].imshow(f_xy_pred_mean.reshape(n,n), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
  ax[0,0].title.set_text('Estimated f_xy')

  #fig.show()
  rate_xy_pred_norm=rate_xy_pred_mean/np.sum(rate_xy_pred)
  _min, _max = np.amin(rate_xy_pred_norm), np.amax(rate_xy_pred_norm)
  im = ax[0,1].imshow(rate_xy_pred_norm.reshape(n,n), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
  ax[0,1].title.set_text('Estimated normalized rate_xy')

  #fig.show()
  _min, _max = np.amin(f_xy_pred), np.amax(f_xy_pred)
  im = ax[1,0].imshow(f_xy_pred_mean.reshape(n,n), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
  ax[1,0].title.set_text('Estimated f_xy')
  ax[1,0].plot(args_train['xy_events'][0],args_train['xy_events'][1],'x',color='red')

  #fig.show()
  rate_xy_pred_norm=rate_xy_pred_mean/np.sum(rate_xy_pred)
  _min, _max = np.amin(rate_xy_pred_norm), np.amax(rate_xy_pred_norm)
  im = ax[1,1].imshow(rate_xy_pred_norm.reshape(n,n), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
  ax[1,1].title.set_text('Estimated normalized rate_xy')
  ax[1,1].plot(args_train['xy_events'][0],args_train['xy_events'][1],'x',color='red')


  if save_me:
    mypath='f_xy_and_true.png'
    plt.savefig(filename+mypath)



fig, ax = plt.subplots(1, 1,figsize=(5,5))
ax.plot(mcmc_samples['loglik'])
print('The mean train LOGLIKELIHOOD is ',mcmc_samples['loglik'].mean(), 'for model with background', args_train['background'], 'run with function', model_mcmc)
LOGLIK_TRAIN=mcmc_samples['loglik'].mean()

if save_me:
  mypath='loglik.png'
  plt.savefig(filename+mypath)
  mypath='loglik.eps'
  plt.savefig(filename+mypath)
  
#### PREDICTION OF FUTURE EVENTS and evaluation of loglikelihood 


n_total=int(args_train['num_samples']/args_train['thinning']*args_train['num_chains'])
n_total

post_samples=500
a_0_post_mean=np.array(mcmc_samples['a_0'][n_total-post_samples:n_total].mean())
a_0_post_samples=np.array(mcmc_samples['a_0'][n_total-post_samples:n_total])


if args_train['background'] in ['LGCP','constant']:# ie in hawkes case
  alpha_post_mean=np.array(mcmc_samples['alpha'][n_total-post_samples:n_total].mean())
  alpha_post_samples=np.array(mcmc_samples['alpha'][n_total-post_samples:n_total])
  
  beta_post_mean=np.array(mcmc_samples['beta'][n_total-post_samples:n_total].mean())
  beta_post_samples=np.array(mcmc_samples['beta'][n_total-post_samples:n_total])

  sigma_x_2_post_mean=np.array(mcmc_samples['sigmax_2'][n_total-post_samples:n_total].mean())
  sigma_x_2_post_samples=np.array(mcmc_samples['sigmax_2'][n_total-post_samples:n_total])
  LOGLIK_TEST=0




if args_train['background'] in ['LGCP','LGCP_only']:
  def normal_dist(mean,var,num_samples=args_train["z_dim_temporal"]):
    z_temporal=numpyro.sample("z_temporal",dist.Normal(mean, var).expand([num_samples]))
    
  normal_predictive = Predictive(normal_dist, num_samples=1)
  normal_predictive_samples = normal_predictive(rng_key, mean=.5, var=2)


  #z_temporal = numpyro.sample("z_temporal", dist.Normal(jnp.zeros(args["z_dim_temporal"]), jnp.ones(args["z_dim_temporal"])))
  z_temporal= normal_predictive(rng_key,jnp.zeros(args_train["z_dim_temporal"]), jnp.ones(args_train["z_dim_temporal"]))
  decoder_nn_temporal = vae_decoder_temporal(args_train["hidden_dim_temporal"], args["n_t"])  
  decoder_params = args_train["decoder_params_temporal"]
  v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal['z_temporal']))



args_test={}
n_simul=100
args_test['n_t']=80
args_test['x_t']=np.arange(0,T_test,1)
args_test['T']=80
args_test['n_xy']=args['n_xy']
args_test['x_xy']=args['x_xy']
args_test['a_0']=a_0_post_mean

if args_train['background'] in ['LGCP','LGCP_only','LGCP_Hawkes']:
  args_test['hidden_dim_temporal']=args_train["hidden_dim_temporal"]
  args_test['z_dim_temporal']=args_train["z_dim_temporal"]
  args_test['decoder_params_temporal']=args_train["decoder_params_temporal"]
  args_test['z_dim_spatial']=args_train["z_dim_spatial"]
  args_test['hidden_dim1_spatial']=args_train["hidden_dim1_spatial"]
  args_test['hidden_dim2_spatial']=args_train["hidden_dim2_spatial"]
  args_test['decoder_params_spatial']=args_train["decoder_params_spatial"]


  args_test['a_0_post_mean']=a_0_post_mean
  args_test['z_temporal']=mcmc_samples['z_temporal']
  args_test['z_spatial']=mcmc_samples['z_spatial']
  args_test['n_total']=n_total
  args_test["n_xy"]=n_xy

  GP_predictive = Predictive(sample_GP, num_samples=n_simul)
  GP_predictive_samples = GP_predictive(rng_key, args_test, 'post')

  #plt.plot(np.exp(GP_predictive_samples['f_t'][0]+a_0_post_mean))
  #plt.hist(simulated_output_Hawkes['G_tot_t'][0])
  plt.plot(GP_predictive_samples['f_t'][0])
  
  if save_me:
    mypath='GPt_post_test.eps'
    plt.savefig(filename+mypath)
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
    mypath='GPxy_post_test.eps'
    plt.savefig(filename+mypath)
    mypath='GPxy_post_test.png'
    plt.savefig(filename+mypath)

if args_train['background'] in ['LGCP','LGCP_only']:
  args_test['Itot_xy']=np.array(GP_predictive_samples['Itot_xy'])


past_times=t_events_total
past_locs=xy_events_total.transpose()
n_test=n_obs-n_train;print(n_test)
N_new=n_test

lambda_0_post_samples=np.exp(a_0_post_samples);
lambda_0_post_mean=lambda_0_post_samples.mean()

args_test['x_t']=np.arange(50,80,1)

post_mean=True



if args_train['background']=='Poisson':
  post_samples=100
  a_0_post_mean=np.array(mcmc_samples['a_0'][n_total-post_samples:n_total].mean())
  a_0_post_samples=np.array(mcmc_samples['a_0'][n_total-post_samples:n_total])
    
  args_new={}
  args_new['t_min']=50
  args_new['t_max']=80
  args_new['a_0']=a_0_post_mean
  args_new['b_0']=0
  args_new['t_events']=None
  args_new['xy_events']=None
  
  args_prior={}
  args_prior['t_events']=None
  args_prior['xy_events']=None
  args_prior['t_min']=50
  args_prior['t_max']=80
  #loglik_test=a_0_post_mean*n_test-jnp.exp(a_0_post_samples)*(args_prior['t_max']-args_prior['t_min'])
  #LOGLIK_TEST[j]=a_0_post_samples*n_test-jnp.exp(a_0_post_samples)*(args_prior['t_max']-args_prior['t_min'])
  
if args_train['background'] in ['LGCP','LGCP_only','LGCP_Hawkes']:
  f_t_pred_mean=np.mean(GP_predictive_samples['f_t'],0)
  f_xy_pred_mean=np.mean(GP_predictive_samples['f_xy'],0)
 
#
n_simul=100

simulate_predictions=True
if simulate_predictions:

  PREDICTIONS={};
  PREDICTIONS['T']=np.zeros((n_simul,n_test));PREDICTIONS['X']=np.zeros((n_simul,n_test));PREDICTIONS['Y']=np.zeros((n_simul,n_test))
  LOGLIK_TEST=np.zeros((n_simul,2))

  if args_train['background'] in ['LGCP','LGCP_only']:
    rng_key, rng_key_predict = random.split(random.PRNGKey(2))
    #gp_predictive = Predictive(spatiotemporal_GP, num_samples=n_simul)
    print(T/n_t,'\n\n')
    #GP_prior_samples = gp_predictive(rng_key_predict, args['T'], args['x_t'], args['x_xy'], gp_kernel=exp_sq_kernel, jitter=1e-5, a_0=0, b_0=0,  var_t=1, length_t=10, var_xy=1, length_xy=.25)


  for j in range(0,n_simul):
    if j%10 ==0:
      print('sample',j)
      #simulate from the underlying process 
    if args_train['background']=='LGCP':
      print('predicting from LGCP-HAWKES')

      if post_mean:
        print(T_train)
        T_pred, X_pred, Y_pred, T_pred_all,X_pred_all,Y_pred_all=simulate_spatiotemporal_hawkes_predictions(past_times, 
          past_locs, N_new,  args['x_min'], args['x_max'], args['y_min'], args['y_max'], lambda_0_post_mean, alpha_post_mean, beta_post_mean, sigma_x_2_post_mean, 
          GP_predictive_samples['Itot_xy'][0], args_train['background'], f_t_pred_mean)   
      else:
        T_pred, X_pred, Y_pred, T_pred_all,X_pred_all,Y_pred_all=simulate_spatiotemporal_hawkes_predictions(past_times, 
        past_locs, N_new, args['x_min'], args['x_max'], args['y_min'], args['y_max'], lambda_0_post_samples[j], alpha_post_samples[j], beta_post_samples[j], sigma_x_2_post_samples[j], 
        GP_predictive_samples['Itot_xy'][0], args_train['background'], np.array(GP_predictive_samples['f_t'][j])) 
      
      ###loglikelihood
      args_test=args
      args_test['background']='LGCP'
      args_test['alpha']=alpha_post_mean
      args_test['beta']=beta_post_mean
      args_test['a_0']=np.log(lambda_0_post_mean)
      args_test['sigmax_2']=sigma_x_2_post_mean
      args_test['sigmay_2']=sigma_x_2_post_mean
      args['t_events']=T_pred
      args['xy_events']=np.array((X_pred,Y_pred))
      Hawkes_lik_predictive = Predictive(Hawkes_likelihood, num_samples=1)
      Hawkes_likelihood=Hawkes_lik_predictive(rng_key, args_test)
      LOGLIK_TEST[j,1]=Hawkes_likelihood['loglik']


    elif args_train['background']=='Poisson':
        ST_background_predictive = Predictive(spatiotemporal_homogenous_poisson, num_samples=args["batch_size"])
        simulated_output_background = ST_background_predictive(rng_key_predict,args_new)
        n_obs_back=simulated_output_background['N']
        T_pred=np.sort(simulated_output_background['t_events'])[0,0:n_test]
        X_pred=simulated_output_background['xy_events'][0,0:n_test,0]
        Y_pred=simulated_output_background['xy_events'][0,:n_test,1]  
        ###loglikelihood
        print(T_test)
        print(args_test['T'])
        LOGLIK_TEST[j,0]=a_0_post_samples[j]*n_test-jnp.exp(a_0_post_samples[j])*(T_test-T_train)
        LOGLIK_TEST[j,1]=a_0_post_mean*n_test-jnp.exp(a_0_post_mean)*(T_test-T_train)
        

    
    elif args_train['background']=='constant':
      if post_mean:
        T_pred, X_pred, Y_pred, T_pred_all, X_pred_all, Y_pred_all=simulate_spatiotemporal_hawkes_predictions(past_times, 
          past_locs, N_new, args['x_min'], args['x_max'], args['y_min'], args['y_max'], lambda_0_post_mean, alpha_post_mean, beta_post_mean, sigma_x_2_post_mean, 0,
          args_train['background'])  
      else:
        T_pred, X_pred, Y_pred, T_pred_all,X_pred_all,Y_pred_all=simulate_spatiotemporal_hawkes_predictions(past_times, 
          past_locs, N_new, args['x_min'], args['x_max'], args['y_min'], args['y_max'], lambda_0_post_samples[j], alpha_post_samples[j], beta_post_samples[j], sigma_x_2_post_samples[j],0, 
          args_train['background'])  
      
      ###loglikelihood
      args_test=args
      args_test['background']='constant'
      args_test['alpha']=alpha_post_mean
      args_test['beta']=beta_post_mean
      args_test['a_0']=np.log(lambda_0_post_mean)
      args_test['sigmax_2']=sigma_x_2_post_mean
      args_test['sigmay_2']=sigma_x_2_post_mean
      args_test['t_events']=T_pred
      args_test['xy_events']=np.array((X_pred,Y_pred))
      Hawkes_lik_predictive = Predictive(Hawkes_likelihood, num_samples=1)
      Hawkes_lik=Hawkes_lik_predictive(rng_key, args_test)
      LOGLIK_TEST[j,1]=Hawkes_lik['loglik']

      

        
    elif args_train['background']=='LGCP_only':
      if post_mean:
        N_0=n_test
        ind_t_i, t_i, rate_t_i=rej_sampling_new(N_0, np.arange(args_train['T'], T_test,1), lambda_0_post_mean*np.exp(f_t_pred_mean[T_train:]), 30)
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
        ind_t_i, t_i, rate_t_i=rej_sampling_new(N_0, np.arange(args_train['T'], T_test,1), lambda_0_post_samples[j]*np.exp(GP_predictive_samples['f_t'][j][args_train['T']:]), 30)
        N_0 = t_i.shape[0]
        ind_xy_i, xy_i, rate_xy_i=rej_sampling_new(N_0, args['x_xy'], GP_predictive_samples['rate_xy'][j,:], args['n_xy']**2)
        ord=t_i.sort()
        T_pred=t_i[ord].flatten()
        X_pred=xy_i[:,0][ord].flatten()
        Y_pred=xy_i[:,1][ord].flatten()
        #T_pred_all=np.concatenate((past_times,T_pred.flatten()))
        #X_pred_all=np.concatenate((past_locs[0],X_pred.flatten()))
        #Y_pred_all=np.concatenate((past_locs[1],Y_pred.flatten()))
      
      ###loglikelihood
      Itot_t=jnp.sum(jnp.exp(GP_predictive_samples['f_t'][j][args_train['T']:]))/args["n_t"]*(T_test-T_train)
      Itot_xy=jnp.sum(jnp.exp(GP_predictive_samples['f_xy'][j]))/args_train["n_xy"]**2
      LOGLIK_TEST[j,0]=Itot_xy*Itot_t*lambda_0_post_mean

      Itot_t_mean=jnp.sum(jnp.exp(f_t_pred_mean[args_train['T']:]))/args["n_t"]*(T_test-T_train)
      Itot_xy_mean=jnp.sum(jnp.exp(f_xy_pred_mean['f_xy']))/args_train["n_xy"]**2
      LOGLIK_TEST[j,0]=Itot_t_mean*Itot_xy_mean*lambda_0_post_samples[j]
      
    
    len_t=T_pred.size
    PREDICTIONS['T'][j][0:len_t]=T_pred#/50*160+2532.87
    #PREDICTIONS['T'][j]=T_pred/50*160+2532.87
    
    PREDICTIONS['X'][j][0:len_t]=X_pred#*const_x
    PREDICTIONS['Y'][j][0:len_t]=Y_pred#*const_y
    

    
  min_y_test=df['Y'][indices_test].min()
  print(min_y_test)
  max_y_test=(df['Y'][indices_test]-df['Y'][indices_test].min()).to_numpy().max()
  print(max_y_test);print(max_y)


  #### this is mapping we want
  #T_pred/50*max_t+min_t
  PREDICTIONS_SCALED={};
  PREDICTIONS_SCALED['T']=np.zeros((n_simul,n_test));
  PREDICTIONS_SCALED['X']=np.zeros((n_simul,n_test));
  PREDICTIONS_SCALED['Y']=np.zeros((n_simul,n_test))

  PREDICTIONS_SCALED['T']=PREDICTIONS['T']/50*max_t+min_t
  PREDICTIONS_SCALED['X']=PREDICTIONS['X']*max_x+min_x
  PREDICTIONS_SCALED['Y']=PREDICTIONS['Y']*max_y_test+min_y_test


  ErrorA=np.zeros(n_simul);ErrorA_t=np.zeros(n_simul);ErrorA_xy=np.zeros(n_simul)
  fig,ax=plt.subplots()
  DIFF=np.zeros(n_simul)
  for j in range(n_simul): 
    T_pred=PREDICTIONS_SCALED['T'][j]
    X_pred=PREDICTIONS_SCALED['X'][j]
    Y_pred=PREDICTIONS_SCALED['Y'][j]
    
    
    n_stop=10
    ind=np.arange(n_stop)
    Et=square_mean(T_pred[ind],data_train_test['G_tot_t_test'][:n_stop].to_numpy())
    Ey=square_mean(Y_pred[ind],data_train_test['G_tot_y_test'][:n_stop].to_numpy())
    Ex=square_mean(X_pred[ind],data_train_test['G_tot_x_test'][:n_stop].to_numpy())
    ErrorA_t[j]=np.sqrt(+Et)
    ErrorA_xy[j]=np.sqrt(Ex+Ey)
    ErrorA[j]=np.sqrt(Ex+Ey)+np.sqrt(+Et)

    ax.plot(T_pred[0:n_stop],'red')
    ax.plot(data_train_test['G_tot_t_test'][0:n_stop].to_numpy())


  print('Root Mean Square Error A for time \n', np.mean(ErrorA_t),'with variance', np.std(ErrorA_t),'\n')
  print('Root Mean Square Error A for space xy\n', np.mean(ErrorA_xy), 'with variance', np.std(ErrorA_xy), '\n')
  print('Root Mean Square Error A for time and space\n', np.mean(ErrorA), 'with variance', np.std(ErrorA), '\n')


print('Loglikelihood train set is\n', LOGLIK_TRAIN)
print('Loglikelihood test set is\n', LOGLIK_TEST.mean(), 'with std',  np.std(LOGLIK_TEST)/np.sqrt(n_simul), '\n')
print('train_runtime', train_runtime)


filename='output/'+data_folder+model_folder+"/output.txt"
f = open(filename, "a")
print("train_runtime!",train_runtime, file=f)
print("Loglikelihood test set.",LOGLIK_TEST, file=f)
print("Loglikelihood test set mean",LOGLIK_TEST[:,0].mean(), file=f)
print("Loglikelihood test set standard error",np.std(LOGLIK_TEST[:,0])/np.sqrt(n_simul), file=f)
print("Loglikelihood test set mean B",LOGLIK_TEST[:,1].mean(), file=f)
print("------------------------------------", file=f)
f.close()


