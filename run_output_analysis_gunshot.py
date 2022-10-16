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
import matplotlib.dates as mdates

myFmt = mdates.DateFormatter('%m')

df = pd.read_csv (r'data/gunfire_data_filtered_2006_2013.csv')
print (df)
#T IS TIME IN HOURS SINCE THE FIRST EVENT ON 26 JAN 2006


df.dropna(subset = ["X","Y","T"], inplace=True)
index_2013=pd.to_datetime(df['date']).dt.year==2013
indices=index_2013#&index_unique&index_holidays

fig,ax=plt.subplots(1,1,figsize=(10,7))
norm_rate_t_emp=ax.hist(df['T'][indices].to_numpy(),bins=50)
ax.xaxis.set_major_formatter(myFmt)
ax.set_xlabel('month')
filename='output/gunfire_2013/model_LGCP_Hawkes'
mypath='temporal_map.png'
plt.savefig(filename+mypath)

plt.plot(df[indices]['X'], df[indices]['Y'],'x',alpha=.4)
plt.xlabel('x')
plt.ylabel('y')
filename='output/gunfire_2013/'
mypath='spatial_map.png'
plt.savefig(filename+mypath)



# args for exponential time kernel and gaussian spatial kernel
args={}
args['T']=50 #####
args['t_min']=0
args['x_min']=0
args['x_max']=1
args['y_min']=0
args['y_max']=1


t_events_total=((df['T'][indices]-df['T'][indices].min())).to_numpy()
t_events_total/=t_events_total.max()
t_events_total*=50
t_events_total


x_events_total=(df['X'][indices]-df['X'][indices].min()).to_numpy()
x_events_total/=x_events_total.max()
x_events_total

y_events_total=(df['Y'][indices]-df['Y'][indices].min()).to_numpy()
y_events_total/=y_events_total.max()
y_events_total

xy_events_total=np.array((x_events_total,y_events_total)).transpose()


args['background']='LGCP'

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

  # MCMC inference
args["num_warmup"]= 100
args["num_samples"] = 400
args["num_chains"] =2
args["thinning"] =2


import dill, pickle
#r'data/gunfire_processed_data.csv'
with open('output/gunfire_2013/file_new.pkl', 'rb') as f:
    input_dict = dill.load(f)

mcmc=input_dict
mcmc_samples=mcmc.get_samples()


save_me=True


fig, ax = plt.subplots(3, 1,figsize=(8,5), sharex=True)

ax[0].plot(mcmc_samples['a_0'])
ax[0].set_ylabel("$a_0$")

ax[1].plot(mcmc_samples['alpha'])
ax[1].set_ylabel(r"${\alpha} $")

ax[2].plot(mcmc_samples['beta'])
ax[2].set_ylabel(r"$\beta$")
ax[2].set_xlabel("MCMC iterations")


for i in range(3):
  #for j in range(2):
    ##ax[i,j].set_xlabel('x')
    #ax[i,j].set_ylabel('y')
    for item in ([ ax[i].xaxis.label, ax[i].yaxis.label]):
        item.set_fontsize(15)


if save_me:
  mypath='trace_plots_part_A.png'
  plt.savefig(filename+mypath)

fig, ax = plt.subplots(2, 1,figsize=(8,5), sharex=True, sharey=True)

ax[0].plot(mcmc_samples['sigmax_2'])
ax[0].set_ylabel("$\sigma^2_x$")
ax[0].set_xlabel("MCMC iterations")

ax[1].plot(mcmc_samples['sigmay_2'])
ax[1].set_ylabel("$\sigma^2_y$")
ax[1].set_xlabel("MCMC iterations")


for i in range(2):
  #for j in range(2):
    ##ax[i,j].set_xlabel('x')
    #ax[i,j].set_ylabel('y')
    for item in ([ ax[i].xaxis.label, ax[i].yaxis.label]):
        item.set_fontsize(15)

if save_me:
  mypath='trace_plots_part_B.png'
  plt.savefig(filename+mypath)
  


fig, ax = plt.subplots(5, 1,figsize=(8,14), sharex=True)

ax[0].plot(mcmc_samples['a_0'])
ax[0].set_ylabel("$a_0$")

ax[1].plot(mcmc_samples['alpha'])
ax[1].set_ylabel(r"${\alpha} $")

ax[2].plot(mcmc_samples['beta'])
ax[2].set_ylabel(r"$\beta$")


ax[3].plot(mcmc_samples['sigmax_2'])
ax[3].set_ylabel("$\sigma^2_x$")

ax[4].plot(mcmc_samples['sigmay_2'])
ax[4].set_ylabel("$\sigma^2_y$")

ax[4].set_xlabel("MCMC iterations")


for i in range(5):
    for item in ([ ax[i].xaxis.label, ax[i].yaxis.label]):
        item.set_fontsize(15)

if save_me:
  mypath='trace_plots_all.png'
  plt.savefig(filename+mypath)



fig, ax = plt.subplots(5, 1,figsize=(8,14))

ax[0].hist(mcmc_samples['a_0'], bins=100,density=True)

ax[1].hist(mcmc_samples['alpha'], bins=100,density=True)
ax[1].set_xlabel(r"${\alpha} $")

ax[2].hist(mcmc_samples['beta'], bins=100,density=True)
ax[2].set_xlabel(r"$\beta$")


ax[3].hist(mcmc_samples['sigmax_2'], bins=100,density=True)
ax[3].set_xlabel("$\sigma^2_x$")

ax[4].hist(mcmc_samples['sigmay_2'], bins=100,density=True)

ax[0].set_xlabel("$a_0$")
ax[4].set_xlabel("$\sigma^2_y$")

#ax[4].set_xlabel("MCMC iterations")


for i in range(5):
  #for j in range(2):
    ##ax[i,j].set_xlabel('x')
    #ax[i,j].set_ylabel('y')
    for item in ([ ax[i].xaxis.label, ax[i].yaxis.label]):
        item.set_fontsize(15)

plt.show()
if save_me:
  mypath='trace_plots_all_hist.png'
  plt.savefig(filename+mypath)



#@title
args["hidden_dim_temporal"]= 35
args["z_dim_temporal"]= 11
args["T"]=T
with open('decoders/decoder_1d_T50_fixed_ls', 'rb') as file:
    decoder_params = pickle.load(file)
    print(len(decoder_params))

args["decoder_params_temporal"] = decoder_params


# spatial VAE training
args["hidden_dim1_spatial"]= 35
args["hidden_dim2_spatial"]= 30
args["z_dim_spatial"]=10
n_xy=25


n=n_xy
if args['background']=='LGCP':
  #Load 2d spatial trained decoder
  with open('./decoders/decoder_2d_n25_infer_hyperpars'.format(n), 'rb') as file:
      decoder_params = pickle.load(file)
      print(len(decoder_params))

  args["decoder_params_spatial"] = decoder_params



def spatiotemporal_hawkes_model(args):
    t_events=args["t_events"]
    xy_events=args["xy_events"]
    N=t_events.shape[0]

    #########LGCP BACsKGROUND
    # temporal rate
    # mean
    if args['background']=='constant':     
      a_0 = numpyro.sample("a_0", dist.Normal(.5,1))# this was 2,2
      b_0=0 #b_0 = numpyro.sample("b_0", dist.Normal(1.5,1))
      mu_xyt=numpyro.deterministic("mu_xyt",jnp.exp(a_0+b_0))
      Itot_txy_back=numpyro.deterministic("Itot_txy_back",mu_xyt*args['T'] )

    if args['background']=='LGCP':
      a_0 = numpyro.sample("a_0", dist.Normal(0,2))# this was 2,2
      #zero mean temporal gp ft 
      z_temporal = numpyro.sample("z_temporal", dist.Normal(jnp.zeros(args["z_dim_temporal"]), jnp.ones(args["z_dim_temporal"])))
      decoder_nn_temporal = vae_decoder_temporal(args["hidden_dim_temporal"], args["n_t"])  
      decoder_params = args["decoder_params_temporal"]
      v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal))
      f_t = numpyro.deterministic("f_t", v_t[0:n_t])
      rate_t = numpyro.deterministic("rate_t",jnp.exp(f_t+a_0))
      Itot_t=numpyro.deterministic("Itot_t", jnp.sum(rate_t)/args["n_t"]*args["T"])
      #Itot = numpyro.deterministic("Itot", v[len(x)])#Itot_t=jnp.trapz(mu_0*jnp.exp(f), back_t)
      f_t_events=f_t[args["indices_t"]]

      # zero mean spatial gp
      b_0=0
      #numpyro.sample("b_0", dist.Normal(1,3))# this was 2,2
      z_spatial = numpyro.sample("z_spatial", dist.Normal(jnp.zeros(args["z_dim_spatial"]), jnp.ones(args["z_dim_spatial"])))
      decoder_nn = vae_decoder_spatial(args["hidden_dim2_spatial"], args["hidden_dim1_spatial"], args["n_xy"])  
      decoder_params = args["decoder_params_spatial"]
      f_xy = numpyro.deterministic("f_xy", decoder_nn[1](decoder_params, z_spatial))
      rate_xy = numpyro.deterministic("rate_xy",jnp.exp(f_xy+b_0))
      Itot_xy=numpyro.deterministic("Itot_xy", jnp.sum(rate_xy)/args["n_xy"]**2)
      f_xy_events=f_xy[args["indices_xy"]]
      Itot_txy_back=numpyro.deterministic("Itot_txy_back",Itot_t*Itot_xy)#jnp.sum(mu_xyt*args['T']/args['n_t']/args['n']**2))


    #### EXPONENTIAL KERNEL for the excitation part
    #temporal exponential kernel parameters
    alpha = numpyro.sample("alpha", dist.HalfNormal(0.8))#numpyro.sample("alpha", dist.Gamma(.5,1))##numpyro.sample("alpha", dist.HalfNormal(0.5,2))# has to be within 0,1
    beta = numpyro.sample("beta", dist.HalfNormal(0.3))#numpyro.sample("beta", dist.Gamma(.7,1))
    
    #spatial gaussian kernel parameters     
    sigmax_2 = numpyro.sample("sigmax_2", dist.Exponential(.1))
    sigmay_2 = numpyro.sample("sigmay_2", dist.Gamma(0.5,1))#Exponential(.3))
    
    T,x_min,x_max,y_min,y_max = args['T'],args['x_min'],args['x_max'],args['y_min'],args['y_max']  
    
    T_diff=difference_matrix(t_events);
    S_mat_x = difference_matrix(xy_events[0])
    S_mat_y = difference_matrix(xy_events[1])
    S_diff_sq=(S_mat_x**2)/sigmax_2+(S_mat_y**2)/sigmay_2; 
    l_hawkes_sum=alpha*beta/(2*jnp.pi*jnp.sqrt(sigmax_2*sigmay_2))*jnp.exp(-beta*T_diff-0.5*S_diff_sq)
    l_hawkes = numpyro.deterministic('l_hawkes',jnp.sum(jnp.tril(l_hawkes_sum,-1),1))

    if args['background']=='constant':
      ell_1=numpyro.deterministic('ell_1',jnp.sum(jnp.log(l_hawkes+jnp.exp(a_0+b_0))))# the extra a_0 is for the firs term
    elif args['background']=='LGCP':
      ell_1=numpyro.deterministic('ell_1',jnp.sum(jnp.log(l_hawkes+jnp.exp(a_0+b_0+f_t_events+f_xy_events))))# the extra a_0 is for the firs term

    #### hawkes integral
    exponpart = alpha*(1-jnp.exp(-beta*(T-t_events)))
    numpyro.deterministic("exponpart",exponpart)
    
    s1max=(x_max-xy_events[0])/(jnp.sqrt(2*sigmax_2))
    s1min=(xy_events[0])/(jnp.sqrt(2*sigmax_2))
    gaussianpart1=0.5*jax.scipy.special.erf(s1max)+0.5*jax.scipy.special.erf(s1min)
    
    s2max=(y_max-xy_events[1])/(jnp.sqrt(2*sigmay_2))
    s2min=(xy_events[1])/(jnp.sqrt(2*sigmay_2))
    gaussianpart2=0.5*jax.scipy.special.erf(s2max)+0.5*jax.scipy.special.erf(s2min)
    gaussianpart=gaussianpart2*gaussianpart1
    numpyro.deterministic("gaussianpart",gaussianpart)    

    ## total integral
    Itot_txy=jnp.sum(exponpart*gaussianpart)+Itot_txy_back
    #Itot_txy=Itot_txy_true
    numpyro.deterministic("Itot_txy",Itot_txy)
    loglik=numpyro.deterministic('loglik',ell_1-Itot_txy)

    numpyro.factor("t_events", loglik) 
    numpyro.factor("xy_events", loglik) 



rng_key_pred = random.PRNGKey(2)
predictive = Predictive(spatiotemporal_hawkes_model, mcmc_samples)
predictions = predictive(rng_key_pred, args=args)



if args['background']=='LGCP':
  fig,ax=plt.subplots(1,1,figsize=(8,5))
  ax.plot(t_events_total, np.ones(len(indices_t))*np.min(f_t_hpdi[0]), 'b+',color="red", label="observed times")
  ax.set_ylabel('$f_t$')
  ax.set_xlabel('t')
  ax.plot(x_t, f_t_pred_mean, color="green", label="mean estimated $f_t$")
  ax.fill_between(x_t, f_t_hpdi[0], f_t_hpdi[1], alpha=0.4, color="palegoldenrod", label="90%CI rate")
  ax.legend()


  for item in ([ ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)

  mypath='f_t.pdf'
  plt.savefig(filename+mypath)

  mypath='f_t.png'
  plt.savefig(filename+mypath)


rate_xy_pred=predictions['rate_xy']
rate_xy_pred_mean=jnp.mean(rate_xy_pred, axis=0)
rate_xy_hpdi = hpdi(rate_xy_pred, 0.9)

f_xy_pred=predictions["f_xy"]
f_xy_pred_mean=jnp.mean(f_xy_pred, axis=0)
f_xy_hpdi = hpdi(f_xy_pred, 0.9)





rate_xy_post=mcmc_samples['rate_xy']
rate_xy_post_mean=jnp.mean(rate_xy_post, axis=0)
rate_xy_post_di = hpdi(rate_xy_post, 0.9)

f_xy_post=mcmc_samples["f_xy"]
f_xy_post_mean=jnp.mean(f_xy_post, axis=0)
f_xy_post_di = hpdi(f_xy_post, 0.9)

fig, ax = plt.subplots(1,2, figsize=(10, 5))
#_min, _max = np.amin(f_xy_true), np.amax(f_xy_true)
#im = ax[0].imshow(f_xy_true.reshape(n,n), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
#ax[0].title.set_text('Simulated f_xy')
#fig.colorbar(im, ax=ax[0])
#fig.show()
_min, _max = np.amin(f_xy_post), np.amax(f_xy_post)
im = ax[0].imshow(f_xy_post_mean.reshape(n,n), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
ax[0].title.set_text('Estimated $f_s$')
#fig.colorbar(im, ax=ax)

im2 = ax[1].imshow(f_xy_post_mean.reshape(n,n), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
ax[1].plot(xy_events_total[:,0],xy_events_total[:,1],'x', alpha=.25,color='red',label='true event locations')
#ax.plot(x_xy[indices_xy][:,0],x_xy[indices_xy][:,1],'x', label='true event locations')
ax[1].title.set_text('Estimated $f_s$ with true locations')


for i in range(2):
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('y')
    for item in ([ ax[i].xaxis.label, ax[i].yaxis.label, ax[i].title] ):
        item.set_fontsize(15)

fig.subplots_adjust(right=0.8)
#cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cax = fig.add_axes([ax[1].get_position().x1+0.03,ax[1].get_position().y0,0.02,ax[1].get_position().height])
fig.colorbar(im, cax=cax)# fraction=0.01, pad=0.01, shrink=.06)


mypath='f_s_2.png'
plt.savefig(filename+mypath)

