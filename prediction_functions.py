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
from jax.example_libraries import stax

from functools import partial
from utils import *

# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS, init_to_median, Predictive, RenyiELBO
from numpyro.diagnostics import hpdi
from random import randint

from inference_functions import vae_encoder_temporal,vae_decoder_temporal,vae_model_temporal,vae_guide_temporal,vae_encoder_spatial,vae_decoder_spatial,vae_model_spatial



#past_times, past_locs, N_new, x_min, x_max, y_min, y_max, 
#lambda_0: the background
#alpha, beta, sigma_sq: hawkes kernel params

def simulate_spatiotemporal_hawkes_predictions(past_times, past_locs, N_new, x_min, x_max, y_min, y_max, lambda_0, alpha, beta, sigma_sq, integ_xy_back,background, f_t=None):
    #f_t and f_s will be evaluated on a grid
    #### simulates N hawkes event pairs (s,t)
    a = 0; b = 0; c = 0;
    back_points = 0
    gauss_points = 0


    N_past=len(past_times);
    event_times = np.zeros(N_past+N_new); event_spatial_X=np.zeros(N_past+N_new); event_spatial_Y=np.zeros(N_past+N_new);
     
    event_spatial_back=np.zeros(N_past+N_new+1);event_spatial_gauss=event_spatial_back;
    event_times[0:N_past]=past_times; event_spatial_X[0:N_past]=past_locs[0,:]; event_spatial_Y[0:N_past]=past_locs[1,:];
    
    #event_times[0]=0; #event_spatial_gauss[0]=0; event_spatial_back[0]=0
    back_points= N_past-1# because we have zero as the first temporal and spatial point
    T_max = np.max(past_times)
    T_train=50
    T_test=T_train+30
    x_t=np.arange(0,T_test,1)

    i=N_past
    
    t_last = past_times[-1]
    x_last = past_locs[0,-1];y_last = past_locs[1,-1]

    # generate first event (s,t) from the background
    # firstly the time coordinate which coincides with the interarrival time 
    
    if background=='constant':
      integ_xy_back=(x_max-x_min)*(y_max-y_min)
      nu_0 = lambda_0
      
    elif background=='LGCP':
      #integ_xy_back=args_test['Itot_xy'][0]### take it from the output
      nu_0 = lambda_0*np.exp(f_t[T_train-1])*integ_xy_back
      f_t_test=f_t[T_train:]

      
    gaussianpart1=0.5*jax.scipy.special.erf(x_max)+0.5*jax.scipy.special.erf(x_min)
    gaussianpart2=jax.scipy.special.erf(y_max)/2+jax.scipy.special.erf(y_min)/2
    integ_xy_gauss=gaussianpart2*gaussianpart1 
    
    
    #i=0  
    if t_last>0:# i.e. prediction given some historic events
      S_a =t_last; Lambda_c = nu_0+np.sum(alpha*beta*np.exp(-beta*(t_last-past_times))[past_times<t_last]);
      #print('Lambda_c', Lambda_c.shape) ##### THIS IS WRONG DIMENSION, IT IS 50 where it should be 1

      #u_a = -np.log(np.random.uniform())/Lambda_c
      ##next new point
      #T_i = u_a; J=0;
      ## secondly the spatial event (from the base intensity) homogeneous poisson intensity lambda_0
      #X_i = np.random.uniform(x_min,x_max)
      #Y_i = np.random.uniform(y_min,y_max)
      #i+=1 
      #event_times[i] = T_i
      #event_spatial_X[i] = X_i
      #event_spatial_Y[i] = Y_i
    else:  
      S_a =0; Lambda_c = nu_0;
      u_a = -np.log(np.random.uniform())/Lambda_c

      T_i = u_a; J=0;
      # secondly the spatial event (from the base intensity) homogeneous poisson intensity lambda_0
      X_i = np.random.exponential(lambda_0)
      Y_i = np.random.exponential(lambda_0)

      i+=1 
      event_times[i] = T_i
      event_spatial_X[i] = X_i
      event_spatial_Y[i] = Y_i
    #Simulate all the subsequent (s,t) events using thinning
    #print('total train', N_past)
    #print('total test', N_new)

    for i in range(N_past, N_past+N_new):
    
        while True:
            #propose an interarrival time 
            b+=1; u_a = -np.log(np.random.uniform())/Lambda_c
            S_a+=u_a

            #thinning
            # accept if new porposed intensity is below lambda max which is the inten at prev time point
            # otherwise try again
            b+=1;
            #k = cov_function(S_a, S_a, var, lengthscale, noise)
            #pass the new GP to calculate intensity
            ##find shich index you want
            #f(t_k)
            #set nu_0=np.exp(a_0+f(t_k))*integ_xy_back
            if background=='LGCP':
              ind=find_index(S_a, x_t[0])
              nu_0=lambda_0*np.exp(f_t_test[ind])*integ_xy_back            
            
            l=lambda_S(nu_0, integ_xy_back, alpha, beta, S_a, event_times)
            if np.random.uniform()>l/Lambda_c:
                c+=1;
                Lambda_c = l
            else:
                break # ie success and accept the point
        
        T_i = S_a; b+=1

        #determine if temporal point came from background or triggering kernel       
        J_star=np.argmin(np.random.uniform()< np.cumsum(integ_xy_gauss*alpha*beta*np.exp(-beta*(T_i-event_times))[event_times<T_i])/lambda_S(nu_0,integ_xy_back,alpha,beta,T_i,event_times))
        
        #now given a point coordinate, simulate the space coordinate
        if J_star==0:
          #from the base
            back_points+=1
            X_i = np.random.uniform(x_min,x_max)
            Y_i = np.random.uniform(y_min,y_max)
        else:
          #from the gaussian kernel
            gauss_points+=1
            X_i = np.random.normal(X_i,sigma_sq) 
            Y_i = np.random.normal(Y_i,sigma_sq) 
                 

        event_times[i]=T_i
        event_spatial_X[i]=X_i
        event_spatial_Y[i]=Y_i
        i+=1
        #need J_star
        #r=np.sqrt(-2*np.log( np.random.uniform()))/np.sqrt(sigma_sq)
        #s_i=event_spatial[J_star]+np.random.uniform()*r
        #event_spatial[i]=s_i
      
    return event_times[N_past:], event_spatial_X[N_past:],event_spatial_Y[N_past:], event_times, event_spatial_X, event_spatial_Y




#@title
#past_times, past_locs, N_new, x_min, x_max, y_min, y_max, 
#lambda_0: the background
#alpha, beta, sigma_sq: hawkes kernel params

def simulate_spatiotemporal_hawkes_const_back_predictions(past_times, past_locs, N_new, x_min, x_max, y_min, y_max, lambda_0, alpha, beta, sigma_sq):
    #### simulates N hawkes event pairs (s,t)
    a = 0; b = 0; c = 0;
    back_points = 0
    gauss_points = 0
    
    N_past=len(past_times); 
    event_times = np.zeros(N_past+N_new+1); event_spatial_X=np.zeros(N_past+N_new+1); event_spatial_Y=np.zeros(N_past+N_new+1);
     
    event_spatial_back=np.zeros(N_past+N_new+1);event_spatial_gauss=event_spatial_back;
    event_times[0:N_past]=past_times; event_spatial_X[0:N_past]=past_locs[0,:]; event_spatial_Y[0:N_past]=past_locs[1,:];
    
    #event_times[0]=0; #event_spatial_gauss[0]=0; event_spatial_back[0]=0
    back_points= N_past-1# because we have zero as the first temporal and spatial point
    T_max = np.max(past_times)
    i=N_past
    
    t_last = past_times[-1]
    x_last = past_locs[0,-1];y_last = past_locs[1,-1]

    # generate first event (s,t) from the background
    # firstly the time coordinate which coincides with the interarrival time 
    integ_xy_back=(x_max-x_min)*(y_max-y_min)
    nu_0 = lambda_0

    #i=0  
    if t_last>0:
      S_a =t_last; Lambda_c = nu_0+np.sum(alpha*np.exp(-beta*(t_last-past_times))[past_times<t_last]);
      #u_a = -np.log(np.random.uniform())/Lambda_c
      ##next new point
      #T_i = u_a; J=0;
      ## secondly the spatial event (from the base intensity) homogeneous poisson intensity lambda_0
      #X_i = np.random.uniform(x_min,x_max)
      #Y_i = np.random.uniform(y_min,y_max)

      #i+=1 
      #event_times[i] = T_i
      #event_spatial_X[i] = X_i
      #event_spatial_Y[i] = Y_i

    else:  
      S_a =0; Lambda_c = nu_0;
      u_a = -np.log(np.random.uniform())/Lambda_c

      T_i = u_a; J=0;
      # secondly the spatial event (from the base intensity) homogeneous poisson intensity lambda_0
      X_i = np.random.exponential(lambda_0)
      Y_i = np.random.exponential(lambda_0)

      i+=1 
      event_times[i] = T_i
      event_spatial_X[i] = X_i
      event_spatial_Y[i] = Y_i
    #Simulate all the subsequent (s,t) events using thinning
    for i in range(N_past, N_past+N_new):
        
        while True:
            #first the time coordinate
            #propose an interarrival time 
            b+=1; u_a = -np.log(np.random.uniform())/Lambda_c
            S_a+=u_a

            #thinning
            # accept if new porposed intensity is below lambda max which is the inten at prev time point
            # otherwise try again
            b+=1;
            #k = cov_function(S_a, S_a, var, lengthscale, noise)

            
            l=lambda_S(nu_0,integ_xy_back,alpha,beta, S_a,event_times)
            if np.random.uniform()>l/Lambda_c:
                c+=1;
                Lambda_c = l
            else:
                break # ie success and accept the point
        
        T_i = S_a; b+=1
        #determine if temporal point came from background or not
        J_star=np.argmin(np.random.uniform()< integ_xy_back*np.cumsum(alpha*np.exp(-beta*(T_i-event_times))[event_times<T_i])/lambda_S(nu_0,integ_xy_back,alpha,beta,T_i,event_times))
        
        #now given a point coordinate, simulate the space coordinate
        if J_star==0:
          #from the base
            back_points+=1
            X_i = np.random.uniform(x_min,x_max)
            Y_i = np.random.uniform(y_min,y_max)
        else:
          #from the gaussian kernel
            gauss_points+=1
            X_i = np.random.normal(X_i,sigma_sq) 
            Y_i = np.random.normal(Y_i,sigma_sq) 
                 
        i+=1
        event_times[i]=T_i
        event_spatial_X[i]=X_i
        event_spatial_Y[i]=Y_i
        #need J_star
        #r=np.sqrt(-2*np.log( np.random.uniform()))/np.sqrt(sigma_sq)
        #s_i=event_spatial[J_star]+np.random.uniform()*r
        #event_spatial[i]=s_i
      
    return event_times[N_past+1:], event_spatial_X[N_past+1:],event_spatial_Y[N_past+1:]


def lambda_S(nu_0, const_S, alpha,beta,t,event_times):
    #nu_0=exp(a_0)
    #nu_0=exp(a_0+f(t))integ_back_xy
    return nu_0 + const_S*np.sum(alpha*beta*np.exp(-beta*(t-event_times))[event_times<t])

 

def sample_GP(args,typ='posterior'):
  a_0=args['a_0_post_mean']
  z_temporal=args['z_temporal']
  z_spatial=args['z_spatial']
  n_total=args['n_total']

  if typ=='prior':
    z_temporal= numpyro.sample("z_temporal",dist.Normal(jnp.zeros(args["z_dim_temporal"])), jnp.ones(args["z_dim_temporal"]))
  else:
    z_temporal=numpyro.deterministic("z_temporal",np.mean(z_temporal[100:],0))

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
    z_spatial=numpyro.deterministic("z_spatial",np.mean(z_spatial[n_total-500:n_total],0))

  decoder_nn = vae_decoder_spatial(args["hidden_dim2_spatial"], args["hidden_dim1_spatial"], args["n_xy"])  
  decoder_params = args["decoder_params_spatial"]
  f_xy = numpyro.deterministic("f_xy", decoder_nn[1](decoder_params, z_spatial))
  rate_xy = numpyro.deterministic("rate_xy",jnp.exp(f_xy+b_0))
  Itot_xy=numpyro.deterministic("Itot_xy", jnp.sum(rate_xy)/args["n_xy"]**2)
  #f_xy_events=f_xy[args["indices_xy"]]
  Itot_txy_back=numpyro.deterministic("Itot_txy_back",Itot_t*Itot_xy)#jnp.sum(mu_xyt*args['T']/args['n_t']/args['n']**2))


#@title
## simulation function
def simulate_LGCP_predictions(x_t, x_xy, T, gp_kernel=exp_sq_kernel, jitter=1e-5, a_0=None, b_0=None, var_t=None, length_t=None, var_xy=None, length_xy=None, t=None, xy=None, typ='posterior'):
      
    n_t=x_t.shape[0]
    n_sq=x_xy.shape[0]
    
    if a_0==None:     
        a_0 = numpyro.sample("a_0", dist.Gamma(1, 1))#=1# or add a prior
    else:
      a_0=numpyro.deterministic("a_0",a_0)

    if length_t==None:
        length_t = numpyro.sample("kernel_length_t", dist.InverseGamma(4,1))
    if var_t==None:
        var_t = numpyro.sample("kernel_var_t", dist.LogNormal(0.,0.1))
        
    k_t = gp_kernel(x_t, x_t, var_t, length_t, jitter)

    if typ=='prior':
      f_t = numpyro.sample("f_t", dist.MultivariateNormal(loc=jnp.zeros(x_t.shape[0]), covariance_matrix=k_t))
    else:
      z_temporal=numpyro.deterministic("z_temporal",np.mean(mcmc_samples['z_temporal'][100:],0))
      decoder_nn_temporal = vae_decoder_temporal(args_train["hidden_dim_temporal"], args_train["n_t"])  
      decoder_params = args_train["decoder_params_temporal"]
      v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal))
      f_t = numpyro.deterministic("f_t", v_t[0:args_train["n_t"]])
          
    rate_t = numpyro.deterministic("rate_t", jnp.exp(a_0 + f_t))
    Itot_t = numpyro.deterministic("Itot_t",jnp.sum(rate_t)*T/n_t)

    if length_xy==None:
        length_xy = numpyro.sample("kernel_length_xy", dist.InverseGamma(4,1))#mean1/3
    if var_xy==None:
        var_xy = numpyro.sample("kernel_var_xy", dist.LogNormal(0.,0.1))#mean1
        

    if typ=='prior':
      k_xy = gp_kernel(x_xy, x_xy, var_xy, length_xy, jitter)
      f_xy = numpyro.sample("f_xy", dist.MultivariateNormal(loc=jnp.zeros(x_xy.shape[0]), covariance_matrix=k_xy))
    else:
      z_spatial=numpyro.deterministic("z_spatial",np.mean(mcmc_samples['z_spatial'][100:],0))
      decoder_nn = vae_decoder_spatial(args_train["hidden_dim2_spatial"], args_train["hidden_dim1_spatial"], args_train["n_xy"])  
      decoder_params = args_train["decoder_params_spatial"]
      f_xy = numpyro.deterministic("f_xy", decoder_nn[1](decoder_params, z_spatial))
      
    if b_0==None:     
        b_0 = numpyro.sample("b_0", dist.Gamma(1, 1))#=1# or add a prior
    else:
      b_0=numpyro.deterministic("b_0",b_0)

    rate_xy = numpyro.deterministic("rate_xy", jnp.exp(f_xy+b_0))
    Itot_xy = numpyro.deterministic("Itot_xy",jnp.sum(rate_xy)*1/n_sq)

    N_0=numpyro.sample("N_0",dist.Poisson(rate=Itot_xy*Itot_t)) #or 

    #_, _, t, ft_star = randdist(x_t, rate_t/jnp.sum(rate_t), N_0) # draw events treating the rate as a density
    ind_t_i, t_i, rate_t_i=rej_sampling_new(N_0, x_t, rate_t, args['n_t'])
    numpyro.deterministic("rate_t_i", rate_t_i) 
    numpyro.deterministic("t_i", t_i) 
    numpyro.deterministic("ind_t_i", ind_t_i) 
    N_0 = t_i.shape[0]
    
    ind_xy_i, xy_i, rate_xy_i=rej_sampling_new(N_0, x_xy, rate_xy, args['n_xy']**2)
    numpyro.deterministic("rate_xy_i", rate_xy_i) 
    numpyro.deterministic("xy_i", xy_i) 
    numpyro.deterministic("ind_xy_i", ind_xy_i) 


