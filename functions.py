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
from numpyro.diagnostics import hpdi
from random import randint

import sys

from utils import exp_sq_kernel, find_index

#

def generate_background_uniform_events(T, a_0=None, b_0=0):
  if a_0==None:     
    a_0 = numpyro.sample("a_0", dist.Gamma(2, 2))#=1# or add a prior
  else:
    a_0=numpyro.deterministic("a_0",a_0)

  if b_0==None:     
    b_0 = numpyro.sample("b_0", dist.Gamma(2, 2))#=1# or add a prior
  else:
    b_0=numpyro.deterministic("b_0",b_0)

  N=numpyro.sample("N",dist.Poisson(jnp.exp(a_0)*T))

  t_events=np.random.uniform(0,T,N)
  t_events=numpyro.deterministic("t_events",t_events)
  xy_events=np.random.uniform(0,1,2*N).reshape(N,2)
  xy_events=numpyro.deterministic("xy_events",xy_events)



  #@title


## simulation function
def spatiotemporal_LGCP(args, x_t, x_xy, T, gp_kernel=exp_sq_kernel, jitter=1e-5, a_0=None, b_0=None, var_t=None, length_t=None, var_xy=None, length_xy=None, t=None, xy=None):
      
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

    f_t = numpyro.sample("f_t", dist.MultivariateNormal(loc=jnp.zeros(x_t.shape[0]), covariance_matrix=k_t))
    
    rate_t = numpyro.deterministic("rate_t", jnp.exp(a_0 + f_t))
    Itot_t = numpyro.deterministic("Itot_t",jnp.sum(rate_t)*T/n_t)

    if length_xy==None:
        length_xy = numpyro.sample("kernel_length_xy", dist.InverseGamma(4,1))#mean1/3
    if var_xy==None:
        var_xy = numpyro.sample("kernel_var_xy", dist.LogNormal(0.,0.1))#mean1
        
    k_xy = gp_kernel(x_xy, x_xy, var_xy, length_xy, jitter)

    f_xy = numpyro.sample("f_xy", dist.MultivariateNormal(loc=jnp.zeros(x_xy.shape[0]), covariance_matrix=k_xy))
    
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


  
def GP(args, jitter=1e-4, y=None, var=None, length=None, sigma=None, noise=False):
    
    x = args["x"]
    #obs_idx = args["obs_idx"]
    gp_kernel=args["gp_kernel"] 

    if length==None:
        length = numpyro.sample("kernel_length", dist.InverseGamma(4,1))
        
    if var==None:
        var = numpyro.sample("kernel_var", dist.LogNormal(0.,0.1))
    
    k = gp_kernel(x, x, var, length, jitter)
    
    if noise==False:
        numpyro.sample("y",  dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k), obs=y)
    else:
        sigma = numpyro.sample("noise", dist.HalfNormal(0.05))
        f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(x.shape[0]), covariance_matrix=k))
        numpyro.sample("y", dist.Normal(f, sigma), obs=y)

def spatiotemporal_GP(T, x_t, x_xy, gp_kernel=exp_sq_kernel, jitter=1e-5, a_0=None, b_0=None, var_t=None, length_t=None, var_xy=None, length_xy=None):
      
    n_t=x_t.shape[0]
    n_sq=x_xy.shape[0]
    n_xy=25
    
    if a_0==None:     
        a_0 = numpyro.sample("a_0", dist.Gamma(1, 1))#=1# or add a prior
    else:
      a_0=numpyro.deterministic("a_0",a_0)

    if length_t==None:
        length_t = numpyro.sample("kernel_length_t", dist.InverseGamma(4,1))
    if var_t==None:
        var_t = numpyro.sample("kernel_var_t", dist.LogNormal(0.,0.1))
        
    k_t = gp_kernel(x_t, x_t, var_t, length_t, jitter)

    f_t = numpyro.sample("f_t", dist.MultivariateNormal(loc=jnp.zeros(x_t.shape[0]), covariance_matrix=k_t))
    
    rate_t = numpyro.deterministic("rate_t", jnp.exp(a_0 + f_t))
    Itot_t = numpyro.deterministic("Itot_t",jnp.sum(rate_t)*T/n_t)

    if length_xy==None:
        length_xy = numpyro.sample("kernel_length_xy", dist.InverseGamma(4,1))#mean1/3
    if var_xy==None:
        var_xy = numpyro.sample("kernel_var_xy", dist.LogNormal(0.,0.1))#mean1
        
    k_xy = gp_kernel(x_xy, x_xy, var_xy, length_xy, jitter)

    f_xy = numpyro.sample("f_xy", dist.MultivariateNormal(loc=jnp.zeros(x_xy.shape[0]), covariance_matrix=k_xy))
    rate_xy = numpyro.deterministic("rate_xy",jnp.exp(f_xy+b_0))
    Itot_xy=numpyro.deterministic("Itot_xy", jnp.sum(rate_xy)/n_xy**2)



#@title

def generate_spatiotemporal_offspring(args, t_star, s_star, alpha, beta, method):
    if not(t_star.size==s_star.shape[0]):
        print(t_star.size, 'background temporal events')
        print(s_star.shape[0], 'background spatial events')
        raise ValueError
    N_0=t_star.size
    print(N_0,' background events')  
    l=0  

    G_0_t=t_star;numpyro.deterministic("G_0_t",G_0_t)
    G_0_x=s_star[:,0];numpyro.deterministic("G_0_x",G_0_x)
    G_0_y=s_star[:,1];numpyro.deterministic("G_0_y",G_0_y) 
    #print('generation 0 times',G_0)
    #G_l_t=G_0_t; G_l_x=G_0_x; G_l_y=G_0_y
    G_l_t=np.array([]); G_l_x=np.array([]); G_l_y=np.array([]); #G_l_t_test=np.array([]);G_l_x_test=np.array([]);G_l_y_test=np.array([])
    G_l_t_prev=G_0_t;
    G_l_x_prev=G_0_x;
    G_l_y_prev=G_0_y;
    N_l=G_l_t_prev.size;N_l_gen=np.array([N_l]);
    G_all_t=G_l_t;G_all_x=G_l_x;G_all_y=G_l_y
    #G_all_t_test=G_all_t; G_all_x_test=G_all_x; G_all_y_test=G_all_y
    
    while N_l>0:
        #print('generation',l)
        new_gen_offspring=np.random.poisson(alpha,N_l)
        #print('new offsrping',new_gen_offspring)
        #G_l_t=np.array([]);G_l_x=np.array([]);G_l_y=np.array([])
        inter_times=np.zeros((N_l,100))*np.nan
        x_dist=np.zeros((N_l,100))*np.nan
        y_dist=np.zeros((N_l,100))*np.nan    

        for i in range(N_l):
            C_i=new_gen_offspring[i]#number of new offspring
            inter_times[i,:C_i]=np.random.exponential(beta,C_i)
            G_l_t_aux=inter_times[i,0:C_i]+G_l_t_prev[i] # new temporal offsrping
                        
            if method=='use_truncated':
              indx_boundary=G_l_t_aux<=T
              G_l_t_aux=G_l_t_aux[indx_boundary]
              C_i=G_l_t_aux.size  
              lower_bound=(args['x_min'] - G_l_x_prev[i]) / np.sqrt(args['sigmax_2']);
              upper_bound=(args['x_max'] - G_l_x_prev[i]) / np.sqrt(args['sigmax_2']);
              X=stats.truncnorm(lower_bound, upper_bound, loc=G_l_x_prev[i], scale=np.sqrt(args['sigmax_2']))
              x_dist[i,:C_i]=X.rvs(C_i)
              G_l_x_aux=x_dist[i,:C_i]
              lower_bound=(args['y_min'] - G_l_y_prev[i]) / np.sqrt(args['sigmay_2']);
              upper_bound=(args['y_max'] - G_l_y_prev[i]) / np.sqrt(args['sigmay_2']);
              Y=stats.truncnorm(lower_bound, upper_bound, loc=G_l_y_prev[i], scale=np.sqrt(args['sigmay_2']))
              y_dist[i,:C_i]=Y.rvs(C_i)
              G_l_y_aux=y_dist[i,:C_i]


            elif method=='remove_xy_outside_boundary':
                x_dist[i,:C_i]=np.random.normal(0, args['sigmax_2'], C_i)
                G_l_x_aux=x_dist[i,:C_i]+G_l_x_prev[i] 
                y_dist[i,:C_i]=np.random.normal(0, args['sigmay_2'], C_i)            
                G_l_y_aux=y_dist[i,:C_i]+G_l_y_prev[i]          
                indx_boundary=(G_l_t_aux<=args['T'])&(G_l_x_aux>=args['x_min'])&(G_l_x_aux<=args['x_max'])&(G_l_y_aux>=args['y_min'])&(G_l_y_aux<=args['y_max'])
                G_l_t_aux=G_l_t_aux[indx_boundary]
                G_l_x_aux= G_l_x_aux[indx_boundary]
                G_l_y_aux=G_l_y_aux[indx_boundary]

                #indx_boundary_test=(G_l_t_aux<=T_test)&(G_l_x_aux>=args['x_min'])&(G_l_x_aux<=args['x_max'])&(G_l_y_aux>=args['y_min'])&(G_l_y_aux<=args['y_max'])
                #G_l_t_aux_test=G_l_t_aux[indx_boundary]
                #G_l_x_aux_test= G_l_x_aux[indx_boundary]
                #G_l_y_aux_test=G_l_y_aux[indx_boundary]


 
            elif method=='assign_to_boundary':
                ## times
                indx_boundary=G_l_t_aux<=T
                G_l_t_aux=G_l_t_aux[indx_boundary]
                C_i=G_l_t_aux.size                     
                ## x coordinates 
                x_dist[i,:C_i]=np.random.normal(0, args['sigmax_2'], C_i)
                G_l_x_aux=x_dist[i,:C_i]+G_l_x_prev[i]
                G_l_x_aux[(G_l_x_aux<args['x_min'])]=args['x_min']
                G_l_x_aux[(G_l_x_aux>args['x_max'])]=args['x_max']               
                ## y coordinates
                y_dist[i,:C_i]=np.random.normal(0, args['sigmay_2'], C_i)            
                G_l_y_aux=y_dist[i,:C_i]+G_l_y_prev[i]              
                G_l_y_aux[(G_l_y_aux<args['y_min'])]=args['y_min']
                G_l_y_aux[(G_l_y_aux>args['y_max'])]=args['y_max']
            
            elif method=='repeat_sampling':
                indx_boundary=G_l_t_aux<=T
                G_l_t_aux=G_l_t_aux[indx_boundary]
                C_i=G_l_t_aux.size 
                
                G_l_x_aux=np.random.normal(0, args['sigmax_2'], C_i+100)+G_l_x_prev[i] 
                ind_x=(G_l_x_aux>=args['x_min'])&(G_l_x_aux<=args['x_max'])
                x_dist[i,:C_i]=G_l_x_aux[ind_x][0:C_i]
                #G_l_x_aux=x_dist[i,:C_i]+G_l_x_prev[i] 
                G_l_y_aux=np.random.normal(0, args['sigmay_2'], C_i+100) +G_l_y_prev[i] 
                ind_y=(G_l_y_aux>=args['y_min'])&(G_l_y_aux<=args['y_max'])
                
                y_dist[i,:C_i]=G_l_y_aux[ind_y][0:C_i]          
                #G_l_y_aux=y_dist[i,:C_i]+G_l_y_prev[i]          
                #indx_boundary=(G_l_t_aux<=T)&(G_l_x_aux>=args['x_min'])&(G_l_x_aux<=args['x_max'])&(G_l_y_aux>=args['y_min'])&(G_l_y_aux<=args['y_max'])
                G_l_x_aux= x_dist[i,:C_i]#G_l_x_aux[indx_boundary]
                G_l_y_aux=y_dist[i,:C_i]#G_l_y_aux[indx_boundary]



            #G_l_t_test=np.concatenate((G_l_t_test,G_l_t_aux))
            #G_l_x_test=np.concatenate((G_l_x_test,G_l_x_aux))
            #G_l_y_test=np.concatenate((G_l_y_test,G_l_y_aux))

            G_l_t=np.concatenate((G_l_t,G_l_t_aux))
            G_l_x=np.concatenate((G_l_x,G_l_x_aux))
            G_l_y=np.concatenate((G_l_y,G_l_y_aux))

        #G_l_t_prev_test=G_l_t_test
        #G_l_x_prev_test=G_l_x_test;
        #G_l_y_prev_test=G_l_y_test

        G_l_t_prev=G_l_t
        G_l_x_prev=G_l_x;
        G_l_y_prev=G_l_y
        
        l+=1
        N_l=G_l_t.size
        N_l_gen=jnp.concatenate((N_l_gen,np.array([N_l])))
        G_all_t=jnp.concatenate((G_all_t,G_l_t))
        G_all_x=jnp.concatenate((G_all_x,G_l_x))        
        G_all_y=jnp.concatenate((G_all_y,G_l_y)) 
        
        #G_all_t_test=jnp.concatenate((G_all_t_test,G_l_t_test))
        #G_all_x_test=jnp.concatenate((G_all_x_test,G_l_x_test))        
        #G_all_y_test=jnp.concatenate((G_all_y_test,G_l_y_test)) 

        G_l_t=np.array([]);G_l_x=np.array([]);G_l_y=np.array([])
        #G_l_t_test=np.array([]);G_l_x_test=np.array([]);G_l_y_test=np.array([])
    
    print(G_all_t.size, 'offpsring eventS')
    #print(G_all_t_test.size, 'offpsring events up to test time')


    numpyro.deterministic("N_l_gen",N_l_gen)
    numpyro.deterministic("G_last_gen_t",G_l_t_prev)
    numpyro.deterministic("G_last_gen_x",G_l_x_prev)
    numpyro.deterministic("G_last_gen_y",G_l_y_prev)

    G_all_t=numpyro.deterministic("G_offspring_t",G_all_t)
    G_all_x=numpyro.deterministic("G_offspring_x",G_all_x)
    G_all_y=numpyro.deterministic("G_offspring_y",G_all_y)

    G_tot_t=np.concatenate((G_0_t,G_all_t));    
    print(G_tot_t.size, 'total events')
    G_tot_x=np.concatenate((G_0_x,G_all_x))
    G_tot_y=np.concatenate((G_0_y,G_all_y))
    G_tot_labels=np.concatenate((np.ones(G_0_t.size),np.zeros(G_all_t.size)))
        
    index_array=numpyro.deterministic("index_array", np.argsort(G_tot_t))
    G_tot_labels=G_tot_labels[index_array];
    G_tot_labels=numpyro.deterministic("G_tot_labels",G_tot_labels)

    G_tot_t=G_tot_t[index_array];
    G_tot_t=numpyro.deterministic("G_tot_t",G_tot_t)
    G_tot_x=G_tot_x[index_array];
    G_tot_x=numpyro.deterministic("G_tot_x",G_tot_x)
    G_tot_y=G_tot_y[index_array];
    G_tot_y=numpyro.deterministic("G_tot_y",G_tot_y)  
    #return [G_0_t,G_0_x,G_0_y], [G_all_t.sort(),G_all_x,G_all_y], [G_tot_t,G_tot_x,G_tot_y],G_tot_labels




#@title


#def spatiotemporal_homogenous_poisson(args):
#  t_min=args['t_min']
#  t_max=args['t_max']
#  a_0=args['a_0']
#  b_0=args['b_0']
#  t_events=args['t_events']
#  xy_events=args['xy_events']#

#  if a_0==None:     
#    a_0 = numpyro.sample("a_0", dist.Gamma(2, 2))#=1# or add a prior
#  else:
#    a_0=numpyro.deterministic("a_0",a_0)#

#  if b_0==None:     
#    b_0 = numpyro.sample("b_0", dist.Gamma(2, 2))#=1# or add a prior
#  else:
#    b_0=numpyro.deterministic("b_0",b_0)


#  if t_events==None:
#    N=numpyro.sample("N",dist.Poisson(jnp.exp(a_0+b_0)*(t_max-t_min)))
#    #t_events=np.random.uniform(t_min,t_max,N)
#    t_events=numpyro.deterministic("t_events",np.random.uniform(t_min,t_max,N))
#    xy_events=numpyro.deterministic("xy_events",np.random.uniform(0,1,2*N).reshape(N,2))
#  else:
#    N=t_events.shape[0]
#    loglik=a_0*N-jnp.exp(a_0)*(t_max-t_min)
#    numpyro.factor("t_events", loglik) 
#    numpyro.factor("xy_events", loglik)


def spatiotemporal_homogenous_poisson(args):
  t_min=args['t_min']
  t_max=args['t_max']
  a_0=args['a_0']
  b_0=args['b_0']
  t_events=args['t_events']
  xy_events=args['xy_events']
  if a_0==None:     
    a_0 = numpyro.sample("a_0", dist.Gamma(2, 2))#=1# or add a prior
  else:
    a_0=numpyro.deterministic("a_0",a_0)

  if b_0==None:     
    b_0 = numpyro.sample("b_0", dist.Gamma(2, 2))#=1# or add a prior
  else:
    b_0=numpyro.deterministic("b_0",b_0)

  #print('t_events',t_events)
  if t_events.any()==None:
    #print('here 1')
    N=numpyro.sample("N",dist.Poisson(jnp.exp(a_0+b_0)*(t_max-t_min)))
    #t_events=np.random.uniform(t_min,t_max,N)
    t_events=numpyro.deterministic("t_events",np.random.uniform(t_min,t_max,N))
    xy_events=numpyro.deterministic("xy_events",np.random.uniform(0,1,2*N).reshape(N,2))
  else:
    #print('here 2')
    N=t_events.shape[0];
    loglik=a_0*N-jnp.exp(a_0)*(t_max-t_min)
    numpyro.factor("t_events", loglik) 
    numpyro.factor("xy_events", loglik)



def simulate_uniform_Poisson(args):
  t_min=args['t_min']
  t_max=args['t_max']
  x_min=args['x_min']
  x_max=args['x_max']
  y_min=args['y_min']
  y_max=args['y_max']
  a_0=args['a_0']
  b_0=args['b_0']
  n=args['n_test']
  t_events=args['t_events']
  #xy_events=args['xy_events']
  
  if a_0==None:     
    a_0 = np.random.Gamma(2, 2)#=1# or add a prior

  if b_0==None:     
    b_0 = np.randmo.Gamma(2, 2)#=1# or add a prior

  #N=np.random.poisson(np.exp(a_0+b_0)*(t_max-t_min))
  N=n
  t_events=np.sort(np.random.uniform(t_min,t_max,N));
  x_events=np.random.uniform(x_min,x_max,N)
  y_events=np.random.uniform(y_min,y_max,N)
  loglik=a_0*N-np.exp(a_0)*(t_max-t_min)
  return t_events[0:n], x_events[0:n], y_events[0:n]#T_pred_all,X_pred_all,Y_pred_all


#@title
def rej_sampling_new(N, grid, gp_function, n):
  
  f_max=np.max(gp_function);
  ids=np.arange(0, n)
  if N<100:
    N_max=N*100
  else:
    N_max=N*10;
  index=np.random.choice(ids,N_max)

  candidate_points=grid[index];

  U=np.random.uniform(0, f_max, N_max);
  indices=jnp.where(U<gp_function[index]);
  accepted_points=grid[index][indices][0:N]
  accepted_f=gp_function[index][indices][0:N]
  return jnp.array(index[indices][0:N]), accepted_points, accepted_f


def lambda_S(nu_0, const_S, alpha,beta,t,event_times):
    #nu_0=exp(a_0)
    #nu_0=exp(a_0+f(t))integ_back_xy
    return nu_0 + const_S*np.sum(alpha*beta*np.exp(-beta*(t-event_times))[event_times<t])


def simulate_spatiotemporal_hawkes_predictions(past_times, past_locs, N_new, x_min, x_max, y_min, y_max, lambda_0, alpha, beta, sigma_sq, integ_xy_back,background, f_t=None):
    #f_t and f_s will be evaluated on a grid
    #### simulates N hawkes event pairs (s,t)
    T_train=50
    a = 0; b = 0; c = 0;
    back_points = 0
    gauss_points = 0
    x_t_test=np.arange(T_train,80,1)
    
    N_past=len(past_times);
    event_times = np.zeros(N_past+N_new); event_spatial_X=np.zeros(N_past+N_new); event_spatial_Y=np.zeros(N_past+N_new);
     
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
              if len(x_t_test.shape) == 2:

                ind=find_index(S_a, x_t_test[0])
              else:
                ind=find_index(S_a, x_t_test)

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


