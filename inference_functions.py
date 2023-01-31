import os
import pickle 
import time 
import jax
import jax.numpy as jnp
from jax import random, lax, jit, ops
from jax.example_libraries import stax


from functools import partial

# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS, init_to_median, Predictive, RenyiELBO
from numpyro.diagnostics import hpdi
from random import randint


from utils import difference_matrix

def vae_encoder_temporal(hidden_dim, z_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Elu,
        stax.FanOut(2),
        stax.parallel(
            stax.Dense(z_dim, W_init=stax.randn()), # mean
            stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp), # std -- i.e. diagonal covariance
        ),
    )


def vae_decoder_temporal(hidden_dim, out_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Elu,
        stax.Dense(out_dim, W_init=stax.randn()) 
    )


def vae_model_temporal(batch, hidden_dim, z_dim):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    decode = numpyro.module("decoder", vae_decoder_temporal(hidden_dim, out_dim), (batch_dim, z_dim))
    z = numpyro.sample("z", dist.Normal(jnp.zeros((z_dim,)), jnp.ones((z_dim,))))
    v_decode = decode(z)    
    return numpyro.sample("obs", dist.Normal(v_decode, .1), obs=batch) 
    

def vae_guide_temporal(batch, hidden_dim, z_dim):
    batch = jnp.reshape(batch, (batch.shape[0], -1))
    batch_dim, out_dim = jnp.shape(batch)
    encode = numpyro.module("encoder", vae_encoder_temporal(hidden_dim, z_dim), (batch_dim, out_dim))
    z_loc, z_std = encode(batch)
    z = numpyro.sample("z", dist.Normal(z_loc, z_std))
    return z



#@title

def vae_encoder_spatial(hidden_dim1, hidden_dim2, z_dim):
  return stax.serial(
      stax.Dense(hidden_dim1, W_init=stax.randn()),
      stax.Elu,
      stax.Dense(hidden_dim2, W_init=stax.randn()),
      stax.Elu,
      stax.FanOut(2),
      stax.parallel(
          stax.Dense(z_dim, W_init=stax.randn()), # mean
          stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp), # std -- i.e. diagonal covariance
      ),
  )


def vae_decoder_spatial(hidden_dim1, hidden_dim2, out_dim):
  return stax.serial(
      stax.Dense(hidden_dim1, W_init=stax.randn()),
      stax.Elu,
      stax.Dense(hidden_dim2, W_init=stax.randn()),
      stax.Elu,
      stax.Dense(out_dim, W_init=stax.randn()) 
  )

def vae_model_spatial(batch, hidden_dim1, hidden_dim2, z_dim):
  batch = jnp.reshape(batch, (batch.shape[0], -1))
  batch_dim, out_dim = jnp.shape(batch)
  decode = numpyro.module("decoder", vae_decoder_spatial(hidden_dim1, hidden_dim2, out_dim), (batch_dim, z_dim))
  z = numpyro.sample("z", dist.Normal(jnp.zeros((z_dim,)), jnp.ones((z_dim,))))
  gen_loc = decode(z)    
  return numpyro.sample("obs", dist.Normal(gen_loc, .1), obs=batch) 
  

def vae_guide_spatial(batch, hidden_dim1, hidden_dim2, z_dim):
  batch = jnp.reshape(batch, (batch.shape[0], -1))
  batch_dim, out_dim = jnp.shape(batch)
  encode = numpyro.module("encoder", vae_encoder_spatial(hidden_dim1, hidden_dim2, z_dim), (batch_dim, out_dim))
  z_loc, z_std = encode(batch)
  z = numpyro.sample("z", dist.Normal(z_loc, z_std))
  return z


def Hawkes_likelihood(args):
    t_events=args["t_events"]
    xy_events=args["xy_events"]
    N=t_events.shape[0]

    ####### LGCP BACKGROUND
    # temporal rate
    # mean
    a_0 = args['a_0']
    if args['background']in ['constant']:     
      b_0=0
      mu_xyt=numpyro.deterministic("mu_xyt",jnp.exp(a_0+b_0))
      Itot_txy_back=numpyro.deterministic("Itot_txy_back",mu_xyt*args['T'] )

    if args['background']=='LGCP':
      #zero mean temporal gp ft 
      z_temporal = numpyro.sample("z_temporal", dist.Normal(jnp.zeros(args["z_dim_temporal"]), jnp.ones(args["z_dim_temporal"])))
      decoder_nn_temporal = vae_decoder_temporal(args["hidden_dim_temporal"], args["n_t"])  
      decoder_params = args["decoder_params_temporal"]
      v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal))
      f_t = numpyro.deterministic("f_t", v_t[0:args["n_t"]])
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
    alpha = args['alpha']
    beta =  args['beta']
    
    #spatial gaussian kernel parameters     
    sigmax_2 = args['sigmax_2']
    sigmay_2 = args['sigmay_2']#numpyro.sample("sigmay_2", dist.HalfNormal(.5))##numpyro.sample("sigmay_2", dist.Normal(0.5,1))#Exponential(.3))
    
    #spatial gaussian kernel parameters     
    #sigmax_2 = numpyro.sample("sigmax_2", dist.Exponential(.1))
    #sigmay_2 = numpyro.sample("sigmay_2", dist.Gamma(0.5,1))#Exponential(.3))
    
    T,x_min,x_max,y_min,y_max = args['T'],args['x_min'],args['x_max'],args['y_min'],args['y_max']  
    
    T_diff=difference_matrix(t_events);
    S_mat_x = difference_matrix(xy_events[0])
    S_mat_y = difference_matrix(xy_events[1])
    S_diff_sq=(S_mat_x**2)/sigmax_2+(S_mat_y**2)/sigmay_2; 
    l_hawkes_sum=alpha*beta/(2*jnp.pi*jnp.sqrt(sigmax_2*sigmay_2))*jnp.exp(-beta*T_diff-0.5*S_diff_sq)
    l_hawkes = numpyro.deterministic('l_hawkes',jnp.sum(jnp.tril(l_hawkes_sum,-1),1))

    if args['background'] in ['Hawkes','constant']:
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




def spatiotemporal_hawkes_model(args):
    t_events=args["t_events"]
    xy_events=args["xy_events"]
    N=t_events.shape[0]

    ####### LGCP BACKGROUND
    # temporal rate
    # mean
    if args['background']in ['constant','Hawkes']:     
      a_0 = numpyro.sample("a_0", dist.Normal(.5,1))#dist.HalfNormal(1)  dist.Normal(.5,1), dist.Normal(.5,1)2,2
      b_0=0 #b_0 = numpyro.sample("b_0", dist.Normal(1.5,1))
      mu_xyt=numpyro.deterministic("mu_xyt",jnp.exp(a_0+b_0))
      Itot_txy_back=numpyro.deterministic("Itot_txy_back",mu_xyt*args['T'] )

    if args['background']=='LGCP':
      a_0 = numpyro.sample("a_0", dist.Normal(0,2))# this was 0,2
      #zero mean temporal gp ft 
      z_temporal = numpyro.sample("z_temporal", dist.Normal(jnp.zeros(args["z_dim_temporal"]), jnp.ones(args["z_dim_temporal"])))
      decoder_nn_temporal = vae_decoder_temporal(args["hidden_dim_temporal"], args["n_t"])  
      decoder_params = args["decoder_params_temporal"]
      v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal))
      f_t = numpyro.deterministic("f_t", v_t[0:args["n_t"]])
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
    alpha = numpyro.sample("alpha", dist.HalfNormal(0.5))#numpyro.sample("alpha", dist.Gamma(.5,1))##numpyro.sample("alpha", dist.HalfNormal(0.5,2))# has to be within 0,1
    beta = numpyro.sample("beta", dist.HalfNormal(0.3))#numpyro.sample("beta", dist.Gamma(.7,1)) FOR CONSTANT HAWKES HALFNOMRAL(.3)
    
    #spatial gaussian kernel parameters     
    sigmax_2 = numpyro.sample("sigmax_2", dist.HalfNormal(1)) # dist.Exponential(.1)) FOR CONSTANT HAWKES HALFNOMRAL(2)
    sigmay_2 = sigmax_2#numpyro.sample("sigmay_2", dist.HalfNormal(.5))##numpyro.sample("sigmay_2", dist.Normal(0.5,1))#Exponential(.3))
    
    #spatial gaussian kernel parameters     
    #sigmax_2 = numpyro.sample("sigmax_2", dist.Exponential(.1))
    #sigmay_2 = numpyro.sample("sigmay_2", dist.Gamma(0.5,1))#Exponential(.3))
    
    T,x_min,x_max,y_min,y_max = args['T'],args['x_min'],args['x_max'],args['y_min'],args['y_max']  
    
    T_diff=difference_matrix(t_events);
    S_mat_x = difference_matrix(xy_events[0])
    S_mat_y = difference_matrix(xy_events[1])
    S_diff_sq=(S_mat_x**2)/sigmax_2+(S_mat_y**2)/sigmay_2; 
    l_hawkes_sum=alpha*beta/(2*jnp.pi*jnp.sqrt(sigmax_2*sigmay_2))*jnp.exp(-beta*T_diff-0.5*S_diff_sq)
    l_hawkes = numpyro.deterministic('l_hawkes',jnp.sum(jnp.tril(l_hawkes_sum,-1),1))

    if args['background'] in ['Hawkes','constant']:
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


#@title

def spatiotemporal_LGCP_model(args):
    t_events=args["t_events"];#print(t_events.shape[0],'time events')
    xy_events=args["xy_events"];#print(xy_events.shape[0],'xy events')
    n_obs=t_events.shape[0]
    
    #temporal rate
    a_0 = numpyro.sample("a_0", dist.Normal(2, 2))   
    #zero mean temporal gp 
    z_temporal = numpyro.sample("z_temporal", dist.Normal(jnp.zeros(args["z_dim_temporal"]), jnp.ones(args["z_dim_temporal"])))
    decoder_nn_temporal = vae_decoder_temporal(args["hidden_dim_temporal"], args["n_t"])  
    decoder_params = args["decoder_params_temporal"]
    v_t = numpyro.deterministic("v_t", decoder_nn_temporal[1](decoder_params, z_temporal))
    f_t = numpyro.deterministic("f_t", v_t[0:args["n_t"]])
    #rate_t = numpyro.deterministic("rate_t",jnp.exp(f_t+a_0))
    Itot_t=numpyro.deterministic("Itot_t", jnp.sum(jnp.exp(f_t))/args["n_t"]*args["T"])
    #Itot = numpyro.deterministic("Itot", v[len(x)])
    #Itot_t=jnp.trapz(mu_0*jnp.exp(f), back_t)
    f_t_i=f_t[args["indices_t"]]

    # spatial rate
    # mean
    b_0 = 0#numpyro.sample("b_0", dist.Normal(2, 5))#set to zero for identifiability but take into account when interpreting the output numpyro.sample("b_0", dist.Normal(6, 3))
    # zero mean spatial gp
    z_spatial = numpyro.sample("z_spatial", dist.Normal(jnp.zeros(args["z_dim_spatial"]), jnp.ones(args["z_dim_spatial"])))
    decoder_nn = vae_decoder_spatial(args["hidden_dim2_spatial"], args["hidden_dim1_spatial"], args["n_xy"])  
    decoder_params = args["decoder_params_spatial"]
    f_xy = numpyro.deterministic("f_xy", decoder_nn[1](decoder_params, z_spatial))
    #rate_xy = numpyro.deterministic("rate_xy",jnp.exp(f_xy+b_0))
    Itot_xy=numpyro.deterministic("Itot_xy", jnp.sum(jnp.exp(f_xy))/args["n_xy"]**2)
    f_xy_i=f_xy[args["indices_xy"]]

    #numpyro.deterministic("rate_txy",jnp.exp(f_t*f_xy+a_0+b_0))
    loglik=jnp.sum(f_t_i+f_xy_i+a_0+b_0)    
    I_tot_txy=numpyro.deterministic("I_tot_txy",Itot_xy*Itot_t*jnp.exp(a_0+b_0))
    loglik-=I_tot_txy
    numpyro.deterministic("loglik",loglik)

    numpyro.factor("t_events", loglik)
    numpyro.factor("xy_events", loglik)


    #@title
def run_mcmc(rng_key, model_mcmc, args):
    start = time.time()

    init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(model_mcmc, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=args["num_warmup"],
        num_samples=args["num_samples"],
        num_chains=args["num_chains"],
        thinning=args["thinning"],
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, args)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc