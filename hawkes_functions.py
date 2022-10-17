## simulation function
# general libraries

import time
import os

from utils import exp_sq_kernel
from vae_functions import vae_encoder_temporal,vae_decoder_temporal,vae_model_temporal,vae_guide_temporal,vae_encoder_spatial,vae_decoder_spatial,vae_model_spatial,vae_guide_spatial

# Numpyro
import numpyro
# Numpyro
import numpyro
import numpyro.distributions as dist
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS, init_to_median, Predictive, RenyiELBO
# JAX
import jax
import jax.numpy as jnp

from utils import exp_sq_kernel, find_index, difference_matrix



def spatiotemporal_LGCP(T, x_t, x_xy, gp_kernel=exp_sq_kernel, jitter=1e-5, a_0=None, b_0=None, var_t=None, length_t=None, var_xy=None, length_xy=None, t=None, xy=None):
      
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




def spatiotemporal_hawkes_model(args):
    t_events=args["t_events"]
    xy_events=args["xy_events"]
    N=t_events.shape[0]

    #########LGCP BACKGROUND
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
    alpha = numpyro.sample("alpha", dist.HalfNormal(0.8,2))#numpyro.sample("alpha", dist.Gamma(.5,1))##numpyro.sample("alpha", dist.HalfNormal(0.5,2))# has to be within 0,1
    beta = numpyro.sample("beta", dist.HalfNormal(0.3,2))#numpyro.sample("beta", dist.Gamma(.7,1))
    
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


   

