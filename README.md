## Spatiotemporal_Cox_Hawkes
This code is used to run all the experiments in the paper. 
For the simulation experiment the code allows you to simulate data from the LGCP-Hawkes model and perform inference. 
For the misspecification experiment the code allows you to simulate data from the models LGCP, LGCP-Hawkes and Hawkes and perform inference under all three. Then, future data are predicted using the estimated model and then compared with the ones under the 'true' model.
For the true dataset on gunshots, the first experiment is on inference under the LGCP-Hawkes model. The second is a comparison of the three models in prediction.

### The code runs with:
* numpyro 0.6.0
* jax 0.2.13
* jaxlib 0.1.67 



#### Simulate Data
* Run the script simulate_spatiotemporal_cox_hawkes with your choice of model

#### Perform inference
* Run the run_inference.sh pecifying the model you are interested in

#### Simulate future (true) events
* Run the run_simualate_future_events.sh specifying the model you are interested in

#### Prediction analysis to compute errors
* run_output_analysis.sh

#### Report error using RMSE
* Run read_error.py


#### True dataset(gunshots)
* Perform inference under the model LGCP-Hawkes


