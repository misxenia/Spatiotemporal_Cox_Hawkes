## Spatiotemporal_Cox_Hawkes
This code is used to run all the experiments in the paper "Cox-Hawkes: doubly stochastic spatiotemporal Poisson processes"
accepted at the Transactions of Machine Learning Research. 
For the simulation experiment the code allows you to simulate data from the LGCP-Hawkes model and perform inference. 
For the misspecification experiment the code allows you to simulate data from the models LGCP, LGCP-Hawkes and Hawkes and perform inference under all three. Then, future data are predicted using the estimated model and then compared with the ones under the 'true' model.
For the true dataset on gunshots, the first experiment is on inference under the LGCP-Hawkes model. The second is a comparison of the three models in prediction.

#### For code requirements see
requirements.txt

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

Reference for this code is [1]. 
The trained decoders and encoder/decoder functions are provided by Dr Elisaveta Semenova following the proposals in [2]. 

[1] X. Miscouridou, G. Mohler, S. Bhatt, S. Flaxman, S. Mishra, Cox-Hawkes: Doubly stochastic spatiotemporal poisson point process, Transaction of Machine Learning Research, 2023

[2] Elizaveta Semenova, Yidan Xu, Adam Howes, Theo Rashid, Samir Bhatt, B. Swapnil Mishra, and Seth R.
Flaxman. Priorvae: encoding spatial priors with variational autoencoders for small-area estimation. Royal
Society Publishing, pp. 73–80, 2022 

