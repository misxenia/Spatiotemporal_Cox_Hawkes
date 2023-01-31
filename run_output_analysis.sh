#to run from terminal you do 
#	chmod +x ./run_output_analysis.sh
#	./run_output_analysis.sh

# LGCP_Hawkes, LGCP, Hawkes, Hawkes
for ((k=0; k<100; k++))
do
    python run_output_analysis_loglik.py --dataset_name 'LGCP_Hawkes' --simulation_number "$k" --n_pred 20 --model_name 'Poisson' --simulate_predictions 'True'
done