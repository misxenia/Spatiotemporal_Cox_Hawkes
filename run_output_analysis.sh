#to run from terminal you do 
#	chmod +x ./run_output_analysis.sh
#	./run_output_analysis.sh

# LGCP_Hawkes, LGCP, Hawkes, Hawkes
for ((k=0; k<100; k++))
do
    python run_output_analysis_values.py --dataset_name 'Hawkes' --simulation_number "$k" --n_pred 200 --model_name 'LGCP_Hawkes' --simulate_predictions 'False' --calculate_error 'False'
done