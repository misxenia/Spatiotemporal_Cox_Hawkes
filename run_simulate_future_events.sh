#to run from terminal you do 
#	chmod +x ./run_simulate_future_events.sh
#	./run_simulate_future_events.sh
# dataset name LGCP_only,
for ((k=0; k<100; k++))
do
    python run_simulate_future_events.py --dataset_name 'LGCP_Hawkes' --simulation_number "$k" --n_pred 10
done