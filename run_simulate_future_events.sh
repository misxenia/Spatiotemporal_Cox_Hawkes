#to run from terminal you do 
#	chmod +x ./run_simulate_future_events.sh
#	./run_simulate_future_events.sh

for ((k=43; k<100; k++))
do
    python run_simulate_future_events.py --dataset_name 'Poisson' --simulation_number "$k" --n_pred 200 #> "output/D1/D1M1S$i.txt"
done