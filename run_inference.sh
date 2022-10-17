
# Generate 100 datasets from a set of parameters. 
# Estimate once from each each of four models for four different dataset types 
# (one fr each true model)

#dataset_name: LGCP_only, LGCP_Hawkes, Hawkes
#model_name: LGCP, LGCP_Hawkes, Hawkes, Poisson

for ((i=0; i<100; i++))
do
    python run_inference.py --dataset_name 'LGCP_only' --simulation_number "$i" --model_name 'LGCP' --num_chains 1 --num_samples 10 --num_warmup 10 --num_thinning 1
done
#to run from terminal you do 
#	chmod +x ./run_inference.sh
#	./run_inference.sh