#idea is  to 
#Generate 10k datasets from a set of parameters. 
#Estimate once from each each of four models for four different dataset types 
#(one fr each true model)
#model_name: LGCP, LGCP_Hawkes, Hawkes
#dataset_name: LGCP_only, LGCP_Hawkes, Hawkes
for ((i=0; i<1; i++))
do
    python run_inference.py --dataset_name 'Poisson' --simulation_number "$i" --model_name 'LGCP_Hawkes' --num_chains 2 --num_samples 1500 --num_warmup 500 --num_thinning 2 #> "output/D1/D1M1S$i.txt"
done
#to run from terminal you do 
#	chmod +x ./run_experiment.sh
#	./run_experiment.sh