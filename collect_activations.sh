echo "Collecting activations"

run_name="Test"

python -m sae.collect_activations \
 --run_name $run_name \
 --final_only 

# python -m sae.activations_gif \
#  --run_name $run_name \
#  --batch_size 64 \
#  --num_instances 1 \
#  --fps 10.0

