echo "Collecting activations"

run_name="Test_Clusters_8_layers"

python -m sae.collect_activations \
 --run_name $run_name \
 --final_only 

## Optional: Generate a GIF of the activations + PCA of activations throughout training
# python -m sae.activations_gif \
#  --run_name $run_name \
#  --batch_size 64 \
#  --num_instances 1 \
#  --fps 10.0

