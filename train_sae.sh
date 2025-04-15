echo "Training SAE"

run_name="Test_Clusters_8_layers"

# for l1_coef in 0.001 0.0001
# do
#   for expansion_factor in 4.0
#   do
#     for k_ratio in 0.1
#     do

l1_coef=0.001
expansion_factor=4.0
k_ratio=0.1
num_epochs=20

python -m sae.train_topk \
  --run_dir ./runs/$run_name \
  --activation_key encoder_output \
  --expansion_factor $expansion_factor \
  --batch_size 64 \
  --l1_coef $l1_coef \
  --lr 1e-3 \
  --num_epochs $num_epochs \
  --num_workers 4 \
  --reinit_dead \
  --save_freq 5 \
  --reinit_freq 2 \
  --k_ratio $k_ratio

#     done
#   done
# done

