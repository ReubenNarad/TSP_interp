#!/bin/bash

# Train config
num_epochs=2000
num_instances=100_000
num_val=100
num_loc=100
batch_size=1024
temperature=0.4
lr=5e-6
checkpoint_freq=20
dropout=0.1
attention_dropout=0.1

# Load from checkpoint (set to empty string to start from scratch)
load_checkpoint="1260"

# Model config
embed_dim=256
n_encoder_layers=5

# Run name
run_name="Long_RandomUniform"



# Construct the checkpoint argument
if [ -n "$load_checkpoint" ]; then
    checkpoint_arg="--load_checkpoint $load_checkpoint"
else
    checkpoint_arg=""
fi

# Train
python -m policy.train_vanilla \
    --lr $lr \
    --num_epochs $num_epochs \
    --num_instances $num_instances \
    --num_val $num_val \
    --num_loc $num_loc \
    --run_name $run_name \
    --temperature $temperature \
    --embed_dim $embed_dim \
    --batch_size $batch_size \
    --n_encoder_layers $n_encoder_layers \
    --checkpoint_freq $checkpoint_freq \
    --dropout $dropout \
    --attention_dropout $attention_dropout \
    $checkpoint_arg

python -m env.solve_td_with_concorde --run_name $run_name

python -m policy.eval --run_name $run_name --num_epochs $num_epochs
