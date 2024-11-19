# Run multiple training experiments with different parameters
python train.py \
    --data_path data/grid_20/dataset_20_100k_rotated.jsonl \
    --max_context_length 20 \
    --batch_size 3000 \
    --num_epochs 150 \
    --lr 5e-3 \
    --weight_decay 5e-9 \
    --plot True \
    --n_layers 4 \
    --n_heads 4

# python train.py \
#     --data_path data/dataset_5/dataset_5_sample_small.jsonl \
#     --max_context_length 5 \
#     --batch_size 5 \
#     --num_epochs 20 \
#     --lr 1e-3 \
#     --weight_decay 5e-9 \
#     --plot True \
#     --n_layers 2 \
#     --n_heads 4
