import torch
from lightning.pytorch.callbacks import Callback
from matplotlib import pyplot as plt
import pickle, os, datetime, argparse, json
from lightning.pytorch.loggers import CSVLogger
import glob

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl4co.envs import TSPEnv
from rl4co.envs.routing import TSPEnv, TSPGenerator
from rl4co.utils.trainer import RL4COTrainer

from distributions import Uniform, DualUniform, RandomUniform, FuzzyCircle, HybridSampler
from policy.policy_hooked import HookedAttentionModelPolicy
from policy.reinforce_clipped import REINFORCEClipped

def main(args):
    # Define the location distribution
    # loc_sampler = DualUniform((0, .2), (1, .4), (.2, 0), (.4, 1))
    # generator = TSPGenerator(num_loc=args.num_loc, loc_sampler=loc_sampler)
    # loc_sampler = RandomUniform(length_factor=5.0)

    # base_sampler = Uniform(min_x=0.0, max_x=1.0, min_y=0.0, max_y=1.0)

    # hybrid_sampler = HybridSampler(
    #     dist1=FuzzyCircle(
    #         radius_mean_lower=0.3,
    #         radius_mean_upper=0.6,
    #         radius_std_lower=0.02,
    #         radius_std_upper=0.05,
    #         center=(0.5, 0.5)
    #     ),
    #     dist2=FuzzyCircle(
    #         radius_mean_lower=0.2,
    #         radius_mean_upper=0.3,
    #         radius_std_lower=0.02,
    #         radius_std_upper=0.05,
    #         center=(0.4, 0.4)
    #     )
    # )

    # Hybrid sampler of uniform, random uniform, and fuzzy circle
    hybrid_sampler = HybridSampler([
        FuzzyCircle(
            radius_mean_lower=0.3,
            radius_mean_upper=0.4,
            radius_std_lower=0.02,
            radius_std_upper=0.1,
            random_center=True,
            center_x_range=(0.3, 0.7),
            center_y_range=(0.3, 0.7)
        ),
        RandomUniform(
            min_loc=0.0,
            max_loc=1.0,
            length_factor=5.0
        ),
        Uniform(
            min_x=0.0,
            max_x=1.0,
            min_y=0.0,
            max_y=1.0
        )
    ])

    generator = TSPGenerator(num_loc=args.num_loc, loc_sampler=hybrid_sampler)
    env = TSPEnv(generator=generator)
    

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    val_td = env.reset(batch_size=[args.num_val]).to(device)

    num_epochs = args.num_epochs
    num_instances = args.num_instances
    num_val = args.num_val

    class LossCallback(Callback):
        def __init__(self):
            self.losses = []

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
            if isinstance(outputs, dict) and 'loss' in outputs:
                self.losses.append(outputs['loss'].item())

    class ResultsCallback(Callback):
        def __init__(self, run_dir):
            self.actions = []
            self.rewards = []
            self.run_dir = run_dir

        def on_train_epoch_end(self, trainer, pl_module, unused=0):
            # test a greedy rollout on the same instances from val_td
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            policy = pl_module.policy.to(device)
            out = policy(val_td.clone(), phase="test", decode_type="greedy", return_actions=True)
            self.actions.append(out['actions'].cpu().detach())
            self.rewards.append(out['reward'].cpu().detach())
            
            # Save the current actions and rewards separately for each epoch
            results = {
                'actions': out['actions'].cpu().detach(),  # Save only current epoch's actions
                'rewards': out['reward'].cpu().detach()    # Save only current epoch's rewards
            }
            pickle.dump(results, open(f"{self.run_dir}/results/results_epoch_{trainer.current_epoch}.pkl", "wb"))

    class CheckpointCallback(Callback):
        def __init__(self, run_dir, checkpoint_freq):
            self.run_dir = run_dir
            self.checkpoint_freq = checkpoint_freq
            
        def on_train_epoch_end(self, trainer, pl_module, unused=0):
            # Save checkpoint if it's a checkpoint epoch or the last epoch
            if (trainer.current_epoch + 1) % self.checkpoint_freq == 0 or \
               (trainer.current_epoch + 1) == trainer.max_epochs:
                checkpoint_path = os.path.join(
                    self.run_dir, 
                    'checkpoints', 
                    f'checkpoint_epoch_{trainer.current_epoch + 1}.ckpt'
                )
                trainer.save_checkpoint(checkpoint_path)

    # Policy: AM with dropout
    policy = HookedAttentionModelPolicy(env_name=env.name,
                                embed_dim=args.embed_dim,
                                num_encoder_layers=args.n_encoder_layers,
                                num_heads=8,
                                temperature=args.temperature,
                                dropout=args.dropout,
                                attention_dropout=args.attention_dropout,
                                )

    # RL Model: Custom Clipped REINFORCE and greedy rollout baseline
    model = REINFORCEClipped(env,
                        policy,
                        baseline="rollout",
                        baseline_kwargs={"warmup": 1000},
                        batch_size=args.batch_size,
                        train_data_size=num_instances,
                        val_data_size=num_val,
                        optimizer_kwargs={"lr": args.lr},
                        )

    run_name = f"./runs/{args.run_name}"
    os.makedirs(run_name, exist_ok=True)
    os.makedirs(f"{run_name}/results", exist_ok=True)
    os.makedirs(f"{run_name}/checkpoints", exist_ok=True)


    # Log the config
    config = {
        "lr": args.lr,
        "num_epochs": num_epochs,
        "num_instances": num_instances,
        "num_val": num_val,
        "num_loc": args.num_loc,
        "temperature": args.temperature,
        "embed_dim": args.embed_dim,
        "n_encoder_layers": args.n_encoder_layers,
        "checkpoint_freq": args.checkpoint_freq,
        "dropout": args.dropout,
        "attention_dropout": args.attention_dropout,
    }
    with open(f"{run_name}/config.json", "w") as f:
        json.dump(config, f)
    pickle.dump(val_td, open(f"{run_name}/val_td.pkl", "wb"))
    pickle.dump(env, open(f"{run_name}/env.pkl", "wb"))

    # Define the callbacks
    loss_callback = LossCallback()
    results_callback = ResultsCallback(run_name)
    checkpoint_callback = CheckpointCallback(run_name, args.checkpoint_freq)

    if args.load_checkpoint:
        # Check if it's just an epoch number
        if args.load_checkpoint.isdigit():
            checkpoint_path = f"{run_name}/checkpoints/checkpoint_epoch_{args.load_checkpoint}.ckpt"
        else:
            checkpoint_path = args.load_checkpoint
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint {checkpoint_path} not found!")
            # Try to find the latest checkpoint if specified one doesn't exist
            checkpoint_files = glob.glob(f"{run_name}/checkpoints/checkpoint_epoch_*.ckpt")
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                print(f"Loading latest available checkpoint: {latest_checkpoint}")
                checkpoint_path = latest_checkpoint
            else:
                print("No checkpoints found. Starting from scratch.")
                checkpoint_path = None
        else:
            print(f"Loading from checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None

    trainer = RL4COTrainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=1,
        logger=CSVLogger(save_dir=run_name, name="logs"),
        callbacks=[loss_callback, results_callback, checkpoint_callback],
        gradient_clip_val=None,
    )

    print("Training...")
    if checkpoint_path:
        trainer.fit(model, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model)
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_instances", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_val", type=int, default=10)
    parser.add_argument("--num_loc", type=int, default=20)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_encoder_layers", type=int, default=3)
    parser.add_argument("--checkpoint_freq", type=int, default=10, 
                       help="Save checkpoint every N epochs")
    parser.add_argument("--dropout", type=float, default=0.0,
                       help="Dropout rate for feedforward layers")
    parser.add_argument("--attention_dropout", type=float, default=0.0,
                       help="Dropout rate for attention layers")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                       help="Path to checkpoint file or epoch number to load (e.g., 'runs/run_name/checkpoints/checkpoint_epoch_260.ckpt' or just '260')")
    args = parser.parse_args()
    main(args)


    # example command:
    # python train.py --lr 1e-4 --num_epochs 10 --num_instances 1000 --num_val 10 --num_loc 20