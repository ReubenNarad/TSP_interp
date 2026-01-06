import argparse
import datetime
import glob
import json
import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl4co.envs.routing.atsp.env import ATSPEnv
from rl4co.envs.routing.atsp.generator import ATSPGenerator
from rl4co.utils.trainer import RL4COTrainer
from rl4co.models.zoo.matnet.policy import MatNetPolicy

from env.seattle_atsp_generator import TSPLIBSubmatrixConfig, TSPLIBSubmatrixGenerator
from env.pool_submatrix_generator import PoolSubmatrixConfig, PoolSubmatrixGenerator
from policy.reinforce_clipped import REINFORCEClipped


def main(args):
    # Environment / generator
    if args.tsplib_path and args.pool_dir:
        raise ValueError("Only one of --tsplib_path or --pool_dir may be set.")

    if args.pool_dir:
        gen_cfg = PoolSubmatrixConfig(
            pool_dir=args.pool_dir,
            num_loc=args.num_loc,
            seed=args.seed,
            mmap=not args.pool_in_memory,
            cost_scale=args.cost_scale,
        )
        generator = PoolSubmatrixGenerator(**asdict(gen_cfg))
        env = ATSPEnv(generator=generator)
    elif args.tsplib_path:
        gen_cfg = TSPLIBSubmatrixConfig(
            tsp_path=args.tsplib_path,
            num_loc=args.num_loc,
            symmetrize=args.symmetrize,
            seed=args.seed,
            cost_scale=args.cost_scale,
        )
        generator = TSPLIBSubmatrixGenerator(**asdict(gen_cfg))
        env = ATSPEnv(generator=generator)
    else:
        generator = ATSPGenerator(num_loc=args.num_loc)
        env = ATSPEnv(generator=generator)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    # If we are sampling from TSPLIB or a pool, persist per-instance coords/indices for plotting later.
    if args.tsplib_path or args.pool_dir:
        cm, idxs, coords = generator.sample_with_meta(int(args.num_val))
        # Save metadata for plotting/animation
        run_dir_preview = os.path.join("./runs", args.run_name)
        os.makedirs(run_dir_preview, exist_ok=True)
        np_path = os.path.join(run_dir_preview, "val_indices.npy")
        import numpy as np
        np.save(np_path, idxs)
        if coords is not None:
            np.save(os.path.join(run_dir_preview, "val_coords_lonlat.npy"), coords)
        # Use the sampled cost matrices as the fixed validation batch
        from tensordict import TensorDict
        val_td = env.reset(td=TensorDict({"cost_matrix": cm}, batch_size=[int(args.num_val)])).to(device)
    else:
        val_td = env.reset(batch_size=[args.num_val]).to(device)

    # Policy / model
    policy = MatNetPolicy(
        env_name=env.name,
        embed_dim=args.embed_dim,
        num_encoder_layers=args.n_encoder_layers,
        num_heads=args.num_heads,
        normalization=args.normalization,
        use_graph_context=args.use_graph_context,
        bias=args.bias,
        tanh_clipping=args.tanh_clipping,
        temperature=args.temperature,
    )

    model = REINFORCEClipped(
        env,
        policy,
        baseline="rollout",
        baseline_kwargs={"warmup": 1000},
        batch_size=args.batch_size,
        train_data_size=args.num_instances,
        val_data_size=args.num_val,
        optimizer_kwargs={"lr": args.lr},
        clip_val=args.clip_val,
        lr_decay=args.lr_decay,
        min_lr=args.min_lr,
        exp_gamma=args.exp_gamma,
        dataloader_num_workers=args.num_workers,
        shuffle_train_dataloader=args.shuffle_train,
    )

    # Run dir / logging
    run_dir = os.path.join("./runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    config = vars(args).copy()
    config["env_name"] = env.name
    config["timestamp"] = datetime.datetime.now().isoformat()
    for k, v in list(config.items()):
        if isinstance(v, Path):
            config[k] = str(v)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    pickle.dump(val_td, open(os.path.join(run_dir, "val_td.pkl"), "wb"))
    pickle.dump(env, open(os.path.join(run_dir, "env.pkl"), "wb"))

    class ResultsCallback(Callback):
        def __init__(self, run_dir: str):
            self.run_dir = run_dir

        def on_train_epoch_end(self, trainer, pl_module, unused=0):
            every = int(args.save_results_every)
            if every <= 0 or (trainer.current_epoch % every) != 0:
                return

            try:
                with torch.inference_mode():
                    device_local = pl_module.device
                    td = val_td.clone().to(device_local)

                    # Evaluate in chunks to reduce peak memory.
                    b = int(td.batch_size[0]) if hasattr(td, "batch_size") else int(td["cost_matrix"].shape[0])
                    chunk = int(args.results_eval_batch_size)
                    actions = []
                    rewards = []
                    for s in range(0, b, chunk):
                        td_chunk = td[s : s + chunk].clone()
                        out = pl_module.policy(td_chunk, pl_module.env, phase="test", decode_type="greedy", return_actions=True)
                        actions.append(out["actions"].cpu())
                        rewards.append(out["reward"].cpu())
                    results = {
                        "actions": torch.cat(actions, dim=0),
                        "rewards": torch.cat(rewards, dim=0),
                    }

                pickle.dump(
                    results,
                    open(os.path.join(self.run_dir, "results", f"results_epoch_{trainer.current_epoch}.pkl"), "wb"),
                )
            except torch.OutOfMemoryError:
                # Avoid crashing long runs if optional per-epoch eval doesn't fit.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"[ResultsCallback] OOM during epoch-end greedy eval; skipping results for epoch {trainer.current_epoch}. "
                    f"Consider lowering --num_val or --results_eval_batch_size, or setting --save_results_every 0."
                )

    class CheckpointCallback(Callback):
        def __init__(self, run_dir: str, checkpoint_freq: int):
            self.run_dir = run_dir
            self.checkpoint_freq = checkpoint_freq

        def on_train_epoch_end(self, trainer, pl_module, unused=0):
            if (trainer.current_epoch + 1) % self.checkpoint_freq == 0 or (
                trainer.current_epoch + 1
            ) == trainer.max_epochs:
                checkpoint_path = os.path.join(
                    self.run_dir, "checkpoints", f"checkpoint_epoch_{trainer.current_epoch + 1}.ckpt"
                )
                trainer.save_checkpoint(checkpoint_path)

    # Resume support (match train_vanilla behavior)
    if args.load_checkpoint:
        if args.load_checkpoint.isdigit():
            checkpoint_path = os.path.join(run_dir, "checkpoints", f"checkpoint_epoch_{args.load_checkpoint}.ckpt")
        else:
            checkpoint_path = args.load_checkpoint
        if not os.path.exists(checkpoint_path):
            checkpoint_files = glob.glob(os.path.join(run_dir, "checkpoints", "checkpoint_epoch_*.ckpt"))
            checkpoint_path = max(checkpoint_files, key=os.path.getctime) if checkpoint_files else None
    else:
        checkpoint_path = None

    precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    trainer = RL4COTrainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices=1,
        precision=precision,
        log_every_n_steps=args.log_every_n_steps,
        logger=CSVLogger(save_dir=run_dir, name="logs"),
        callbacks=[ResultsCallback(run_dir), CheckpointCallback(run_dir, args.checkpoint_freq)],
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
    parser.add_argument("--run_name", type=str, required=True)

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_instances", type=int, default=10_000)
    parser.add_argument("--num_val", type=int, default=128)
    parser.add_argument("--num_loc", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_freq", type=int, default=5)
    parser.add_argument("--clip_val", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers for on-the-fly instance generation.")
    parser.add_argument("--shuffle_train", action="store_true", help="Shuffle train dataloader (usually not needed for generated data).")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Lightning logging frequency (train).")
    parser.add_argument(
        "--save_results_every",
        type=int,
        default=1,
        help="Write greedy eval actions/rewards to runs/<run>/results/ every N epochs (0 disables).",
    )
    parser.add_argument(
        "--results_eval_batch_size",
        type=int,
        default=64,
        help="Batch size for epoch-end greedy eval used to write results_epoch_*.pkl.",
    )

    parser.add_argument("--lr_decay", type=str, default="none", choices=["none", "cosine", "linear", "exponential"])
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--exp_gamma", type=float, default=None)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # MatNet policy params
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_encoder_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--normalization", type=str, default="instance", choices=["batch", "instance", "layer", "none"])
    parser.add_argument("--use_graph_context", action="store_true")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument(
        "--tanh_clipping",
        type=float,
        default=10.0,
        help="Tanh clipping for decoder logits (stability; Attention Model commonly uses 10).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (train); keep at 1.0 unless you know why you want to change it.",
    )

    # TSPLIB-based sampling
    parser.add_argument("--tsplib_path", type=Path, default=None, help="Path to TSPLIB FULL_MATRIX instance to subsample.")
    parser.add_argument("--symmetrize", type=str, default="none", choices=["none", "min", "avg"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cost_scale",
        type=float,
        default=1.0,
        help="Divide the sampled cost_matrix by this constant for numerical stability (also scales rewards).",
    )

    # Pool-based sampling (precomputed KxK matrix)
    parser.add_argument("--pool_dir", type=Path, default=None, help="Directory containing cost_matrix.npy + coords_lonlat.npy + node_ids.npy + meta.json.")
    parser.add_argument("--pool_in_memory", action="store_true", help="Load pool cost matrix into RAM (default uses numpy memmap).")

    args = parser.parse_args()
    main(args)
