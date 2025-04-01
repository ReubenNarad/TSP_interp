from typing import Union, Any
import torch
import torch.nn as nn

from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class REINFORCEClipped(REINFORCE):
    """REINFORCE algorithm with gradient clipping.
    Inherits from REINFORCE and adds gradient clipping functionality.

    Args:
        clip_val: Maximum allowed value for gradient clipping
        clip_type: Type of gradient clipping ('norm' or 'value')
        **kwargs: Arguments passed to parent REINFORCE class
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        clip_val: float = 10.0,
        clip_type: str = "norm",
        **kwargs,
    ):
        super().__init__(env, policy, baseline, **kwargs)
        
        self.clip_val = clip_val
        self.clip_type = clip_type
        assert clip_type in ["norm", "value"], "clip_type must be either 'norm' or 'value'"
        
        # Disable automatic optimization
        self.automatic_optimization = False
        
        self.save_hyperparameters(logger=False)

    def training_step(self, batch: Any, batch_idx: int):
        """Override training_step to add gradient clipping"""
        # Get optimizer
        opt = self.optimizers()
        opt.zero_grad()
        
        # Compute loss using parent's shared_step
        result = self.shared_step(batch, batch_idx, phase="train")
        
        # Get loss from result (could be dict or tensor)
        loss = result['loss'] if isinstance(result, dict) else result
        
        # Manually compute gradients
        self.manual_backward(loss)
        
        if batch_idx % 100 == 0:
            # Get pre-clipping stats
            grad_norms = [param.grad.norm().item() for param in self.parameters() if param.grad is not None]
            pre_max, pre_mean = max(grad_norms), sum(grad_norms)/len(grad_norms)
        
        # Clip gradients
        if self.clip_type == "norm":
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_val)
        else:  # clip_type == "value"
            torch.nn.utils.clip_grad_value_(self.parameters(), self.clip_val)
            
        if batch_idx % 100 == 0:
            # Get post-clipping stats
            grad_norms = [param.grad.norm().item() for param in self.parameters() if param.grad is not None]
            post_max, post_mean = max(grad_norms), sum(grad_norms)/len(grad_norms)
            # print(f"Batch {batch_idx} - Grads (max/mean): pre-clip=({pre_max:.3f}/{pre_mean:.3f}), post-clip=({post_max:.3f}/{post_mean:.3f})")
            
        # Update weights
        opt.step()
        
        return result

    def configure_optimizers(self):
        """Override to prevent double optimization"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer_kwargs["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # reduce LR when reward stops improving
            factor=0.5,  # multiply LR by this factor
            patience=10,  # number of epochs with no improvement after which LR will be reduced
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/reward",  # metric to track
                "interval": "epoch",
                "frequency": 1
            }
        }
