import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLayerTranscoder(nn.Module):
    """
    Sparse cross-layer transcoder inspired by Anthropic's attribution graph work.

    Given activations from an upstream layer, the model learns a sparse latent
    representation that reconstructs the activations of a downstream layer.
    Sparsity is enforced via a top-k winner-take-all non-linearity.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expansion_factor: float = 4.0,
        k_ratio: float = 0.05,
        k: Optional[int] = None,
        tied_weights: bool = False,
        init_method: str = "kaiming_uniform",
    ):
        """
        Args:
            input_dim: dimensionality of the source activations
            output_dim: dimensionality of the target activations
            expansion_factor: latent size as a multiple of the target dimension
            k_ratio: fraction of latent units permitted to stay active
            k: explicitly set the number of active units (overrides k_ratio)
            tied_weights: reuse encoder weights for the decoder dictionary
            init_method: dictionary initialisation strategy
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        latent_dim = max(1, int(expansion_factor * output_dim))
        self.latent_dim = latent_dim
        self.k = k if k is not None else max(1, int(k_ratio * latent_dim))
        self.tied_weights = tied_weights

        # Encoder maps source activations -> latent pre-activations
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)

        # Decoder reconstructs target activations from sparse latents
        if tied_weights:
            self.decoder_bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.decoder = nn.Linear(latent_dim, output_dim, bias=True)

        self._init_dictionary(init_method)

        # Track usage statistics to support dead-unit reinitialisation
        self.register_buffer("neuron_activity", torch.zeros(latent_dim))

    def _init_dictionary(self, init_method: str) -> None:
        """Initialise encoder/decoder weights following the requested scheme."""
        init_method = init_method.lower()
        if init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
            if not self.tied_weights:
                nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        elif init_method == "kaiming_normal":
            nn.init.kaiming_normal_(self.encoder.weight, a=math.sqrt(5))
            if not self.tied_weights:
                nn.init.kaiming_normal_(self.decoder.weight, a=math.sqrt(5))
        elif init_method == "xavier_uniform":
            nn.init.xavier_uniform_(self.encoder.weight)
            if not self.tied_weights:
                nn.init.xavier_uniform_(self.decoder.weight)
        elif init_method == "xavier_normal":
            nn.init.xavier_normal_(self.encoder.weight)
            if not self.tied_weights:
                nn.init.xavier_normal_(self.decoder.weight)
        else:
            raise ValueError(f"Unknown init_method '{init_method}'")

        nn.init.zeros_(self.encoder.bias)
        if not self.tied_weights:
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode source activations into a sparse latent representation.

        Returns:
            Tensor of shape [batch, latent_dim] with top-k sparsity.
        """
        pre_act = self.encoder(x)

        # Winner-take-all top-k gating
        values, _ = torch.topk(pre_act, self.k, dim=1)
        thresholds = values[:, -1:].detach()
        # Shift by threshold and apply ReLU to retain only the winning units
        sparse_latent = F.relu(pre_act - thresholds)

        if self.training:
            self.neuron_activity += (sparse_latent > 0).sum(dim=0)

        return sparse_latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse latents back into the target activation space."""
        if self.tied_weights:
            return F.linear(z, self.encoder.weight.t(), self.decoder_bias)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstructions and latent codes."""
        z = self.encode(x)
        y_hat = self.decode(z)
        return y_hat, z

    # --- diagnostics ----------------------------------------------------- #
    @torch.no_grad()
    def get_dead_neurons(self, min_activity: float = 1e-5) -> torch.Tensor:
        """Boolean mask marking units that never activate above threshold."""
        if self.neuron_activity.sum() == 0:
            return torch.zeros(self.latent_dim, dtype=torch.bool, device=self.neuron_activity.device)
        return self.neuron_activity < min_activity

    @torch.no_grad()
    def reset_dead_neurons(self, init_method: str = "kaiming_uniform") -> int:
        """Reinitialise dead neurons and reset their activity counters."""
        dead_mask = self.get_dead_neurons()
        dead_count = dead_mask.sum().item()
        if dead_count == 0:
            return 0

        init_method = init_method.lower()
        if init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.encoder.weight[dead_mask], a=math.sqrt(5))
            if not self.tied_weights:
                nn.init.kaiming_uniform_(self.decoder.weight[:, dead_mask], a=math.sqrt(5))
        elif init_method == "kaiming_normal":
            nn.init.kaiming_normal_(self.encoder.weight[dead_mask], a=math.sqrt(5))
            if not self.tied_weights:
                nn.init.kaiming_normal_(self.decoder.weight[:, dead_mask], a=math.sqrt(5))
        elif init_method == "xavier_uniform":
            nn.init.xavier_uniform_(self.encoder.weight[dead_mask])
            if not self.tied_weights:
                nn.init.xavier_uniform_(self.decoder.weight[:, dead_mask])
        elif init_method == "xavier_normal":
            nn.init.xavier_normal_(self.encoder.weight[dead_mask])
            if not self.tied_weights:
                nn.init.xavier_normal_(self.decoder.weight[:, dead_mask])
        else:
            raise ValueError(f"Unknown init_method '{init_method}'")

        nn.init.zeros_(self.encoder.bias[dead_mask])
        self.neuron_activity[dead_mask] = 0
        return dead_count

    @staticmethod
    def l0_sparsity(z: torch.Tensor) -> float:
        """Average number of active units (L0 sparsity) across the batch."""
        return (z > 0).float().sum(dim=1).mean().item()

    @staticmethod
    def l1_sparsity(z: torch.Tensor) -> float:
        """Average L1 sparsity across the batch."""
        return z.abs().sum(dim=1).mean().item()
