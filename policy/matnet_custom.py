from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.models.common.constructive.autoregressive import AutoregressivePolicy
from rl4co.models.zoo.matnet.decoder import MatNetDecoder
from rl4co.models.zoo.matnet.encoder import MatNetEncoder
from rl4co.models.nn.env_embeddings.init import MatNetInitEmbedding


MatNetInitMode = Literal["random_onehot", "random", "onehot"]


class DeterministicOneHotMatNetInitEmbedding(nn.Module):
    """Deterministic MatNet init embedding for ATSP.

    RL4CO's default MatNet init uses a per-instance random permutation of one-hot
    basis vectors for the column embeddings. For mechanistic analysis / SAE work,
    it is often preferable to remove that extra injected randomness.

    This embedding sets:
      - row_emb = 0
      - col_emb[b, j] = e_j  (requires num_nodes <= embed_dim)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = int(embed_dim)

    def forward(self, td: TensorDict):
        dmat = td["cost_matrix"]
        b, r, c = dmat.shape

        if c > self.embed_dim:
            raise ValueError(
                f"Deterministic one-hot init requires num_nodes <= embed_dim, got num_nodes={c}, embed_dim={self.embed_dim}. "
                "Use --init_embedding_mode random_onehot (or increase embed_dim)."
            )

        row_emb = torch.zeros(b, r, self.embed_dim, device=dmat.device, dtype=dmat.dtype)
        col_emb = torch.zeros(b, c, self.embed_dim, device=dmat.device, dtype=dmat.dtype)

        # col_emb[b, j, j] = 1
        j = torch.arange(c, device=dmat.device)
        col_emb[:, j, j] = 1.0
        return row_emb, col_emb, dmat


@dataclass(frozen=True)
class MatNetPolicyConfig:
    embed_dim: int = 256
    num_encoder_layers: int = 5
    num_heads: int = 16
    normalization: str = "instance"
    use_graph_context: bool = False
    bias: bool = False
    init_embedding_mode: MatNetInitMode = "random_onehot"


class MatNetPolicyCustom(AutoregressivePolicy):
    """MatNet policy with configurable initial embedding mode.

    This mirrors RL4CO's MatNetPolicy but allows selecting a deterministic
    one-hot init (useful when node ordering is already randomized upstream).
    """

    def __init__(
        self,
        env_name: str = "atsp",
        *,
        embed_dim: int = 256,
        num_encoder_layers: int = 5,
        num_heads: int = 16,
        normalization: str = "instance",
        use_graph_context: bool = False,
        bias: bool = False,
        init_embedding_mode: MatNetInitMode = "random_onehot",
        **kwargs,
    ):
        if env_name != "atsp":
            raise ValueError(f"MatNetPolicyCustom currently supports env_name='atsp' only, got {env_name!r}")

        if init_embedding_mode == "onehot":
            init_embedding = DeterministicOneHotMatNetInitEmbedding(embed_dim=embed_dim)
        elif init_embedding_mode == "random":
            init_embedding = MatNetInitEmbedding(embed_dim=embed_dim, mode="Random")
        elif init_embedding_mode == "random_onehot":
            init_embedding = MatNetInitEmbedding(embed_dim=embed_dim, mode="RandomOneHot")
        else:
            raise ValueError(
                f"Unknown init_embedding_mode={init_embedding_mode!r}; expected one of: random_onehot, random, onehot"
            )

        encoder = MatNetEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            normalization=normalization,
            init_embedding=init_embedding,
            bias=bias,
        )
        decoder = MatNetDecoder(
            env_name=env_name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_graph_context=use_graph_context,
        )

        super().__init__(
            env_name=env_name,
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            use_graph_context=use_graph_context,
            bias=bias,
            init_embedding_mode=init_embedding_mode,
            **kwargs,
        )

