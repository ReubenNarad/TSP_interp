from __future__ import annotations

from typing import Optional

import torch

from policy.matnet_custom import MatNetPolicyCustom, MatNetInitMode


class HookedMatNetPolicy(MatNetPolicyCustom):
    """MatNetPolicyCustom with forward hooks for capturing per-layer embeddings.

    Caches:
      - encoder_layer_{k}:      row embeddings after layer k, shape [B,n,d]
      - encoder_layer_{k}_col:  col embeddings after layer k, shape [B,n,d]
      - encoder_output:         final row embeddings, shape [B,n,d] (decoder-side embeddings)
      - encoder_output_col:     final col embeddings, shape [B,n,d]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooks: dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.activation_cache: dict[str, torch.Tensor] = {}
        self._setup_hooks()

    def _setup_hooks(self) -> None:
        # Encoder layers output (row_emb, col_emb)
        for layer_idx, layer in enumerate(self.encoder.layers):
            def make_hook(idx: int):
                def hook(_module, _inp, out):
                    if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0]) and torch.is_tensor(out[1]):
                        row, col = out
                        self.activation_cache[f"encoder_layer_{idx}"] = row
                        self.activation_cache[f"encoder_layer_{idx}_col"] = col
                return hook

            self.hooks[f"encoder_layer_{layer_idx}"] = layer.register_forward_hook(make_hook(layer_idx))

        # Encoder output: encoder returns (embeddings_tuple, init_embedding)
        def encoder_output_hook(_module, _inp, out):
            if isinstance(out, tuple) and len(out) >= 1:
                emb = out[0]
                if isinstance(emb, tuple) and len(emb) == 2 and torch.is_tensor(emb[0]) and torch.is_tensor(emb[1]):
                    row, col = emb
                    self.activation_cache["encoder_output"] = row
                    self.activation_cache["encoder_output_col"] = col

        self.hooks["encoder_output"] = self.encoder.register_forward_hook(encoder_output_hook)

    def clear_hooks(self) -> None:
        for handle in self.hooks.values():
            try:
                handle.remove()
            except Exception:
                pass
        self.hooks.clear()

    def clear_cache(self) -> None:
        self.activation_cache.clear()

    def get_activation(self, name: str) -> Optional[torch.Tensor]:
        return self.activation_cache.get(name)

    def forward(self, *args, **kwargs):
        self.clear_cache()
        return super().forward(*args, **kwargs)


class EnhancedHookedMatNetPolicy(HookedMatNetPolicy):
    """Alias for parity with AttentionModel collector naming."""

    def __init__(
        self,
        *,
        env_name: str = "atsp",
        embed_dim: int = 256,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "instance",
        use_graph_context: bool = False,
        bias: bool = False,
        init_embedding_mode: MatNetInitMode = "random_onehot",
        **kwargs,
    ):
        super().__init__(
            env_name=env_name,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            use_graph_context=use_graph_context,
            bias=bias,
            init_embedding_mode=init_embedding_mode,
            **kwargs,
        )

