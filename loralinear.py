from torch.nn import functional as F
from diffusers.models.attention_processor import FluxAttnProcessor2_0
import torch
from typing import Optional, Union
from diffusers.models.attention_processor import Attention
import torch.nn as nn

    
class MoELoRALinearLayer(nn.Module):
    r"""
    A linear layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,

    ):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor, 
                top_k_values: Optional[torch.Tensor] = None,
                top_k_indices: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype


        down_hidden_states = self.down(hidden_states.to(dtype))
        # =======================================
        result = torch.zeros_like(down_hidden_states)
        rows = torch.arange(down_hidden_states.size(0)).unsqueeze(1)  # shape (3, 1)
        original_values = down_hidden_states[rows, top_k_indices]
        multiplied_values = original_values * top_k_values
        result[rows, top_k_indices] = multiplied_values
        # =======================================
        up_hidden_states = self.up(result)

        return up_hidden_states.to(orig_dtype)
