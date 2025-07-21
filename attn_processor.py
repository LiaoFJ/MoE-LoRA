from torch.nn import functional as F
from diffusers.models.attention_processor import FluxAttnProcessor2_0
import torch
from typing import Optional, Union
from diffusers.models.attention_processor import Attention
import torch.nn as nn
from loralinear import MoELoRALinearLayer

class MoEFluxAttnProcessor2_0(nn.Module, FluxAttnProcessor2_0):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self,
                num_attetnion_heads: int = 24,
                attention_head_dim: int = 128,
                num_experts: int = 4,
                expert_lora_rank: int = 8,
                topk: int = 2,
                singlelayer: bool = False,
                **kwargs
                 ):
        
        super().__init__()
        self.to_q_moelora = MoELoRALinearLayer(
            in_features=num_attetnion_heads * attention_head_dim,
            out_features=num_attetnion_heads * attention_head_dim,
            rank=expert_lora_rank,
        )
        self.to_k_moelora = MoELoRALinearLayer(
            in_features=num_attetnion_heads * attention_head_dim,
            out_features=num_attetnion_heads * attention_head_dim,
            rank=expert_lora_rank,
        )
        self.to_v_moelora = MoELoRALinearLayer(
            in_features=num_attetnion_heads * attention_head_dim,
            out_features=num_attetnion_heads * attention_head_dim,
            rank=expert_lora_rank,
        )
        
        self.topkgating = TopKGating(num_attetnion_heads * attention_head_dim, num_experts, top_k=topk)

        if not singlelayer:
            self.to_q_proj_moelora = MoELoRALinearLayer(
                in_features=num_attetnion_heads * attention_head_dim,
                out_features=num_attetnion_heads * attention_head_dim,
                rank=expert_lora_rank,
            )
            self.to_k_proj_moelora = MoELoRALinearLayer(
                in_features=num_attetnion_heads * attention_head_dim,
                out_features=num_attetnion_heads * attention_head_dim,
                rank=expert_lora_rank,
            )
            self.to_v_proj_moelora = MoELoRALinearLayer(
                in_features=num_attetnion_heads * attention_head_dim,
                out_features=num_attetnion_heads * attention_head_dim,
                rank=expert_lora_rank,
            )
            self.topkgating_proj = TopKGating(num_attetnion_heads * attention_head_dim, num_experts, top_k=topk)

        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # ==================== for moe ====================
        indices, gates = self.topkgating(hidden_states)
        query_moelora = self.to_q_moelora(hidden_states, indices, gates)
        key_moelora = self.to_k_moelora(hidden_states, indices, gates)
        value_moelora = self.to_v_moelora(hidden_states, indices, gates)

        query += query_moelora
        key += key_moelora
        value += value_moelora
        # ==================== for moe ====================

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            # ==================== for moe ====================
            indices_proj, gates_proj = self.topkgating_proj(encoder_hidden_states)
            query_proj_moelora = self.to_q_proj_moelora(hidden_states, indices_proj, gates_proj)
            key_proj_moelora = self.to_k_proj_moelora(hidden_states, indices_proj, gates_proj)
            value_proj_moelora = self.to_v_pproj_moelora(hidden_states, indices_proj, gates_proj)

            encoder_hidden_states_query_proj += query_proj_moelora
            encoder_hidden_states_key_proj += key_proj_moelora
            encoder_hidden_states_value_proj += value_proj_moelora
            # ==================== for moe ====================


            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2):
        super(TopKGating, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        # get the scores of each expert
        gating_scores = self.gate(x)
        # choose top_k experts, return the idx and weights(softmax weights)
        top_k_values, top_k_indices = torch.topk(F.softmax(gating_scores, dim=1), self.top_k)
        return top_k_indices, top_k_values
