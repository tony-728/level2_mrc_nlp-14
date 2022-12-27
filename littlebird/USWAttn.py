import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class PackAndUnpackAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Attn = nn.MultiheadAttention(config.embed_dim, config.num_heads)
        self.LayerNorm = nn.LayerNorm()




class USWAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0, "hidden_dim % num_attention_heads must be zero."
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.softmax = nn.Softmax()

        self.dp_matrix = config.dp
        self.d_matrix = config.d_matrix
    def forward(
        self,
        x,
        cp
    ):
        dp = self.config.dp
        d = self.config.d
        j = self.config.j #(s,l) only one matrix
        q_x = #input sequence query 
        k_x = #input sequence key 
        v_x = #input sequence value 
        k_cp = 

"""
class MultiHeadAttention(nn.Module):
    def __init__(self, config)->None:
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0, "hidden_dim % num_attention_heads must be zero."
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size)

    def forward(
        self,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        batch_size = value.size(0)
        query = self.query_proj(query).view(batch_size, -1)
"""