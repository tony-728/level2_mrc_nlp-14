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


    def get_d_matrix(max_length, alpha, beta, gamma):
        d_matrix = np.zeros((max_length, max_length))
        for i in range(max_length):
            for j in range(max_length):
                if i==j:
                    d_matrix[i,j] = 0
                elif i>j:
                    d_matrix[i,j] = beta * (i-j)
                else:
                    d_matrix[i,j] = gamma * (j-i)
        d_matrix[0,1:] = alpha
        d_matrix[1:, 0] = alpha

        return torch.Tensor(d_matrix)
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
