import math
import torch
import torch.nn as nn
from USWAttn import (USWAttention, PackAndUnpackAttention)

class LittleBird(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int = 8,
        d_ff: int = 2048,
        dropout_p: float = 0.1,
        project_embedding_length: int = 32, #이게 뭔지 모르겠음
        max_length: int = 2048,
        context_info_length = 
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor(0.1))
        self.beta = nn.Parameter(torch.Tensor(0.1))
        self.gamma = nn.Parameter(torch.Tensor(0.1))
        self.dp_matrix = nn.Parameter(torch.ones())
        self.d_matrix = nn.Parameter(torch.Tensor())