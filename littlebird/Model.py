import math
import torch
import torch.nn as nn
import numpy as np
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
        context_info_length = 64,
        block_size = 1
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor(0.1))
        self.beta = nn.Parameter(torch.Tensor(0.1))
        self.gamma = nn.Parameter(torch.Tensor(0.1))
        self.dp_matrix = nn.Parameter(torch.ones(context_info_length, max_length) * ((self.beta + self.gamma) * block_size) / 2)
        self.d_matrix = get_d_matrix(max_length, self.alpha, self.beta, self.gamma)

