import torch
from torch import nn
from torch.nn import functional as F

from .distributed import all_reduce


class NewVectorQuantizer(nn.Module):
    def __init__(self, n_embed, dim, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        # input.shape = [B, E, H, W]
        input_reshaped = input.permute(0, 2, 3, 1)
        # input.shape = [B, H, W, E]
        flatten = input_reshaped.reshape(-1, self.dim)
        # flatten.shape = [B * H * W, E]
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        # dist.shape = [B * H * W, NUM_Embeddings]
        _, embed_ind = (-dist).max(1)
        # embed_ind.shape = [B * H * W]
        embed_ind = embed_ind.view(*input_reshaped.shape[:-1])
        # embed_ind.shape = [B, H ,W]
        quantize = self.embed_code(embed_ind)
        # quantize.shape = [B, H, W, E]
        quantize_with_grad = input_reshaped + (quantize - input_reshaped).detach()
        quantize_with_grad = quantize_with_grad.permute(0, 3, 1, 2)
        # quantize_with_grad.shape = [B, E, H, W]

        quantize = quantize.permute(0, 3, 1, 2)
        # quantize.shape = = [B, E, H, W]
        return quantize_with_grad, quantize, embed_ind
    
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
    
    def quantize_from_indices(self, indices, batch_size):
        # input.shape = [B, H ,W]
        quantize = self.embed_code(indices)
        # quantize.shape = embed_ind.shape = [B, H, W, E]
        quantize = quantize.permute(0, 3, 1, 2)

        return quantize