import torch
import torch.nn as nn
import torch.nn.functional as F

class NewVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(NewVectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        embed = torch.randn(embedding_dim, num_embeddings)
        self.register_buffer("embed", embed)

    def forward(self, input):
        # input.shape = [B, E, H, W]
        input_reshaped = input.permute(0, 2, 3, 1)
        # input.shape = [B, H, W, E]
        flatten = input_reshaped.reshape(-1, self.embedding_dim)
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