import torch 
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embeddings = nn.Linear(embedding_dim, num_embeddings, bias=False) 
        # This is basically a matrix num_embeddings x embedding_dim

    def forward(self, x):
        # x.shape = (B, E, H_f, W_f)
        input_shape = x.shape

        # 1. Reshape
        # -----------------------
        flattened = x.reshape(-1, self.embedding_dim)
        # flattened.shape = (B * H_f * W_f, E)
        # -----------------------

        # 2. Calculate distances and get the indices of the minimum distances.
        # -----------------------
        indices = self.get_code_indices(flattened)
        # indices.shape = (B * H_f * W_f)
        # -----------------------

        # 3. Index from the "codebook"
        # -----------------------
        encodings = F.one_hot(indices, num_classes=self.num_embeddings).to(flattened.device).float()
        # encodings.shape = (B * H_f * W_f, num_embeddings)
        quantized = torch.matmul(encodings.float(), self.embeddings.weight)
        # quantized.shape = (B * H_f * W_f, E)
        # -----------------------

        # 4. Reshape back
        # -----------------------
        quantized = quantized.reshape(input_shape)
        # quantized.shape = (B, E, H_f, W_f)
        # -----------------------

        # 5. Copy the gradient
        # -----------------------
        quantized_with_grad = x + (quantized - x).detach()
        # quantized_with_grad.shape = (B, E, H_f, W_f)
        # -----------------------
        
        return quantized_with_grad, quantized, indices
    
    def quantize_from_indices(self, indices):

        # 1. Index from the "codebook"
        # -----------------------
        encodings = F.one_hot(indices, num_classes=self.num_embeddings).to(indices.device).float()
        # encodings.shape = (B * H_f * W_f, num_embeddings)
        quantized = torch.matmul(encodings.float(), self.embeddings.weight)
        # quantized.shape = (B, E, H_f, W_f, E)
        # -----------------------
        # 2. Reshape back
        # -----------------------
        quantized = quantized.reshape((100, 64, 7, 7))
        # quantized.shape = (B, E, H_f, W_f)
        # -----------------------

        return quantized
    
    def get_code_indices(self, flattened):
        # a^2
        flattened_squared = flattened.pow(2).sum(dim=1, keepdim=True).sum(dim=1, keepdim=True)

        # b^2
        embedding_squared = self.embeddings.weight.pow(2).sum(dim=1)

        # 2ab
        product = (self.embeddings(flattened) * 2)

        # a^2 + b^2 - 2ab
        distances = flattened_squared + embedding_squared - product

        # argmin
        encoding_indices = torch.argmin(distances, dim=1)

        return encoding_indices

    




    