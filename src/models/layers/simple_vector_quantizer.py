import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(SimpleVectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        w_init = torch.nn.init.uniform_
        self.embeddings = nn.Parameter(w_init(torch.empty(self.embedding_dim, self.num_embeddings)))
        
    def forward(self, x):
        # x.shape = (B, E, H, W)
        input_shape = x.shape

        flattened = x.view(-1, self.embedding_dim)
        # flattened.shape = (batch_size * height * width, embedding_dim)

        # Calculate distances between embedding vectors and input vectors and get the indices of the minimum distances.
        encoding_indices = self.get_code_indices(flattened)
        # embedding_indices.shape = (batch_size * height * width)
        
        encodings = F.one_hot(encoding_indices, num_classes=self.num_embeddings).to(flattened.device)
        # encodings.shape = (batch_size * height * width, num_embeddings)

        quantized = torch.matmul(encodings.float(), self.embeddings.t())
        # quantized.shape = (batch_size * height * width, embedding_dim)

        quantized = quantized.view(input_shape) # Take this for the loss
        # quantized.shape = (batch_size, embedding, height, width)

        quantized_with_grad = x + (quantized - x).detach()  # This part copies the gradient

        return quantized_with_grad, quantized, encoding_indices

    def get_code_indices(self, flattened_inputs):
        similarity = torch.matmul(flattened_inputs, self.embeddings)
        distances = (torch.sum(flattened_inputs ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings ** 2, dim=0)
                     - 2 * similarity)

        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices
    
    def quantize_from_indices(self, indices, batch_size):
        # indices.shape = (B * H_f * W_f)
        # 1. Index from the "codebook"
        # -----------------------
        encodings = F.one_hot(indices, num_classes=self.num_embeddings).to(indices.device).float()
        # encodings.shape = (B * H_f * W_f, num_embeddings)
        quantized = torch.matmul(encodings.float(), self.embeddings.t())
        # quantized.shape = (B, E, H_f, W_f, E)
        # -----------------------
        # 2. Reshape back
        # -----------------------
        quantized = quantized.reshape((batch_size, self.embedding_dim, 7, 7))
        # quantized.shape = (B, E, H_f, W_f)
        # -----------------------

        return quantized