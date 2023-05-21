import torch 
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embeddings = nn.Linear(embedding_dim, num_embeddings, bias=False)

    def forward(self, x):
        # x.shape = (batch_size, height, width, embedding_dim)

        # Flatten input.
        flattened = x.reshape(-1, self.embedding_dim)
        # flattened.shape = (batch_size * height * width, embedding_dim)

        # Calculate distances between embedding vectors and input vectors and get the indices of the minimum distances.
        embedding_indices = self.get_code_indices(flattened)
        # embedding_indices.shape = (batch_size * height * width, 1)

        # Convert the indices into one-hot vectors.
        codebook = nn.functional.one_hot(embedding_indices, self.num_embeddings).float()
        codebook = codebook.view(-1, self.num_embeddings)
        # codebook.shape = (batch_size * height * width, num_embeddings)

        # Calculate the quantized latent vectors.
        quantized = torch.matmul(codebook, self.embeddings.weight)
        # quantized.shape = (batch_size * height * width, embedding_dim)

        # Convert the quantized vectors back to the original shape.
        quantized = quantized.view(x.shape)
        # quantized.shape = (batch_size, height, width, embedding_dim)

        return quantized,  embedding_indices
    
    def get_code_indices(self, x):
        # Calculate L2-normalized distance between the inputs and the codes.
        # x.shape = (batch_size * height * width, embedding_dim)
        similarity = self.embeddings(x)
        distances = torch.sum(x ** 2, dim=1, keepdim=True) + torch.sum(self.embeddings.weight ** 2, dim=1) - 2 * similarity
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        return encoding_indices
    