from models.decoders import Decoder
import torch

def test_conventional_decoder():
    print("Testing Conventional Decoder")
    print("===============")
    print("Testing with shape (128, 2)")
    batch_size = 128
    latent_dim = 2
    z = torch.rand(batch_size, latent_dim)
    decoder = Decoder(latent_dim=latent_dim)
    x_hat = decoder(z)
    print("x_hat shape: ", x_hat.shape)
    print("===============")
    assert x_hat.shape == (batch_size, 1, 28, 28)
    print("Test passed")
    