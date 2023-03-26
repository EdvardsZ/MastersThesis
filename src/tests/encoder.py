from models.encoders import Encoder
import torch


def test_conventional_encoder():
    print("Testing Conventional Encoder")
    print("===============")
    print("Testing with shape (128, 1, 28, 28)")

    batch_size = 128
    image_size= (1, 28, 28)

    image = torch.rand(batch_size, image_size[0], image_size[1], image_size[2])
    encoder = Encoder(latent_dim=2, image_size=image_size)

    z_mean, z_log_var, z = encoder(image)
    print("z_mean shape: ", z_mean.shape)
    print("z_log_var shape: ", z_log_var.shape)
    print("z shape: ", z.shape)
    print("===============")
    assert z_mean.shape == (batch_size, 2)
    assert z_log_var.shape == (batch_size, 2)
    assert z.shape == (batch_size, 2)
    return