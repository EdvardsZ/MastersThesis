from models.decoders import Decoder, LabelConditionedDecoder, PixelConditionedDecoder
import torch

def test_conventional_decoder(tab_count=1):
    tab = "\t" * tab_count

    print(tab + "===============")
    print(tab + "Testing with shape (128, 2)")

    batch_size = 128
    latent_dim = 2

    z = torch.rand(batch_size, latent_dim)
    decoder = Decoder(latent_dim=latent_dim)

    x_hat = decoder(z)

    print(tab + "x_hat shape: ", x_hat.shape)
    print(tab + "---------------")

    assert x_hat.shape == (batch_size, 1, 28, 28)

    print(tab + "Test passed")

    print(tab + "===============")

    return

def test_label_conditioned_decoder(tab_count=1):
    tab = "\t" * tab_count

    print(tab + "===============")
    print(tab + "Testing with shape (128, 2) and (128, 1)")

    batch_size = 128
    latent_dim = 2

    z = torch.rand(batch_size, latent_dim)
    # int label
    label = torch.rand(batch_size, 1).int()

    decoder = LabelConditionedDecoder(latent_dim=latent_dim)

    x_hat = decoder(z, label)

    print(tab + "x_hat shape: ", x_hat.shape)
    print(tab + "---------------")

    assert x_hat.shape == (batch_size, 1, 28, 28)

    print(tab + "Test passed")

    print(tab + "===============")


def test_pixel_conditioned_decoder(tab_count=1):
    tab = "\t" * tab_count

    print(tab + "===============")
    print(tab + "Testing with shape (128, 2) and (128, 1, 28, 28)")

    batch_size = 128
    latent_dim = 2

    z = torch.rand(batch_size, latent_dim)

    # image
    x = torch.rand(batch_size, 1, 28, 28)

    decoder = PixelConditionedDecoder(latent_dim=latent_dim)

    x_hat = decoder(z, x)

    print(tab + "x_hat shape: ", x_hat.shape)
    print(tab + "---------------")

    assert x_hat.shape == (batch_size, 1, 28, 28)

    print(tab + "Test passed")

    print(tab + "===============")

    return










    