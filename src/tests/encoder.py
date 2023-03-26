from models.encoders import Encoder
import torch



def test_conventional_encoder(tab_count=1):

    tab = "\t" * tab_count
    print(tab + "===============")
    print(tab + "Testing with shape (128, 1, 28, 28)")

    batch_size = 128
    image_size= (1, 28, 28)

    image = torch.rand(batch_size, image_size[0], image_size[1], image_size[2])
    encoder = Encoder(latent_dim=2, image_size=image_size)

    z_mean, z_log_var, z = encoder(image)
    print(tab + "z_mean shape: ", z_mean.shape)
    print(tab + "z_log_var shape: ", z_log_var.shape)
    print(tab + "z shape: ", z.shape)
    print(tab + "---------------")

    assert z_mean.shape == (batch_size, 2)
    assert z_log_var.shape == (batch_size, 2)
    assert z.shape == (batch_size, 2)

    print(tab + "Test passed")
    print(tab + "===============")
    print(tab + "Testing with shape (128, 3, 220, 220)")

    batch_size = 128
    image_size= (3, 220, 220)

    image = torch.rand(batch_size, image_size[0], image_size[1], image_size[2])
    encoder = Encoder(latent_dim=2, image_size=image_size)

    z_mean, z_log_var, z = encoder(image)

    print(tab + "z_mean shape: ", z_mean.shape)
    print(tab + "z_log_var shape: ", z_log_var.shape)
    print(tab + "z shape: ", z.shape)
    print(tab + "---------------")

    assert z_mean.shape == (batch_size, 2)
    assert z_log_var.shape == (batch_size, 2)
    assert z.shape == (batch_size, 2)

    print(tab + "Test passed")
    print(tab + "===============")

    return