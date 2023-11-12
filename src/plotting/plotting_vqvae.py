from matplotlib import pyplot as plt

def plot_samples_with_reconstruction_and_indices(model, data_loader, n=6, save_name=None):
    # plot n images and their reconstruction
    x, x_cond, y = next(iter(data_loader))
    model.eval()
    image_shape = model.model.image_shape

    output = model(x, x_cond, y)
    print(output[0].shape)
    indices = output[3].reshape(-1, 1, image_shape[1] // 4, image_shape[2] // 4)

    #make the plot smaller
    plt.figure(figsize=(n, 3))

    for i in range(n):
        image = x[i].detach().cpu().reshape(image_shape).permute(1, 2, 0)
        indice_image = indices[i][0].detach().cpu().reshape((image_shape[1] // 4, image_shape[2] // 4))
        reconstruction = output[0][i].detach().cpu().reshape(image_shape).permute(1, 2, 0)

        # axis off
        plt.subplot(3, n, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(3, n, i + 1 + n)
        plt.imshow(indice_image)
        plt.axis('off')
        plt.subplot(3, n, i + 1 + 2 *n)
        plt.imshow(reconstruction)
        plt.axis('off')

    if save_name is not None:
        plt.savefig("assets/reconstructions/" + save_name + ".png")


    plt.show()