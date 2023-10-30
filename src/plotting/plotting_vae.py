import matplotlib.pyplot as plt
from numpy import imag

def plot_samples_with_reconstruction(model, data_loader, n=6, save_name=None):
    # plot n images and their reconstruction

    x, x_cond, y = next(iter(data_loader))
    model.eval()

    images_to_plot = []

    outputs, z, z_mean, z_log_var = model(x, x_cond, y)

    if isinstance(outputs, list):
        images_to_plot.append(x)
        images_to_plot.append(outputs[0])
        images_to_plot.append(outputs[1])
    else:
        images_to_plot.append(x)
        images_to_plot.append(outputs)


    fig, axes = plt.subplots(len(images_to_plot), n, figsize=(n, len(images_to_plot)))

    for i in range(n):
        for j in range(len(images_to_plot)):
            ax = axes[j, i]
            ax.imshow(images_to_plot[j][i].detach().cpu().numpy().reshape(28, 28))
            ax.axis('off')

    images_to_plot_titles = ["Original", "Reconstruction_1", "Reconstruction (Conditioned)"]

    #plot titles
    if isinstance(outputs, list):
        fig.suptitle("Original vs Reconstruction (Non conditioned) vs Reconstruction (Conditioned)")
    else:
        fig.suptitle("Original vs Reconstruction")



    fig = plt.figure(figsize=(n, len(images_to_plot)))
    #fig.suptitle('Figure title')

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=len(images_to_plot), ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(images_to_plot_titles[row])

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=n)
        for col, ax in enumerate(axs):
            # plot images
            ax.imshow(images_to_plot[row][col].detach().cpu().numpy().reshape(28, 28))
            ax.axis('off')
            #ax.set_title(f'Plot title {col}')
    


    # #if is instance of list the it it has 2 reconstructions
    # if isinstance(output[0], list):
    #     plt.figure(figsize=(n, 3))
    #     reconstruction_1 = output[0][0][i].detach().cpu().numpy().reshape(28, 28)
    #     reconstruction_2 = output[0][1][i].detach().cpu().numpy().reshape(28, 28)
    # else:
    #     plt.figure(figsize=(n, 2))

    # for i in range(n):
    #     image = data[0][i].detach().cpu().numpy().reshape(28, 28)
    #     if isinstance(output[0], list):
    #         reconstruction_1 = output[0][0][i].detach().cpu().numpy().reshape(28, 28)
    #         reconstruction_2 = output[0][1][i].detach().cpu().numpy().reshape(28, 28)
    #     else:
    #         reconstruction = output[0][i].detach().cpu().numpy().reshape(28, 28)

    #     if isinstance(output[0], list):
            
    #         plt.subplot(3, n, i + 1)
    #         plt.imshow(image)
    #         plt.axis('off')
    #         plt.subplot(3, n, i + 1 + n)
    #         plt.imshow(reconstruction_1)
    #         plt.axis('off')
    #         plt.subplot(3, n, i + 1 + 2 * n)
    #         plt.imshow(reconstruction_2)
    #         plt.axis('off')
    #     else:
    #         plt.subplot(2, n, i + 1)
    #         plt.imshow(image)
    #         plt.axis('off')
    #         plt.subplot(2, n, i + 1 + n)
    #         plt.imshow(reconstruction)
    #         plt.axis('off')

    # #plot titles
    # if isinstance(output[0], list):
    #     plt.suptitle("Original vs Reconstruction (Non conditioned) vs Reconstruction (Conditioned)")
    # else:
    #     plt.suptitle("Original vs Reconstruction")

    # if save_name is not None:
    #     plt.savefig("assets/reconstructions/" + save_name + ".png")

    # plt.show()