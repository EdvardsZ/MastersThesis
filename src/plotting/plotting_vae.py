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

    images_to_plot_titles = ["Original", "Reconstruction", "Reconstruction(Conditioned)"]

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