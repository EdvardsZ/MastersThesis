from typing import List
from matplotlib import pyplot as plt

def plot_sample_images(images_to_plot: List, images_to_plot_titles: List[str], save_name : str | None = None, n = 6):
    # expects list of tensors of shape (n, c, h, w)
    # expects list of titles of length n
    # expects save_name to be a string or None
    # expects n to be an int

    fig = plt.figure(figsize=(n, len(images_to_plot)))
    #fig.suptitle('Figure title')

    subfigs = fig.subfigures(nrows=len(images_to_plot), ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(images_to_plot_titles[row])

        axs = subfig.subplots(nrows=1, ncols=n)
        for col, ax in enumerate(axs):
            # plot images
            ax.imshow(images_to_plot[row][col].detach().cpu().permute(1, 2, 0))
            ax.axis('off')
            #ax.set_title(f'Plot title {col}')

    if save_name is not None:
        plt.savefig("assets/reconstructions/" + save_name + ".png")

    plt.show()