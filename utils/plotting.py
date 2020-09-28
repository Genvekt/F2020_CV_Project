import matplotlib.pyplot as plt

def showInRow(list_of_images, titles = None, disable_ticks = False, vertical=False):
    count = len(list_of_images)
    plt.figure(figsize=(15,10))
    for idx in range(count):
        if vertical:
            subplot = plt.subplot(count, 1, idx+1)
        else:
            subplot = plt.subplot(1, count, idx+1)
        if titles is not None:
            subplot.set_title(titles[idx])
      
        img = list_of_images[idx]
        cmap = 'gray' if (len(img.shape) == 2 or img.shape[2] == 1) else None
        subplot.imshow(img, cmap=cmap)
        if disable_ticks:
            plt.xticks([]), plt.yticks([])
    plt.show()