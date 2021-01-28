import os
import skimage.io
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    image = skimage.io.imread(path)
    return image


def load_mask(path):
    mask = skimage.io.imread(path, as_gray=True)
    return mask


def save_output(image, mask, directory, number=0):
    # create paths for output image and mask
    output = os.path.join(directory, f'output_{str(number).zfill(2)}.png')
    output_mask = os.path.join(directory, f'output_{str(number).zfill(2)}_mask.png')

    # save to directory
    skimage.io.imsave(output, image, check_contrast=False)
    skimage.io.imsave(output_mask, mask, check_contrast=False)


def show_image(image, mask):
    titles = ['Image', 'Mask']
    images = [image, mask]
    plt.figure(figsize=(5.1, 2))

    for i, (name, image) in enumerate(zip(titles, images)):
        plt.subplot(1, len(images), i + 1)
        plt.title(name)
        plt.imshow(image)

    plt.show()


def show_pair(target, mask, background, background_mask):
    titles = ['Target', 'Target Mask', 'Background', 'Background Mask']
    images = [target, mask, background, background_mask]
    plt.figure(figsize=(10, 2))

    for i, (name, image) in enumerate(zip(titles, images)):
        plt.subplot(1, len(images), i + 1)
        plt.title(name)
        plt.imshow(image)

    plt.show()


def show_cutout(image, cutout, mask):
    titles = ['Image', 'Cutout', 'Mask']
    images = [image, cutout, mask]
    plt.figure(figsize=(7.4, 2))

    for i, (name, image) in enumerate(zip(titles, images)):
        plt.subplot(1, len(images), i + 1)
        plt.title(name)
        plt.imshow(image)

    plt.show()


def is_background(image, threshold=192, min_ratio=0.75):
    assert image.shape[-1] == 3
    values = np.mean(image, axis=-1).flatten()
    ratio = np.sum((values < threshold)) / values.shape[0]
    return 1 if ratio >= min_ratio else 0
