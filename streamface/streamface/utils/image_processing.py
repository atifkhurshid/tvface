import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from PIL import Image


def imread(filepath):
    
    image = None
    with Image.open(filepath) as img:
        image = np.asarray(img)
    return image


def imwrite(filepath, image):

    Image.fromarray(image).save(filepath)


def imshow(image):

    io.imshow(image)
    plt.show()


def imresize(image, size, soft=False):

    image = Image.fromarray(image)
    if soft:
        image.thumbnail(size, resample=Image.LANCZOS)
    else:
        image = image.resize(size, resample=Image.LANCZOS)

    return np.asarray(image)


def center_crop(image, margin=0.1):

    H, W = image.shape[:2]

    dH = int(margin * H)
    dW = int(margin * W)

    image = image[dH:-dH, dW:-dW]

    return image
