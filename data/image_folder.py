###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def read_fns(filename):
  """Reads a list of filenames from a file.

  Args:
    filename: The path to the file.

  Returns:
    A list of filenames.
  """
    with open(filename) as f:
        fns = f.readlines()
        fns = [fn.strip() for fn in fns]
    return fns


def is_image_file(filename):
  """Checks if a file is an image file.

  Args:
    filename: The path to the file.

  Returns:
    True if the file is an image file, False otherwise.
  """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, fns=None):
  """Creates a dataset of image paths from a directory.

  Args:
    dir: The path to the directory.
    fns: A list of filenames. If None, all image files in the directory will be used.

  Returns:
    A list of image paths.
  """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    if fns is None:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):                
                    path = os.path.join(root, fname)
                    images.append(path)
    else:
        for fname in fns:
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                images.append(path)

    return images


def default_loader(path):
  """Loads an image from a path.

  Args:
    path: The path to the image.

  Returns:
    The loaded image.
  """
    return Image.open(path).convert('RGB')
