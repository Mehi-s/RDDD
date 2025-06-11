import torch
import torch.nn as nn
import os
import time
from tools import mutils

saved_grad = None
saved_name = None

base_url = './results'
os.makedirs(base_url, exist_ok=True)


def normalize_tensor_mm(tensor):
  """Normalizes a tensor to the range [0, 1] using min-max normalization.

  Args:
    tensor: The input tensor.

  Returns:
    The normalized tensor.
  """
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def normalize_tensor_sigmoid(tensor):
  """Normalizes a tensor using the sigmoid function.

  Args:
    tensor: The input tensor.

  Returns:
    The normalized tensor.
  """
    return nn.functional.sigmoid(tensor)


def save_image(tensor, name=None, save_path=None, exit_flag=False, timestamp=False, nrow=4, split_dir=None):
  """Saves a tensor as an image.

  Args:
    tensor: The input tensor (usually an image or a batch of images).
    name: The base name for the saved image file.
    save_path: The full path to save the image. If None, a path is generated
               based on `base_url`, `split_dir`, and `name`.
    exit_flag: If True, exits the program after saving the image.
    timestamp: If True, appends a timestamp to the filename.
    nrow: Number of images to display in each row of the grid.
    split_dir: Optional subdirectory within `base_url` to save the image.
  """
    if split_dir:
        _base_url = os.path.join(base_url, split_dir)
    else:
        _base_url = base_url
    os.makedirs(_base_url, exist_ok=True)
    import torchvision.utils as vutils
    grid = vutils.make_grid(tensor.detach().cpu(), nrow=nrow)

    if save_path:
        vutils.save_image(grid, save_path)
    else:
        if timestamp:
            vutils.save_image(grid, f'{_base_url}/{name}_{mutils.get_timestamp()}.png')
        else:
            vutils.save_image(grid, f'{_base_url}/{name}.png')
    if exit_flag:
        exit(0)


def save_feature(tensor, name, exit_flag=False, timestamp=False):
  """Saves a feature tensor as an image.

  This function is intended for visualizing feature maps. Currently, it only
  saves the original tensor without applying min-max or sigmoid normalization,
  although there is commented-out code suggesting that was a possibility.

  Args:
    tensor: The input feature tensor.
    name: The base name for the saved image file.
    exit_flag: If True, exits the program after saving the image.
    timestamp: If True, appends a timestamp to the filename.
  """
    import torchvision.utils as vutils
    # tensors = [tensor, normalize_tensor_mm(tensor), normalize_tensor_sigmoid(tensor)]
    tensors = [tensor]
    titles = ['original', 'min-max', 'sigmoid']
    if timestamp:
        name += '_' + str(time.time()).replace('.', '')

    for index, tensor in enumerate(tensors):
        _data = tensor.detach().cpu().squeeze(0).unsqueeze(1)
        num_per_row = 4
        if _data.shape[0] / 4 > 4:
            num_per_row = int(_data.shape[0] / 4)
        num_per_row = 8
        grid = vutils.make_grid(_data, nrow=num_per_row)
        vutils.save_image(grid, f'{base_url}/{name}_{titles[index]}.png')
        print(f'{base_url}/{name}_{titles[index]}.png')
    if exit_flag:
        exit(0)


def save(tensor, name, exit_flag=False):
  """Saves a tensor (typically a single feature map or image) as an image.

  Args:
    tensor: The input tensor.
    name: The base name for the saved image file.
    exit_flag: If True, exits the program after saving the image.
  """
    import torchvision.utils as vutils
    grid = vutils.make_grid(tensor.detach().cpu().squeeze(0).unsqueeze(1), nrow=4)
    # grid = (grid - grid.min()) / (grid.max() - grid.min())
    # print(grid)
    vutils.save_image(grid, f'{base_url}/{name}.png')
    if exit_flag:
        exit(0)


def save_grid_direct(grad, name):
  """Saves a gradient tensor as an image and displays its histogram.

  This function reshapes the gradient, scales it, clamps it, and saves it.
  It also prints statistics about the gradient and displays a histogram.

  Args:
    grad: The input gradient tensor.
    name: The base name for the saved image file and plot title.
  """
    grad = grad.view(1, 8, 320, 320) * 255 / (320 * 320)
    # grad = grad.view(grad.shape[0],grad)
    save(grad.clamp(0, 255), name)

    module_grad = grad.clamp(-200, 200)
    print(module_grad.min().item(), module_grad.max().item(), module_grad.mean().item())
    module_grad_flat = module_grad.flatten()
    print(name, len(module_grad_flat[module_grad_flat < 0]) / len(module_grad_flat),
          len(module_grad_flat[module_grad_flat < 0]), len(module_grad_flat[module_grad_flat == 0]))
    import matplotlib.pyplot as plt
    import numpy as np
    y, x = np.histogram(module_grad.cpu().flatten().numpy(), bins=50, density=True)
    # plt.hist(module_grad.cpu().flatten().numpy(), 50)
    # for a, b in zip(x[:-1], y):
    #     print(a, b)
    # print(x)
    # print(y)
    plt.bar(x[:-1], y)
    # print('hist', hist)
    # print(module_grad.cpu().flatten().numpy())
    plt.show()


def save_grid(grad, name, exit_flag=False):
  """Saves a gradient tensor as an image, potentially normalized by a previous gradient.

  If `saved_grad` is None, this function stores the current gradient and name.
  Otherwise, it normalizes the current gradient by the `saved_grad`,
  saves the normalized gradient as an image, prints statistics, and displays a histogram.

  Args:
    grad: The input gradient tensor.
    name: The base name for the saved image file and plot title.
    exit_flag: If True, exits the program after saving/displaying.
  """
    global saved_grad, saved_name
    print(grad.shape)
    if saved_grad is None:
        print(name)
        saved_grad = grad
        saved_name = name
    else:
        # grad_flat = grad.flatten()
        # print(name, len(grad_flat[grad_flat < 0]) / len(grad_flat))

        module_grad = grad / (saved_grad + 1e-7)
        print(module_grad.max())
        save(module_grad.clamp(0, 255) / 255., name)

        module_grad = module_grad.clamp(-300, 300)
        print(module_grad.min().item(), module_grad.max().item(), module_grad.mean().item())
        module_grad_flat = module_grad.flatten()
        print(name, len(module_grad_flat[module_grad_flat < 0]) / len(module_grad_flat),
              len(module_grad_flat[module_grad_flat < 0]), len(module_grad_flat[module_grad_flat == 0]))
        import matplotlib.pyplot as plt
        import numpy as np
        y, x = np.histogram(module_grad.cpu().flatten().numpy(), bins=50, density=True)
        # plt.hist(module_grad.cpu().flatten().numpy(), 50)
        # for a, b in zip(x[:-1], y):
        #     print(a, b)
        # print(x)
        # print(y)
        plt.bar(x[:-1], y)
        # print('hist', hist)
        # print(module_grad.cpu().flatten().numpy())
        plt.show()
        exit(0)
    # print(len(grad))
    # print(grad)
    # print(grad[0].shape)
    # grad = grad[0]
    #
    # grad_flat = grad.flatten()
    # print('--------***')
    # print('--------***')
    # print('--------***')
    # print(name, len(grad_flat[grad_flat < 0]) / len(grad_flat))
    # print('--------***')
    # print('--------***')
    # print('--------***')

    # import torchvision.transforms as vtrans
    # import matplotlib.pyplot as plt
    # plt.hist()
    # plt.imshow(vtrans.ToPILImage()(grid))
    # plt.title(name + ' grad')
    # plt.show()

    #
    # if name in ['HE', 'CE Module', 'SOFT']:
    #     if saved_grad is None:
    #         saved_grad = grad
    #         saved_name = name
    #     else:
    #         grad = grad.reshape_as(saved_grad)
    #         print((saved_grad - grad).mean())
    #         print('diff: ', (saved_grad - grad).abs().max().item())
    #         print('mean: ', name, grad.mean().item(), saved_name, saved_grad.mean().item())
    #
    #         saved_grad = grad
    #         saved_name = name
    if exit_flag:
        exit(0)


def show_grid(grid, name, exit_flag=False):
  """Displays a grid of images using matplotlib.

  The input tensor is normalized to [0, 1] before display.

  Args:
    grid: The input tensor, typically a batch of feature maps or images.
    name: The title for the plot.
    exit_flag: If True, exits the program after displaying the image.
  """
    import torchvision.utils as vutils
    import torchvision.transforms as vtrans
    import matplotlib.pyplot as plt

    grid = (grid - grid.min()) / (grid.max() - grid.min())
    grid = vutils.make_grid(grid.cpu().squeeze(0).unsqueeze(1), nrow=4)

    # name = unique.get_unique(name)
    plt.imshow(vtrans.ToPILImage()(grid))
    plt.title(name)
    plt.show()
    # vutils.save_image(grid, f'/home/huqiming/workspace/Pytorch_Retinaface/results/{name}.png')
    if exit_flag:
        exit(0)


def show_img(img, name, exit_flag=False):
  """Displays a single image using matplotlib.

  Args:
    img: The input image tensor.
    name: The title for the plot.
    exit_flag: If True, exits the program after displaying the image.
  """
    import torchvision.utils as vutils
    import torchvision.transforms as vtrans
    import matplotlib.pyplot as plt

    grid = vutils.make_grid(img.cpu().squeeze(0))

    # name = unique.get_unique(name)
    plt.imshow(vtrans.ToPILImage()(grid))
    plt.title(name)
    plt.show()
    # vutils.save_image(grid, f'/home/huqiming/workspace/Pytorch_Retinaface/results/{name}.png')
    if exit_flag:
        exit(0)


class SaverBlock(nn.Module):
  """A simple nn.Module that saves the input feature tensor during the forward pass.

  This can be inserted into a model to inspect intermediate features.
  """
    def __init__(self):
        super(SaverBlock, self).__init__()

    def forward(self, x):
      """Saves the first element of the input tensor and returns the input tensor unchanged.

      Args:
        x: The input tensor (expected to be a list or tuple if only the first element is saved).

      Returns:
        The input tensor `x`.
      """
        save_feature(x[0], 'intermediate_', timestamp=True)
        return x
