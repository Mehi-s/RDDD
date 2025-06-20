from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity

def get_config(config):
  """Loads a YAML configuration file.

  Args:
    config (str): Path to the YAML configuration file.

  Returns:
    dict: A dictionary containing the configuration.
  """
    with open(config, 'r') as stream:
        return yaml.load(stream)


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
  """Converts a PyTorch tensor to a NumPy image array.

  The tensor is expected to be in the format (B, C, H, W) where B=1.
  The output NumPy array will be in the format (H, W, C) and scaled to [0, 255].
  Handles cases for 1, 3, 6, or 7 channel tensors.
  For 6 channels, it concatenates the first 3 and last 3 channels horizontally.
  For 7 channels, it concatenates the first 3, next 3, and the last channel (replicated to 3 channels) horizontally.

  Args:
    image_tensor (torch.Tensor): Input tensor.
    imtype (np.dtype, optional): Desired NumPy data type for the output image.
                                Defaults to np.uint8.

  Returns:
    np.ndarray: The converted image as a NumPy array.
  """
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(imtype)
    if image_numpy.shape[-1] == 6:
        image_numpy = np.concatenate([image_numpy[:, :, :3], image_numpy[:, :, 3:]], axis=1)
    if image_numpy.shape[-1] == 7:
        edge_map = np.tile(image_numpy[:, :, 6:7], (1, 1, 3))
        image_numpy = np.concatenate([image_numpy[:, :, :3], image_numpy[:, :, 3:6], edge_map], axis=1)
    return image_numpy


def tensor2numpy(image_tensor):
  """Converts a PyTorch tensor to a NumPy array, scaled to [0, 255].

  The tensor is expected to be in the format (C, H, W) or (1, C, H, W).
  The output NumPy array will be in the format (H, W, C).

  Args:
    image_tensor (torch.Tensor): Input tensor.

  Returns:
    np.ndarray: The converted array as a NumPy array of type np.float32.
  """
    image_numpy = torch.squeeze(image_tensor).cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.float32)
    return image_numpy


# Get model list for resume
def get_model_list(dirname, key, epoch=None):
  """Gets the path to a saved model checkpoint file.

  Args:
    dirname (str): The directory containing the checkpoint files.
    key (str): A keyword to identify the model (e.g., 'netG', 'netD').
    epoch (int, optional): The epoch number of the checkpoint. If None,
                           it returns the path to the latest checkpoint.
                           Defaults to None.

  Returns:
    str or None: The path to the checkpoint file, or None if the directory
                 or checkpoint doesn't exist.
  """
    if epoch is None:
        return os.path.join(dirname, key + '_latest.pt')
    if os.path.exists(dirname) is False:
        return None

    print(dirname, key)
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and ".pt" in f and 'latest' not in f]
    epoch_index = [int(os.path.basename(model_name).split('_')[-2]) for model_name in gen_models if
                   'latest' not in model_name]
    print('[i] available epoch list: %s' % epoch_index, gen_models)
    i = epoch_index.index(int(epoch))

    return gen_models[i]


def vgg_preprocess(batch):
  """Preprocesses a batch of images for VGG network input.

  Normalizes the images using ImageNet mean and standard deviation.
  Input tensor is assumed to be in the range [-1, 1].

  Args:
    batch (torch.Tensor): A batch of images (B, C, H, W).

  Returns:
    torch.Tensor: The preprocessed batch.
  """
    # normalize using imagenet mean and std
    mean = batch.new(batch.size())
    std = batch.new(batch.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = (batch + 1) / 2
    batch -= mean
    batch = batch / std
    return batch


def diagnose_network(net, name='network'):
  """Prints the mean absolute gradient of a network's parameters.

  Useful for diagnosing training issues like vanishing or exploding gradients.

  Args:
    net (torch.nn.Module): The network to diagnose.
    name (str, optional): The name of the network to display. Defaults to 'network'.
  """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
  """Saves a NumPy image array to a file.

  Args:
    image_numpy (np.ndarray): The image array to save.
    image_path (str): The path to save the image to.
  """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
  """Prints statistics of a NumPy array.

  Args:
    x (np.ndarray): The input NumPy array.
    val (bool, optional): Whether to print mean, min, max, median, and std.
                          Defaults to True.
    shp (bool, optional): Whether to print the shape of the array.
                          Defaults to False.
  """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
  """Creates directories.

  Args:
    paths (str or list of str): A single path or a list of paths to create.
  """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
  """Creates a directory if it doesn't exist.

  Args:
    path (str): The path to the directory to create.
  """
    if not os.path.exists(path):
        os.makedirs(path)


def set_opt_param(optimizer, key, value):
  """Sets a parameter in an optimizer's parameter groups.

  Args:
    optimizer (torch.optim.Optimizer): The optimizer.
    key (str): The name of the parameter to set (e.g., 'lr', 'weight_decay').
    value: The new value for the parameter.
  """
    for group in optimizer.param_groups:
        group[key] = value


def vis(x):
  """Visualizes a tensor or NumPy array as an image using PIL.

  Args:
    x (torch.Tensor or np.ndarray): The input to visualize.

  Raises:
    NotImplementedError: If the input type is not supported.
  """
    if isinstance(x, torch.Tensor):
        Image.fromarray(tensor2im(x)).show()
    elif isinstance(x, np.ndarray):
        Image.fromarray(x.astype(np.uint8)).show()
    else:
        raise NotImplementedError('vis for type [%s] is not implemented', type(x))


"""tensorboard"""
from tensorboardX import SummaryWriter
from datetime import datetime


def get_summary_writer(log_dir):
  """Creates a TensorBoard SummaryWriter instance.

  The log directory will be created if it doesn't exist, and a subdirectory
  with a timestamp and hostname will be created within it.

  Args:
    log_dir (str): The base directory for TensorBoard logs.

  Returns:
    tensorboardX.SummaryWriter: The SummaryWriter instance.
  """
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)
    return writer
def get_visual(writer,iteration,imgs):
  """Adds images to TensorBoard for visualization.

  Args:
    writer (tensorboardX.SummaryWriter): The TensorBoard SummaryWriter instance.
    iteration (int): The current iteration number.
    imgs (list of torch.Tensor): A list of images to display. Expected to contain
                                 at least two images: 'clean' and 'input'.
  """
    writer.add_image('clean',imgs[0],iteration)
    writer.add_image('input', imgs[1],iteration)
    #writer.add_image('ref', imgs[1],iteration)
    #writer.add_image('input', imgs[2],iteration)


class AverageMeters(object):
  """A class to manage and calculate the average of multiple metrics.

  Args:
    dic (dict, optional): An initial dictionary of metrics. Defaults to None.
    total_num (dict, optional): An initial dictionary of counts for each metric.
                                Defaults to None.
  """
    def __init__(self, dic=None, total_num=None):
        self.dic = dic or {}
        # self.total_num = total_num
        self.total_num = total_num or {}

    def update(self, new_dic):
      """Updates the stored metrics with new values.

      For each key in `new_dic`, if the key already exists in the stored
      metrics, the new value is added to the existing sum and the count
      for that metric is incremented. If the key doesn't exist, it's added
      as a new metric.

      Args:
        new_dic (dict): A dictionary of new metric values to add.
      """
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] += new_dic[key]
                self.total_num[key] += 1
        # self.total_num += 1

    def __getitem__(self, key):
      """Gets the average value of a specific metric.

      Args:
        key (str): The name of the metric.

      Returns:
        float: The average value of the metric.
      """
        return self.dic[key] / self.total_num[key]

    def __str__(self):
      """Returns a string representation of the averaged metrics.

      Returns:
        str: A string displaying the average value of each metric, sorted by key.
      """
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
      """Returns a list of all metric keys.

      Returns:
        list: A list of metric keys.
      """
        return self.dic.keys()


def write_loss(writer, prefix, avg_meters, iteration):
  """Writes loss values to TensorBoard.

  Args:
    writer (tensorboardX.SummaryWriter): The TensorBoard SummaryWriter instance.
    prefix (str): A prefix for the metric names in TensorBoard (e.g., 'train', 'val').
    avg_meters (AverageMeters): An AverageMeters object containing the metrics.
    iteration (int): The current iteration number.
  """
    for key in avg_meters.keys():
        meter = avg_meters[key]
        writer.add_scalar(
            os.path.join(prefix, key), meter, iteration)


"""progress bar"""
import socket

# _, term_width = os.popen('stty size', 'r').read().split()
term_width = 136

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
  """Displays or updates a console progress bar.

  Args:
    current (int): The current progress (e.g., current iteration or epoch).
    total (int): The total number of items.
    msg (str, optional): An optional message to display next to the progress bar.
                         Defaults to None.
  """
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
  """Formats a time duration in seconds into a human-readable string.

  (e.g., D, h, m, s, ms).

  Args:
    seconds (float): The duration in seconds.

  Returns:
    str: The formatted time string.
  """
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def parse_args(args):
  """Parses a comma-separated string of integers into a list of integers.

  Args:
    args (str): A comma-separated string of integers (e.g., "0,1,2").

  Returns:
    list of int: A list of parsed integers.
  """
    str_args = args.split(',')
    parsed_args = []
    for str_arg in str_args:
        arg = int(str_arg)
        if arg >= 0:
            parsed_args.append(arg)
    return parsed_args


def weights_init_kaiming(m):
  """Initializes the weights of a module using Kaiming normal initialization.

  Args:
    m (torch.nn.Module): The module to initialize.
  """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


def batch_PSNR(img, imclean, data_range):
  """Calculates the average Peak Signal-to-Noise Ratio (PSNR) for a batch of images.

  Args:
    img (torch.Tensor): The batch of processed images.
    imclean (torch.Tensor): The batch of clean (ground truth) images.
    data_range (float or int): The data range of the input images (e.g., 255 for uint8).

  Returns:
    float: The average PSNR value for the batch.
  """
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return PSNR / Img.shape[0]


def batch_SSIM(img, imclean):
  """Calculates the average Structural Similarity Index (SSIM) for a batch of images.

  Args:
    img (torch.Tensor): The batch of processed images.
    imclean (torch.Tensor): The batch of clean (ground truth) images.

  Returns:
    float: The average SSIM value for the batch.
  """
    Img = img.data.cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
    Iclean = imclean.data.cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
    SSIM = 0

    for i in range(Img.shape[0]):
        SSIM += structural_similarity(Iclean[i, :, :, :], Img[i, :, :, :], win_size=11,
                                      multichannel=True, data_range=1)
    return SSIM / Img.shape[0]


def data_augmentation(image, mode):
  """Performs data augmentation on an image.

  The input image is expected to be in the format (C, H, W).
  The output image will also be in the format (C, H, W).

  Args:
    image (np.ndarray): The input image.
    mode (int): The augmentation mode.
                0: Original
                1: Flip up and down
                2: Rotate counterwise 90 degrees
                3: Rotate 90 degrees and flip up and down
                4: Rotate 180 degrees
                5: Rotate 180 degrees and flip
                6: Rotate 270 degrees
                7: Rotate 270 degrees and flip

  Returns:
    np.ndarray: The augmented image.
  """
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))
