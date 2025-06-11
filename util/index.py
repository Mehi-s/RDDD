# Metrics/Indexes
try:
    from skimage.measure import compare_ssim, compare_psnr  
except:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
    
from functools import partial
import numpy as np


class Bandwise(object):
  """Applies an index function bandwise (channel-wise) to images.

  Args:
    index_fn: The function to apply to each band (channel) of the images.
              This function should take two 2D numpy arrays (a single band
              from each image) as input and return a scalar value.
  """
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
      """Applies the index function bandwise to the input images.

      Args:
        X: The first input image (numpy array, HxWxC or HxW).
        Y: The second input image (numpy array, HxWxC or HxW).

      Returns:
        A list of scalar values, where each value is the result of applying
        the index function to the corresponding band of the input images.
        If the input images are grayscale, a list with a single element is returned.
      """
        C = X.shape[-1]
        bwindex = []
        for ch in range(C):
            x = X[..., ch]
            y = Y[..., ch]
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex


cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=255))
"""Calculates Peak Signal-to-Noise Ratio (PSNR) bandwise."""
cal_bwssim = Bandwise(partial(compare_ssim, data_range=255))
"""Calculates Structural Similarity Index (SSIM) bandwise."""


def compare_ncc(x, y):
  """Computes the Normalized Cross-Correlation (NCC) between two 2D arrays.

  Args:
    x: The first input array.
    y: The second input array.

  Returns:
    The NCC value.
  """
    return np.mean((x - np.mean(x)) * (y - np.mean(y))) / (np.std(x) * np.std(y))


def ssq_error(correct, estimate):
  """Computes the sum-squared-error (SSE) between two 2D arrays.

  The estimate is optimally scaled by a factor `alpha` to minimize the SSE.
  alpha = sum(correct * estimate) / sum(estimate^2)
  SSE = sum((correct - alpha * estimate)^2)

  Args:
    correct: The ground truth 2D array.
    estimate: The estimated 2D array.

  Returns:
    The sum-squared-error.
  """
    """Compute the sum-squared-error for an image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True. If the inputs are color, each color channel can be
    rescaled independently."""
    assert correct.ndim == 2
    if np.sum(estimate ** 2) > 1e-5:
        alpha = np.sum(correct * estimate) / np.sum(estimate ** 2)
    else:
        alpha = 0.
    return np.sum((correct - alpha * estimate) ** 2)


def local_error(correct, estimate, window_size, window_shift):
  """Computes the local sum-squared-error between two images.

  The error is calculated over sliding windows, where the estimate within
  each window can be rescaled to minimize the error for that window.

  Args:
    correct: The ground truth image (HxWxC numpy array).
    estimate: The estimated image (HxWxC numpy array).
    window_size: The size of the sliding window.
    window_shift: The step size for sliding the window.

  Returns:
    The normalized sum of local sum-squared-errors.
  """
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N, C = correct.shape
    ssq = total = 0.
    for c in range(C):
        for i in range(0, M - window_size + 1, window_shift):
            for j in range(0, N - window_size + 1, window_shift):
                correct_curr = correct[i:i + window_size, j:j + window_size, c]
                estimate_curr = estimate[i:i + window_size, j:j + window_size, c]
                ssq += ssq_error(correct_curr, estimate_curr)
                total += np.sum(correct_curr ** 2)
    # assert np.isnan(ssq/total)
    return ssq / total


def quality_assess(X, Y):
  """Assesses the quality of an estimated image compared to a correct image.

  Calculates PSNR, SSIM, Local Mean Squared Error (LMSE), and NCC.

  Args:
    X: The estimated image (numpy array).
    Y: The correct (ground truth) image (numpy array).

  Returns:
    A dictionary containing the calculated quality metrics:
    {'PSNR': float, 'SSIM': float, 'LMSE': float, 'NCC': float}
  """
    # Y: correct; X: estimate
    psnr = np.mean(cal_bwpsnr(Y, X))
    ssim = np.mean(cal_bwssim(Y, X))
    lmse = local_error(Y, X, 20, 10)
    ncc = compare_ncc(Y, X)
    return {'PSNR': psnr, 'SSIM': ssim, 'LMSE': lmse, 'NCC': ncc}
