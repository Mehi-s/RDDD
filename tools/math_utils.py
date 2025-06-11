import math
import torch


## [-1,1]
def tensor2log(x):
  """Transforms a tensor from the range [-1, 1] to logarithmic scale.

  The transformation is: log(a * x + b), where a and b are chosen
  such that the input range [-1, 1] maps to [1, e].

  Args:
    x: The input tensor.

  Returns:
    The transformed tensor in logarithmic scale.
  """
    a = (math.e - 1.) / 2.
    b = (math.e + 1.) / 2.
    x = a * x + b
    return torch.log(x).float()


def log2tensor(x):
  """Transforms a tensor from logarithmic scale back to the range [-1, 1].

  This is the inverse of the tensor2log function.

  Args:
    x: The input tensor in logarithmic scale.

  Returns:
    The transformed tensor in the range [-1, 1].
  """
    a = 2. / (math.e - 1.)
    b = (math.e + 1.) / (1. - math.e)
    x = torch.exp(x)
    x = a * x + b
    return x.float()


## [0,1]
def _tensor2log(x):
  """Transforms a tensor from the range [0, 1] to logarithmic scale.

  The transformation is: log(a * x + b), where a and b are chosen
  such that the input range [0, 1] maps to [1, e].

  Args:
    x: The input tensor.

  Returns:
    The transformed tensor in logarithmic scale.
  """
    a = math.e - 1.
    b = 1.
    x = a * x + b
    return torch.log(x).float()


def _log2tensor(x):
  """Transforms a tensor from logarithmic scale back to the range [0, 1].

  This is the inverse of the _tensor2log function.

  Args:
    x: The input tensor in logarithmic scale.

  Returns:
    The transformed tensor in the range [0, 1].
  """
    a = 1. / (math.e - 1.)
    b = -a
    x = torch.exp(x)
    x = a * x + b
    return x.float()


if __name__ == '__main__':
    inputx = torch.rand(1, 3, 64, 64)
    print(torch.min(inputx), torch.max(inputx))

    out = _tensor2log(inputx)
    print(torch.min(out), torch.max(out))

    out = _log2tensor(out)
    print(torch.min(out), torch.max(out))
    print(torch.mean(out - inputx))
