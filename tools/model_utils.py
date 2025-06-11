def count_parameters(model):
  """Counts the number of trainable parameters in a model.

  Args:
    model: The PyTorch model.

  Returns:
    The total number of trainable parameters.
  """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_conv_layers(model):
  """Counts the number of Conv2d layers and trainable parameters in a model.

  It prints the model name, the number of Conv2d layers, and the total
  number of trainable parameters.

  Args:
    model: The PyTorch model class (not an instance).
  """
    cnt = 0
    for mo in model().modules():
        if type(mo).__name__ == 'Conv2d':
            cnt += 1

    print(model.__name__, cnt, count_parameters(model()))
