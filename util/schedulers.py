from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier (float): Target learning rate = base lr * multiplier. Should be >= 1.0.
                            If multiplier = 1.0, LR starts from 0 and ends up with the base_lr.
        total_epoch (int): The number of epochs to reach the target learning rate.
        after_scheduler (torch.optim.lr_scheduler._LRScheduler, optional):
                           Scheduler to use after the warmup period. Defaults to None.
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
      """Initializes the GradualWarmupScheduler.

      Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier (float): Target learning rate = base lr * multiplier.
        total_epoch (int): The number of epochs for warmup.
        after_scheduler (torch.optim.lr_scheduler._LRScheduler, optional):
                           Scheduler to use after warmup. Defaults to None.
      """
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
      """Calculates the learning rate for the current epoch.

      Returns:
        A list of learning rates for each parameter group.
      """
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
      """Handles the step when the after_scheduler is ReduceLROnPlateau.

      Args:
        metrics (float): The metric used by ReduceLROnPlateau (e.g., validation loss).
        epoch (int, optional): The current epoch number. Defaults to None.
      """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
      """Performs a scheduler step.

      If an `after_scheduler` is provided and is not ReduceLROnPlateau,
      its step method is called after the warmup period.
      If `after_scheduler` is ReduceLROnPlateau, `step_ReduceLROnPlateau` is called.
      Otherwise, the parent class's step method is called.

      Args:
        epoch (int, optional): The current epoch number. Defaults to None.
        metrics (float, optional): The metric for ReduceLROnPlateau. Defaults to None.
      """
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
