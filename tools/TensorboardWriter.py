import threading
from tensorboardX import SummaryWriter


class SingleSummaryWriter(SummaryWriter):
  """A singleton SummaryWriter class.

  This class ensures that only one instance of SummaryWriter is created,
  even if the constructor is called multiple times. This is useful for
  managing a single TensorBoard logging instance across different parts
  of a project.

  Args:
    logdir (str, optional): The directory to save TensorBoard logs.
                           Defaults to None.
    **kwargs: Additional keyword arguments to pass to the SummaryWriter
              constructor.
  """
    _instance_lock = threading.Lock()

    def __init__(self, logdir=None, **kwargs):
      """Initializes the SingleSummaryWriter.

      This method is called only once when the first instance is created.
      Subsequent calls to the constructor will not re-initialize the instance.

      Args:
        logdir (str, optional): The directory to save TensorBoard logs.
                               Defaults to None.
        **kwargs: Additional keyword arguments to pass to the SummaryWriter
                  constructor.
      """
        super().__init__(logdir, **kwargs)

    def __new__(cls, *args, **kwargs):
      """Creates or returns the singleton instance of SingleSummaryWriter.

      Args:
        *args: Positional arguments to pass to the SummaryWriter constructor.
        **kwargs: Keyword arguments to pass to the SummaryWriter constructor.

      Returns:
        The singleton instance of SingleSummaryWriter.
      """
        if not hasattr(SingleSummaryWriter, "_instance"):
            with SingleSummaryWriter._instance_lock:
                if not hasattr(SingleSummaryWriter, "_instance"):
                    SingleSummaryWriter._instance = object.__new__(cls)
        return SingleSummaryWriter._instance
