import os
import time
import datetime
import shutil


def make_empty_dir(new_dir):
  """Creates an empty directory.

  If the directory already exists, it will be removed and recreated.

  Args:
    new_dir: The path to the directory to create.
  """
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir, exist_ok=True)


def get_timestamp():
  """Gets a timestamp string.

  The timestamp is the current time in seconds since the epoch,
  with the decimal point removed.

  Returns:
    A string representing the timestamp.
  """
    return str(time.time()).replace('.', '')


def get_formatted_time():
  """Gets a formatted time string.

  The time string is formatted as YYYYMMDD-HHMMSS.

  Returns:
    A string representing the formatted time.
  """
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
