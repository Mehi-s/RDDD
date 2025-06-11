import bisect
import warnings


class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        """Adds two datasets together.

        Args:
          other: The other dataset to add.

        Returns:
          A ConcatDataset containing both datasets.
        """
        return ConcatDataset([self, other])

    def reset(self):
      """Resets the dataset.

      This method is intended to be overridden by subclasses.
      """
        return


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
      """Calculates the cumulative sum of a sequence.

      Args:
        sequence: The input sequence.

      Returns:
        A list containing the cumulative sum of the sequence.
      """
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
      """Initializes the ConcatDataset.

      Args:
        datasets: A sequence of datasets to concatenate.
      """
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
      """Returns the total length of the concatenated datasets."""
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
      """Returns the item at the given index from the concatenated datasets.

      Args:
        idx: The index of the item to return.

      Returns:
        The item at the given index.
      """
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
      """Returns the cumulative sizes of the datasets.

      Note:
        This property is deprecated and will be removed in a future version.
        Use `cumulative_sizes` instead.
      """
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes