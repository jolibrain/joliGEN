"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from torch.utils import data


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (dataset_filename, target_dataset_name)
        )

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt, phase):
    dataset_class = find_dataset_using_name(opt.data_dataset_mode)
    dataset = dataset_class(opt, phase)
    return dataset


def create_dataloader(opt, rank, dataset, batch_size):
    data_loader = CustomDatasetDataLoader(opt, rank, dataset, batch_size)
    dataset = data_loader.load_data()
    return dataset


def create_dataset_temporal(opt, phase):
    dataset_class = find_dataset_using_name("temporal")
    dataset = dataset_class(opt, phase)
    return dataset


def create_iterable_dataloader(opt, rank, dataset, batch_size):
    data_loader = IterableCustomDatasetDataLoader(opt, rank, dataset, batch_size)
    dataset = data_loader.load_data()
    return dataset


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) > 0:
        return torch.utils.data.dataloader.default_collate(batch)
    else:
        return None


class CustomDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, rank, dataset, batch_size):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.dataset = dataset
        if rank == 0:
            print("dataset [%s] was created" % type(self.dataset).__name__)
        if len(opt.gpu_ids) > 1:
            world_size = len(opt.gpu_ids)
            sampler = data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=not opt.data_serial_batches,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = not opt.data_serial_batches
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=int(opt.data_num_threads),
            collate_fn=collate_fn,
        )

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.data_max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if data is None:
                continue
            if i * self.opt.train_batch_size >= self.opt.data_max_dataset_size:
                break
            yield data


class IterableCustomDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, rank, dataset, batch_size):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.dataset = dataset
        if rank == 0:
            print("dataset [%s] was created" % type(self.dataset).__name__)
        sampler = None
        shuffle = not opt.data_serial_batches
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=int(opt.data_num_threads),
            collate_fn=collate_fn,
        )

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return self.opt.data_max_dataset_size

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if data is None:
                continue
            yield data
