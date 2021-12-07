import logging

import numpy as np
import torch
from schnetpack import Properties
from schnetpack.data.loader import AtomsLoader


logger = logging.getLogger(__name__)


__all__ = ["AtomsLoaderTriple"]


def _collate_aseatoms_modify(examples):
    """
    Build batch from systems and properties & apply padding
    Parameters
    ----------
    examples : list

    Returns
    -------
    dict : str->torch.Tensor
        mini-batch of atomistic systems
    """
    properties = examples[0]

    # initialize maximum sizes
    max_size = {
        prop: np.array(val.size(), dtype=np.int) for prop, val in properties.items()
    }

    # get maximum sizes
    for properties in examples[1:]:
        for prop, val in properties.items():
            max_size[prop] = np.maximum(
                max_size[prop], np.array(val.size(), dtype=np.int)
            )

    # initialize batch
    batch = {
        p: torch.zeros(len(examples), *[int(ss) for ss in size]).type(
            examples[0][p].type()
        )
        for p, size in max_size.items()
    }
    has_atom_mask = Properties.atom_mask in batch.keys()
    has_neighbor_mask = Properties.neighbor_mask in batch.keys()

    if not has_neighbor_mask:
        batch[Properties.neighbor_mask] = torch.zeros_like(
            batch[Properties.neighbors]
        ).float()
    if not has_atom_mask:
        batch[Properties.atom_mask] = torch.zeros_like(batch[Properties.Z]).float()

    # If neighbor pairs are requested, construct mask placeholders
    # Since the structure of both idx_j and idx_k is identical
    # (not the values), only one cutoff mask has to be generated
    if Properties.neighbor_pairs_j in properties:
        batch[Properties.neighbor_pairs_mask] = torch.zeros_like(
            batch[Properties.neighbor_pairs_j]
        ).float()

    # build batch and pad
    for k, properties in enumerate(examples):
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val

        # add mask
        if not has_neighbor_mask:
            nbh = properties[Properties.neighbors]
            shape = nbh.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            mask = nbh >= 0
            batch[Properties.neighbor_mask][s] = mask
            batch[Properties.neighbors][s] = nbh * mask.long()

        if not has_atom_mask:
            z = properties[Properties.Z]
            shape = z.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Properties.atom_mask][s] = z > 0

        # Check if neighbor pair indices are present
        # Since the structure of both idx_j and idx_k is identical
        # (not the values), only one cutoff mask has to be generated
        if Properties.neighbor_pairs_j in properties:
            nbh_idx_j = properties[Properties.neighbor_pairs_j]
            nbh_idx_k = properties[Properties.neighbor_pairs_k]
            shape = nbh_idx_j.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            triple_mask = nbh_idx_j >= 0
            batch[Properties.neighbor_pairs_mask][s] = triple_mask
            batch[Properties.neighbor_pairs_j][s] = nbh_idx_j * triple_mask.long()
            batch[Properties.neighbor_pairs_k][s] = nbh_idx_k * triple_mask.long()

    return batch


class AtomsLoaderTriple(AtomsLoader):
    """
    Specialized ``torch.data.DataLoader`` which uses the correct
    collate_fn for AtomsData and provides functionality for calculating mean
    and stddev.

    Parameters
    ----------
    dataset : Dataset
        dataset from which to load the data.
    batch_size : int, default=1
        how many samples per batch to load.
    shuffle : bool, default=False
        set to ``True`` to have the data reshuffled at every epoch.
    sampler : Sampler
        defines the strategy to draw samples from the dataset. If specified, ``shuffle`` must be False.
    batch_sampler : Sampler
        like sampler, but returns a batch of indices at a time. Mutually exclusive with batch_size, shuffle,
        sampler, and drop_last.
    num_workers : int, default=0
        how many subprocesses to use for data loading.
        0 means that the data will be loaded in the main process.
    collate_fn : callable, default=_collate_aseatoms_modify
        merges a list of samples to form a mini-batch.
    pin_memory : bool
        If ``True``, the data loader will copy tensors into CUDA pinned memory before returning them.
    drop_last : bool, default=False
        set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
        If ``False`` and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
    timeout : numeric, default=0
        if positive, the timeout value for collecting a batch from workers. Should always be non-negative.
    worker_init_fn : callable, default=None
        If not None, this will be called on each worker subprocess with the worker id (an int in``[0, num_workers - 1]``)
        as input, after seeding and before data loading.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=_collate_aseatoms_modify,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(AtomsLoaderTriple, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
