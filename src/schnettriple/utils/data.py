import os
import numpy as np
import torch
import schnetpack as spk
from torch.utils.data.sampler import RandomSampler
from schnetpack.utils.script_utils.script_error import ScriptError

import schnettriple as snt
from schnettriple.data.loader import AtomsLoaderTriple


__all__ = ["get_loaders", "get_statistics", "get_dataset"]


def get_statistics(
    args, split_path, train_loader, atomref, divide_by_atoms=False, logging=None
):
    """
    Get statistics for molecular properties. Use split file if possible.

    Parameters
    ----------
    args : argparse.Namespace
        parsed script arguments
    split_path : str
        path to the split file
    train_loader : schnetpack.data.AtomsLoader
        dataloader for training set
    atomref : dict
        atomic references
    divide_by_atoms : dict or bool, default=None
        divide mean by number of atoms if True
    logging : default=None
        logger

    Returns
    -------
    mean : dict of torch.Tensor
        mean values for the selected properties
    stddev : dict of torch.Tensor
        stddev values for the selected properties
    """
    # check if split file exists
    if not os.path.exists(split_path):
        raise ScriptError("No split file found ad {}".format(split_path))
    split_data = np.load(split_path)

    # check if split file contains statistical data
    if "mean" in split_data.keys():
        mean = {args.property: torch.from_numpy(split_data["mean"])}
        stddev = {args.property: torch.from_numpy(split_data["stddev"])}
        if logging is not None:
            logging.info("cached statistics was loaded...")

    # calculate statistical data
    else:
        mean, stddev = train_loader.get_statistics(
            args.property, divide_by_atoms, atomref
        )
        np.savez(
            split_path,
            train_idx=split_data["train_idx"],
            val_idx=split_data["val_idx"],
            test_idx=split_data["test_idx"],
            mean=mean[args.property].numpy(),
            stddev=stddev[args.property].numpy(),
        )

    return mean, stddev


def get_loaders(args, dataset, split_path, logging=None):
    """
    Create train-val-test splits for dataset and
    return the corresponding dataloaders.

    Parameters
    ----------
    args : argparse.Namespace
        parsed script arguments
    dataset : schnetpack.AtomsData
        total dataset
    split_path : str
        path to split file
    logging : default=None
        logger

    Returns
    -------
    schnettriple.AtomsLoader, schettriple.AtomsLoader, schnettriple.AtomsLoader
        dataloaders for train, val and test
    """
    if logging is not None:
        logging.info("create splits...")

    # create or load dataset splits depending on args.mode
    if args.mode == "train":
        data_train, data_val, data_test = spk.data.train_test_split(
            dataset, *args.split, split_file=split_path
        )
    else:
        data_train, data_val, data_test = spk.data.train_test_split(
            dataset, split_file=split_path
        )

    if logging is not None:
        logging.info("load data...")

    # build dataloaders
    train_loader = AtomsLoaderTriple(
        data_train,
        batch_size=args.batch_size,
        sampler=RandomSampler(data_train),
        num_workers=4,
        pin_memory=args.cuda,
    )
    val_loader = AtomsLoaderTriple(
        data_val, batch_size=args.batch_size, num_workers=2, pin_memory=args.cuda
    )
    test_loader = AtomsLoaderTriple(
        data_test, batch_size=args.batch_size, num_workers=2, pin_memory=args.cuda
    )

    return train_loader, val_loader, test_loader


def get_dataset(args, environment_provider, logging=None):
    """
    Get dataset from arguments.

    Parameters
    ----------
    args : argparse.Namespace
        parsed arguments
    environment_provider : schetpack.environment.BaseEnvironmentProvider
        environment-provider of dataset
    logging : default=None
        logger

    Returns
    -------
    dataset : schnetpack.data.AtomsData
        dataset
    """
    if args.dataset == "qm9":
        if logging:
            logging.info("QM9 will be loaded...")
        qm9 = spk.datasets.QM9(
            args.datapath,
            download=True,
            load_only=[args.property],
            collect_triples=True,
            remove_uncharacterized=args.remove_uncharacterized,
            environment_provider=environment_provider,
        )
        return qm9
    elif args.dataset == "ani1":
        if logging:
            logging.info("ANI1 will be loaded...")
        ani1 = spk.datasets.ANI1(
            args.datapath,
            download=True,
            load_only=[args.property],
            collect_triples=True,
            num_heavy_atoms=args.num_heavy_atoms,
            environment_provider=environment_provider,
        )
        return ani1
    elif args.dataset == "md17":
        if logging:
            logging.info("MD17 will be loaded...")
        md17 = spk.datasets.MD17(
            args.datapath,
            args.molecule,
            download=True,
            collect_triples=True,
            environment_provider=environment_provider,
        )
        return md17
    elif args.dataset == "matproj":
        if logging:
            logging.info("Materials project will be loaded...")
        mp = spk.datasets.MaterialsProject(
            args.datapath,
            apikey=args.apikey,
            download=True,
            load_only=[args.property],
            environment_provider=environment_provider,
        )
        if args.timestamp:
            mp = mp.at_timestamp(args.timestamp)
        return mp
    elif args.dataset == "omdb":
        if logging:
            logging.info("Organic Materials Database will be loaded...")
        omdb = spk.datasets.OrganicMaterialsDatabase(
            args.datapath,
            download=True,
            load_only=[args.property],
            environment_provider=environment_provider,
        )
        return omdb
    elif args.dataset == "custom":
        if logging:
            logging.info("Custom dataset will be loaded...")

        # define properties to be loaded
        load_only = [args.property]
        if args.derivative is not None:
            load_only.append(args.derivative)

        if args.stress is not None:
            load_only.append(args.stress)

        dataset = spk.AtomsData(
            args.datapath,
            load_only=load_only,
            collect_triples=True,
            environment_provider=environment_provider,
        )
        return dataset
    else:
        raise snt.utils.script_utils.ScriptError("Invalid dataset selected!")
