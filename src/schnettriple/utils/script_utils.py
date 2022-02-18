import time
import random
import os
import logging
from shutil import rmtree
import numpy as np
import torch
import schnetpack as spk


__all__ = ["ScriptError", "setup_run", "get_environment_provider"]


class ScriptError(Exception):
    pass


def setup_run(args):
    if args.mode == "train":

        # build modeldir
        if args.overwrite and os.path.exists(args.modelpath):
            logging.info("existing model will be overwritten...")
            rmtree(args.modelpath)
        if not os.path.exists(args.modelpath):
            os.makedirs(args.modelpath)

        _set_random_seed(args.seed)
        train_args = args
    else:
        # check if modelpath is valid
        if not os.path.exists(args.modelpath):
            raise ScriptError(
                "The selected modeldir does not exist " "at {}!".format(args.modelpath)
            )

        # load training arguments
        train_args = args

    # apply alias definitions
    train_args = _apply_aliases(train_args)
    return train_args


def _set_random_seed(seed: int):
    """
    This function sets the random seed (if given) or creates one for torch and numpy random state initialization

    Parameters
    ----------
    seed : int or None, default=None
        if seed not present, it is generated based on time
    """
    if seed is None:
        seed = int(time.time() * 1000.0)
        # Reshuffle current time to get more different seeds within shorter time intervals
        # Taken from https://stackoverflow.com/questions/27276135/python-random-system-time-seed
        # & Gets overlapping bits, << and >> are binary right and left shifts
        seed = (
            ((seed & 0xFF000000) >> 24)
            + ((seed & 0x00FF0000) >> 8)
            + ((seed & 0x0000FF00) << 8)
            + ((seed & 0x000000FF) << 24)
        )
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    logging.info("Random state initialized with seed {:<10d}".format(seed))


def _apply_aliases(args):
    # force alias for custom dataset
    if args.dataset == "custom":
        if args.force is not None:
            if args.derivative is not None:
                raise ScriptError(
                    "Force and derivative define the same property. Please don`t use "
                    "both."
                )
            args.derivative = args.force
            args.negative_dr = True

            # add rho value if selected
            if "force" in args.rho.keys():
                args.rho["derivative"] = args.rho.pop("force")

    return args


def get_environment_provider(args, device):
    if args.environment_provider == "simple":
        return spk.environment.SimpleEnvironmentProvider()
    elif args.environment_provider == "ase":
        return spk.environment.AseEnvironmentProvider(cutoff=args.cutoff)
    elif args.environment_provider == "torch":
        return spk.environment.TorchEnvironmentProvider(
            cutoff=args.cutoff, device="cpu"
        )
    else:
        raise NotImplementedError
