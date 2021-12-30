#!/usr/bin/env python
import os
import argparse
import torch
import logging
import schnetpack as spk
from schnetpack.utils import (
    setup_run,
    get_divide_by_atoms,
)
from schnetpack.utils.script_utils.settings import get_environment_provider

from schnettriple.utils.evaluation import evaluate
from schnettriple.utils.data import get_dataset, get_statistics, get_loaders
from schnettriple.utils.model import get_model
from schnettriple.utils.training import get_metrics, get_trainer
from schnettriple.utils.script_utils import ScriptError, read_from_json

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def main(args):
    # setup
    train_args = setup_run(args)

    device = torch.device("cuda" if args.cuda else "cpu")

    # get dataset
    environment_provider = get_environment_provider(train_args, device=device)
    dataset = get_dataset(train_args, environment_provider=environment_provider)

    # get dataloaders
    split_path = os.path.join(args.modelpath, "split.npz")
    train_loader, val_loader, test_loader = get_loaders(
        args, dataset=dataset, split_path=split_path, logging=logging
    )

    # define metrics
    metrics = get_metrics(train_args)

    # train or evaluate
    if args.mode == "train":

        # get statistics
        atomref = dataset.get_atomref(args.property)
        mean, stddev = get_statistics(
            args=args,
            split_path=split_path,
            train_loader=train_loader,
            atomref=atomref,
            divide_by_atoms=get_divide_by_atoms(args),
            logging=logging,
        )

        # build model
        model = get_model(args, train_loader, mean, stddev, atomref, logging=logging)

        # build trainer
        logging.info("training...")
        trainer = get_trainer(args, model, train_loader, val_loader, metrics)

        # run training
        trainer.train(
            device,
            n_epochs=args.n_epochs,
            regularization=args.regularization,
            l1_lambda=args.l1_lambda,
        )
        logging.info("...training done!")

    elif args.mode == "eval":

        # remove old evaluation files
        evaluation_fp = os.path.join(args.modelpath, "evaluation.txt")
        if os.path.exists(evaluation_fp):
            if args.overwrite:
                os.remove(evaluation_fp)
            else:
                raise ScriptError(
                    "The evaluation file does already exist at {}! Add overwrite flag"
                    " to remove.".format(evaluation_fp)
                )

        # load model
        logging.info("loading trained model...")
        model = spk.utils.load_model(
            os.path.join(args.modelpath, "best_model"), map_location=device
        )

        # run evaluation
        logging.info("evaluating...")
        if spk.utils.get_derivative(train_args) is None:
            with torch.no_grad():
                evaluate(
                    args,
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    device,
                    metrics=metrics,
                )
        else:
            evaluate(
                args,
                model,
                train_loader,
                val_loader,
                test_loader,
                device,
                metrics=metrics,
            )
        logging.info("... evaluation done!")

    else:
        raise ScriptError("Unknown mode: {}".format(args.mode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("argsjsonpath", help="json file path")
    args = parser.parse_args()
    args = read_from_json(args.argsjsonpath)

    main(args)
