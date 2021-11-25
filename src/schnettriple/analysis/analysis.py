import os
import warnings
import logging
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch
import schnetpack as spk


__all__ = ["SchNetAnalysis"]


class SchNetAnalysis:
    """
    Generate the object for analysis of schnet based on
    the directory of the learned model and the path of the database.

    Attributes
    ----------
    modeldir : str or path-like
        The path of the directory that holds the results
        of schnet's calculations.
    dbpath : str or path-like
        The path of the database used for schnet's calculations.
    triples : bool, default=False
        Whether model using triple properties or not.

    """

    def __init__(self, modeldir, dbpath, triples=False):
        self.modeldir = os.path.abspath(modeldir)
        self.dbpath = os.path.abspath(dbpath)
        self.triples = triples

        self.available_properties = spk.AtomsData(self.dbpath).available_properties
        self.plot_fontfamily = "Arial"

    def _split_load(self, data="train"):
        """
        Return a split index from 'split.npz'

        Parameters
        ----------
            data : {'train', 'validation', 'test'}, default='train'
                Choose which type of data you want to load.

        Returns
        -------
            indices: numpy.array
                An array containing the indices of the split data.

        """
        indices = np.load(os.path.join(self.modeldir, "split.npz"))
        if data == "validation":
            data = "val"
        index = "_".join([data, "idx"])

        return indices[index]

    @staticmethod
    def _plot_config(fontfamily="Arial"):
        plt.rcParams["font.family"] = fontfamily
        plt.rcParams["font.size"] = 14
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.minor.visible"] = False
        plt.rcParams["ytick.minor.visible"] = False
        # plt.rcParams["xtick.minor.width"] = 1.0
        # plt.rcParams["ytick.minor.width"] = 1.0
        # plt.rcParams["xtick.minor.size"] = 5
        # plt.rcParams["ytick.minor.size"] = 5
        plt.rcParams["xtick.major.width"] = 1.0
        plt.rcParams["ytick.major.width"] = 1.0
        plt.rcParams["xtick.major.size"] = 5
        plt.rcParams["ytick.major.size"] = 5
        plt.rcParams["xtick.top"] = True
        plt.rcParams["ytick.right"] = True
        plt.rcParams["axes.linewidth"] = 1.0
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["mathtext.default"] = "it"
        plt.rcParams["legend.frameon"] = False

        return None

    def log_plot(
        self,
        error="RMSE",
        props=("energy", "forces"),
        units=("eV", "eV/angstrom"),
        axes=None,
        verbose=True,
    ):
        """
        Plot the progress of the learning on SchNet.
        Generate 3 plots :
            the training and validation loss and epoch,
            the errors of property 1 and epoch,
            the errors of property 2 and epoch

        Parameters
        ----------
            error : {'RMSE', 'MAE'}, default='RMSE'
                Select the type of error.
            props : tuple of shape (2,), default=('energy', 'forces')
                Specify the properties to be plotted. Write in str.
            units : tuple of shape (2,), default=('eV', 'eV/\u212B')
                Specifies the units that the properties have. Write in str.
            axes : array-like of shape (3,) or None, default=None
                Axes to use for plotting the curves.
            verbose : bool default=True
                Specify whether or not to display the last error value
                on the plot.

        Returns
        -------
            plt : module
        """
        self._plot_config(fontfamily=self.plot_fontfamily)

        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        df = pd.read_csv(os.path.join(self.modeldir, "log/log.csv"))

        # plot loss_epoch curve
        axes[0].plot(df["Train loss"], color="r", label="Training loss")
        axes[0].plot(df["Validation loss"], color="g", label="Validation loss")
        axes[0].legend(loc="best")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].set_title("Loss-Epoch")

        # plot error for prpops[0]
        error_0 = "_".join([error, props[0]])
        axes[1].plot(df[error_0], c="b")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel(f"{error} ({units[0]})")
        axes[1].set_title(f"{error} for {props[0]}")

        # plot error for prpops[1]
        error_1 = "_".join([error, props[1]])
        axes[2].plot(df[error_1], c="orange")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel(f"{error} ({units[1]})")
        axes[2].set_title(f"{error} for {props[1]}")

        if verbose:
            middle_epoch = len(df) / 2

            middle0 = (df[error_0].max() + df[error_0].min()) / 2
            score0 = np.array(df[error_0])[-1]
            axes[1].text(middle_epoch, middle0, f"score={score0:.3f} ({units[0]})")

            middle1 = (df[error_1].max() + df[error_1].min()) / 2
            score1 = np.array(df[error_1])[-1]
            axes[2].text(middle_epoch, middle1, f"score={score1:.3f} ({units[1]})")

        return plt

    def inout_property(
        self,
        prop="energy",
        data="train",
        divided_by_atoms=True,
        device="cpu",
        save=True,
        _return=False,
    ):
        """
        Return the atom-by-atom predictions of the system properties
        as an array corresponding to the input values.

        Parameters
        ----------
            prop : str, default=True
                Specify the property you want to predict.
            data : {'train', 'validation', 'test'}, default='train'
                Choose which type of data you want to caluculate.
            divided_by_atoms : bool, default=True
                Specifies whether or not the output should be a value
                for each atom in the system.
            device : {'cpu', 'cuda'}, default='cpu'
                Device for computation.
            save : bool, default=True
                Specify whether or not save the data in
                'modeldir/io_{prop}_{data}.npz'.
            _return : bool, default=False
                Specify whether or not to return the calculation results.
                If you do not want to return the calculation results,
                set ```save=True```

        Returns
        -------
            in_prop : np.array
            out_prop : np.array
                Return the input and output values of some property
                as an two arrays with corresponding each indices.
        """
        logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

        device = torch.device(device)

        indexes = self._split_load(data)

        bestmodel = torch.load(
            os.path.join(self.modeldir, "best_model"), map_location=device
        )

        logging.info("the calculation starts ...")
        first = True
        for idx in indexes:
            idx = int(idx)
            atom_num, props, preds = self._pred_one_schnet(
                model=bestmodel, idx=idx, device=device
            )
            ndim, in_data, out_data = self._make_io_property_data(
                prop, atom_num, props, preds, divided_by_atoms=divided_by_atoms
            )
            if ndim == 1:
                if first:
                    in_prop = in_data
                    out_prop = out_data
                    first = False
                    continue
                if not first:
                    in_prop = np.concatenate((in_prop, in_data), axis=0)
                    out_prop = np.concatenate((out_prop, out_data), axis=0)
            if ndim == 2:
                if first:
                    in_prop = []
                    out_prop = []
                    in_prop.append(in_data)
                    out_prop.append(out_data)
                    first = False
                    continue
                if not first:
                    in_prop.append(in_data)
                    out_prop.append(out_data)

        in_prop = np.array(in_prop)
        out_prop = np.array(out_prop)
        logging.info("the calculation is done!")

        if save:
            dataname = f"io_{prop}_{data}.npz"
            savepath = os.path.abspath(os.path.join(self.modeldir, dataname))
            np.savez(savepath, in_prop=in_prop, out_prop=out_prop)
            logging.info(f"data is saved in {self.modeldir}/{dataname}")

        if _return:
            return in_prop, out_prop
        else:
            return None

    def _make_io_property_data(self, prop, atom_num, props, preds, divided_by_atoms):
        """ """
        ndim = props[prop].ndim
        if ndim == 1:
            if divided_by_atoms:
                out_data = preds[prop].detach().cpu().numpy() / atom_num
                if isinstance(props[prop], np.ndarray):
                    in_data = np.expand_dims(props[prop] / atom_num, axis=0)
                if not isinstance(props[prop], np.ndarray):
                    in_data = np.expand_dims(
                        props[prop].cpu().numpy() / atom_num, axis=0
                    )
            if not divided_by_atoms:
                out_data = preds[prop].detach().cpu().numpy()
                if isinstance(props[prop], np.ndarray):
                    in_data = np.expand_dims(props[prop], axis=0)
                if not isinstance(props[prop], np.ndarray):
                    in_data = np.expand_dims(props[prop].cpu().numpy(), axis=0)

        else:
            if divided_by_atoms:
                out_data = preds[prop].detach().cpu().numpy()[0] / atom_num
                if isinstance(props[prop], np.ndarray):
                    in_data = props[prop] / atom_num
                if not isinstance(props[prop], np.ndarray):
                    in_data = props[prop].cpu().numpy() / atom_num
            if not divided_by_atoms:
                out_data = preds[prop].detach().cpu().numpy()[0]
                if isinstance(props[prop], np.ndarray):
                    in_data = props[prop]
                if not isinstance(props[prop], np.ndarray):
                    in_data = props[prop].cpu().numpy()
        # else:
        #     raise AssertionError

        return ndim, in_data, out_data

    def _pred_one_schnet(self, model, idx, device):
        """
        Return the predicted value in schnet for a single input value.

        Parameters
        ----------
            model : schnetpack.atomistic.model.AtomisticModel
                The model you want to use for prediction.
            idx : int
                Specify the index of the input value of the data
                to be used for prediction.
            device : {torch.device('cpu'), torch.device('cuda')}
                Device for computation.
        Returns
        -------
            atmo_num : int
                Number of atoms constituting the system
            props : dict
                Dict with input values in torch.tensor type.
            preds : dict
                Dict with predictions values in torch.tensor type.
        """
        dataset = spk.AtomsData(self.dbpath, collect_triples=self.triples)
        converter = spk.data.AtomsConverter(collect_triples=self.triples, device=device)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            at, props = dataset.get_properties(idx=idx)
            inputs = converter(at)

        # ase.Atoms object
        atom_num = at.get_global_number_of_atoms()

        # calculate the prediction of the model
        model.eval()
        preds = model(inputs)

        return atom_num, props, preds

    def inout_plot(
        self,
        prop="energy",
        data=("train", "validation"),
        axes=None,
        line=True,
        xlabel="DFT energy",
        unit="eV/atom",
    ):
        """
        Plot the predictions in learned SchNet for individual inputs.

        Parameters
        ----------
            prop : str, default='energy'
                the property to plot
            data : tuple of size under 3, default=('train', 'validation')
                Type of data to be plotted.
            axes : array-like of shape (1,) or None, default=None
                Axes to use for plotting the curves.
            line : bool, default=True
                Whether or not to plot the y=x line.
            xlabel : str, default='DFT energy'
            unit : str, default='ev/atom'

        Returns
        -------
            plt : module
        """
        self._plot_config(fontfamily=self.plot_fontfamily)

        data_num = len(data)
        if data_num > 3:
            raise NotImplementedError(
                "The ``data`` should contain no more than three elements."
            )
        colors = ["red", "navy", "green"]

        if axes is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 6))

        max_x = None
        min_x = None
        for i in range(data_num):
            filepath = os.path.join(self.modeldir, f"io_{prop}_{data[i]}.npz")
            try:
                f = np.load(filepath, allow_pickle=True)
            except FileNotFoundError:
                raise FileNotFoundError(
                    """
                    Please do the calculation with the 'inout_prop()'
                    method first, and save the calculation result.
                    """
                )
            else:
                in_prop = f["in_prop"]
                out_prop = f["out_prop"]
                ax.scatter(
                    in_prop,
                    out_prop,
                    s=4,
                    c=colors[i],
                    alpha=1.1 - (i + 1) / 5,
                    label=data[i],
                )

                if max_x is None:
                    max_x = in_prop.max()
                    min_x = in_prop.min()

                else:
                    if max_x < in_prop.max():
                        max_x = in_prop.max()
                    if min_x > in_prop.min():
                        min_x = in_prop.min()

        # set the axis limitations
        data_length = max_x - min_x
        min_lim = min_x - data_length / 4
        max_lim = max_x + data_length / 4
        ax.set_xlim([min_lim, max_lim])
        ax.set_ylim([min_lim, max_lim])

        # set the axis ticks
        min_tick = float(
            Decimal(min_lim).quantize(Decimal(".1"), rounding=ROUND_CEILING)
        )
        max_tick = float(Decimal(max_lim).quantize(Decimal(".1"), rounding=ROUND_FLOOR))
        tick_step = float(
            Decimal(data_length / 5).quantize(Decimal(".1"), rounding=ROUND_HALF_UP)
        )
        ax.set_xticks(np.arange(min_tick, max_tick, tick_step))
        ax.set_yticks(np.arange(min_tick, max_tick, tick_step))

        # set the axis label
        ax.set_xlabel(f"{xlabel} ({unit})")
        ax.set_ylabel(f"SchNet {prop} ({unit})")
        ax.legend(loc="best")

        if line:
            ax.plot([min_lim, max_lim], [min_lim, max_lim], "--", c="gray")

        plt.gca().set_aspect("equal", "box")

        return plt

    def rmse(self, prop="energy", data="train", save=True):
        """
        Calculate the rmse from 'io_{data}_{prop}.npz' file.

        Parameters
        ----------
            prop : str, default='energy'
                The property to compute.
            data : str, default='train'
                Type of data to be computed.
            save : bool, default=True
                Specify whether to save the calculation results
                to a text file(rmse_{prop}_{data}.txt).

        Returns
        -------
            rmse : np.float
        """
        filepath = os.path.join(self.modeldir, f"io_{prop}_{data}.npz")
        try:
            f = np.load(filepath, allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                """
                Please do the calculation with the 'inout_prop()' method first,
                and save the calculation result (set 'save=True').
                """
            )
        else:
            y_true = f["in_prop"]
            y_pred = f["out_prop"]
            rmse = 0
            for true, pred in zip(y_true, y_pred):
                rmse += mean_squared_error(true, pred)
            rmse /= len(y_true)
            rmse = np.sqrt(rmse)

        if save:
            filename = f"rmse_{prop}_{data}.txt"
            filepath = os.path.join(self.modeldir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                print("RMSE\n", f"{rmse}", file=f)

        return rmse
