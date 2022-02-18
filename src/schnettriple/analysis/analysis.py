import os
import warnings
import logging
import numpy as np
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

    def _split_load(self, data="train"):
        """
        Return a split index from 'modeldir/split.npz'

        Parameters
        ----------
        data : str of {'train', 'validation', 'test'}, default='train'
            choose which type of data you want to load.

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

    def inout_property(
        self,
        prop="energy",
        data="train",
        divided_by_atoms=True,
        environment="simple",
        cutoff=None,
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
        environment : str of {"simple", "ase", "torch"}, default="simple"
            define how neighborhood is calculated.
        cutoff : float or None, default=None
            cutoff radious.
        device : str of {'cpu', 'cuda'}, default='cpu'
            device for computation.
        save : bool, default=True
            whether or not save the data in 'modeldir/io_{prop}_{data}.npz'.
        _return : bool, default=False
            whether or not to return the calculation results.
            If you do not want to return the calculation results, set `save=True`

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
                model=bestmodel,
                environment=environment,
                cutoff=cutoff,
                idx=idx,
                device=device,
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

        in_prop = np.array(in_prop, dtype=object)
        out_prop = np.array(out_prop, dtype=object)
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

    def _get_environment_provider(self, environment_provider, cutoff):
        """ """
        if environment_provider == "simple":
            return spk.environment.SimpleEnvironmentProvider()
        elif environment_provider == "ase":
            return spk.environment.AseEnvironmentProvider(cutoff=cutoff)
        elif environment_provider == "torch":
            return spk.environment.TorchEnvironmentProvider(cutoff=cutoff, device="cpu")
        else:
            raise NotImplementedError

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

    def _pred_one_schnet(self, model, environment, cutoff, idx, device):
        """
        Return the predicted value in schnet for a single input value.

        Parameters
        ----------
        model : schnetpack.atomistic.model.AtomisticModel
            the model you want to use for prediction.
        environment : str of {"simple", "ase", "torch"}, default="simple"
            define how neighborhood is calculated.
        cutoff : float
            cutoff radious.
        idx : int
            the index of the input value of the data to be used for prediction.
        device : {torch.device('cpu'), torch.device('cuda')}
            device for computation.

        Returns
        -------
        atmo_num : int
            number of atoms constituting the system
        props : dict
            dict with input values in torch.tensor type.
        preds : dict
            dict with predictions values in torch.tensor type.
        """
        environment_provider = self._get_environment_provider(
            environment_provider=environment, cutoff=cutoff, device=device
        )
        dataset = spk.AtomsData(
            self.dbpath,
            collect_triples=self.triples,
            environment_provider=environment_provider,
        )
        converter = spk.data.AtomsConverter(
            environment_provider=environment_provider,
            collect_triples=self.triples,
            device=device,
        )

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
            whether to save the calculation results
            to a text file(rmse_{prop}_{data}.txt).

        Returns
        -------
        rmse : np.float
            rmse value
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
            y_true = y_true.flatten()
            y_pred = f["out_prop"]
            y_pred = y_pred.flatten()
            rmse = np.average((y_pred - y_true) ** 2)
            rmse = np.sqrt(rmse)

        if save:
            filename = f"rmse_{prop}_{data}.txt"
            filepath = os.path.join(self.modeldir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                print(f"RMSE {prop}({data})", f"{rmse}", file=f)

        return rmse
