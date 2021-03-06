import numpy as np
import torch
from torch import nn as nn
from torch.autograd import grad
import schnetpack
from schnetpack import Properties
from schnetpack.nn import shifted_softplus

from schnettriple.nn.base import Dense

__all__ = ["MLP", "Atomwise"]


class AtomwiseError(Exception):
    pass


class MLP(nn.Module):
    """
    Multiple layer fully connected perceptron neural network.

    Parameters
    ----------
    n_in : int
        number of input nodes.
    n_out : int
        number of output nodes.
    n_hidden : list of int or int or None, default=None
        number hidden layer nodes.
        If an integer, same number of node is used for all hidden layers resulting
        in a rectangular network.
        If None, the number of neurons is divided by two after each layer starting
        n_in resulting in a pyramidal network.
    n_layers : int, default=2
        number of hidden layers.
    activation : callable, default=schnetpack.nn.activations.shifted_softplus
        activation function. All hidden layers would the same activation function
        except the output layer that does not apply any activation function.
    """

    def __init__(
        self,
        n_in,
        n_out,
        n_hidden=None,
        n_layers=2,
        activation=shifted_softplus,
    ):
        super(MLP, self).__init__()
        # get list of number of nodes in input, hidden & output layers
        if n_hidden is None:
            c_neurons = n_in
            self.n_neurons = []
            for i in range(n_layers):
                self.n_neurons.append(c_neurons)
                c_neurons = max(n_out, c_neurons // 2)
            self.n_neurons.append(n_out)
        else:
            # get list of number of nodes hidden layers
            if type(n_hidden) is int:
                n_hidden = [n_hidden] * (n_layers - 1)
            self.n_neurons = [n_in] + n_hidden + [n_out]

        # assign a Dense layer (with activation function) to each hidden layer
        layers = [
            Dense(self.n_neurons[i], self.n_neurons[i + 1], activation=activation)
            for i in range(n_layers - 1)
        ]
        # assign a Dense layer (without activation function) to the output layer
        layers.append(Dense(self.n_neurons[-2], self.n_neurons[-1], activation=None))
        # put all layers together to make the network
        self.out_net = nn.Sequential(*layers)

    def forward(self, inputs):
        """
        Compute neural network output.

        Parameters
        ----------
        inputs : torch.Tensor
            network input.

        Returns
        -------
        torch.Tensor
            network output.
        """
        return self.out_net(inputs)


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction,
    e.g. for the energy.

    Parameters
    ----------
    n_in : int
        input dimension of representation
    n_out : int, default=1
        output dimension of target property
    aggregation_mode : str, default=sum
        one of {sum, avg}
    n_layers : int, default=2
        number of nn in output network
    n_neurons : list of int or None, default=None
        number of neurons in each layer of the output network.
        If `None`, divide neurons by 2 in each layer.
    activation : collable, default=schnetpack.nn.activations.shifted_softplus
        activation function for hidden nn
    property : str, default="y"
        name of the output property
    contributions : str or None, default=None
        Name of property contributions in return dict.
        No contributions returned if None.
    derivative : str or None, default=None
        Name of property derivative. No derivative returned if None.
    negative_dr : bool, default=False
        Multiply the derivative with -1 if True.
    stress : str or None, default=None
        Name of stress property. Compute the derivative with
        respect to the cell parameters if not None.
    create_graph : bool, default=False
        If False, the graph used to compute the grad will be freed.
        Note that in nearly all cases setting this option to True is not
        needed and often can be worked around in a much more efficient way.
        Defaults to the value of create_graph.
    mean : torch.Tensor or None, default=None
        mean of property
    stddev : torch.Tensor or None, default=None
        standard deviation of property
    atomref : torch.Tensor or None, default=None
        reference single-atom properties. Expects
        an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
        property of element Z. The value of atomref[0] must be zero, as this
        corresponds to the reference property for for "mask" atoms.
    outnet : callable, default=None
        Network used for atomistic outputs. Takes schnetpack input
        dictionary as input. Output is not normalized. If set to None,
        a pyramidal network is generated automatically.
    """

    def __init__(
        self,
        n_in,
        n_out=1,
        aggregation_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=schnetpack.nn.activations.shifted_softplus,
        property="y",
        contributions=None,
        derivative=None,
        negative_dr=False,
        stress=None,
        create_graph=False,
        mean=None,
        stddev=None,
        atomref=None,
        outnet=None,
    ):
        super(Atomwise, self).__init__()

        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.contributions = contributions
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.stress = stress

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        # initialize single atom energies
        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(
                torch.from_numpy(atomref.astype(np.float32))
            )
        else:
            self.atomref = None

        # build output network
        if outnet is None:
            self.out_net = nn.Sequential(
                schnetpack.nn.base.GetItem("representation"),
                MLP(n_in, n_out, n_neurons, n_layers, activation),
            )
        else:
            self.out_net = outnet

        # build standardization layer
        self.standardize = schnetpack.nn.base.ScaleShift(mean, stddev)

        # build aggregation layer
        if aggregation_mode == "sum":
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=False)
        elif aggregation_mode == "avg":
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=True)
        elif aggregation_mode == "max":
            self.atom_pool = schnetpack.nn.base.MaxAggregate(axis=1)
        elif aggregation_mode == "softmax":
            self.atom_pool = schnetpack.nn.base.SoftmaxAggregate(axis=1)
        else:
            raise AtomwiseError(
                "{} is not a valid aggregation " "mode!".format(aggregation_mode)
            )

    def forward(self, inputs):
        """
        predicts atomwise property

        Parameters
        ----------
        inputs : torch.Tensor
            batch of input values.

        Returns
        -------
        dict
            prediction for property
            If contributions is not None additionally returns atom-wise contributions.
            If derivative is not None additionally returns derivative w.r.t. atom positions.
        """
        atomic_numbers = inputs[Properties.Z]
        atom_mask = inputs[Properties.atom_mask]

        # run prediction
        yi = self.out_net(inputs)
        yi = self.standardize(yi)

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0

        y = self.atom_pool(yi, atom_mask)

        # collect results
        result = {self.property: y}

        if self.contributions is not None:
            result[self.contributions] = yi

        create_graph = True if self.training else self.create_graph

        if self.derivative is not None:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                result[self.property],
                inputs[Properties.R],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            result[self.derivative] = sign * dy

        if self.stress is not None:
            cell = inputs[Properties.cell]
            # Compute derivative with respect to cell displacements
            stress = grad(
                result[self.property],
                inputs["displacement"],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            # Compute cell volume
            volume = torch.sum(
                cell[:, 0, :] * torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                dim=1,
                keepdim=True,
            )[..., None]
            # Finalize stress tensor
            result[self.stress] = stress / volume

        return result
