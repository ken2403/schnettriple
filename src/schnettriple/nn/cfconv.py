import torch
from torch import nn
from schnetpack.nn.base import Aggregate

from schnettriple.nn.base import Dense


__all__ = ["CFConvTriple"]


class CFConvTriple(nn.Module):
    """
    Continuous-filter convolution block used in SchNetTriple module.

    Attributes
    ----------
    n_in : int
        number of input (i.e. atomic embedding) dimensions.
    n_filters : int
        number of filter dimensions.
    n_out : int
        number of output dimensions.
    filter_network_double : nn.Module
        filter block for double propperties.
    filter_network_triple : nn.Module
        filter block for triple properties.
    cutoff_network : nn.Module, default=None
        if None, no cut off function is used.
    activation : callable, default=None
        if None, no activation function is used.
    normalize_filter : bool, default=False
        If True, normalize filter to the number of
        neighbors when aggregating.
    """

    def __init__(
        self,
        n_in,
        n_filters,
        n_out,
        filter_network_double,
        filter_network_triple,
        cutoff_network=None,
        activation=None,
        normalize_filter=False,
    ):
        super(CFConvTriple, self).__init__()
        self.in2f = Dense(n_in, n_filters, bias=False, activation=None)
        self.f2out = Dense(2 * n_filters, n_out, bias=True, activation=activation)
        self.filter_network_double = filter_network_double
        self.filter_network_triple = filter_network_triple
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=2, mean=normalize_filter)

    def forward(
        self,
        x,
        r_double,
        f_double,
        r_ij,
        r_ik,
        triple_ijk,
        neighbors,
        neighbor_mask,
        neighbors_j,
        neighbors_k,
        triple_mask,
    ):
        """
        Compute convolution block.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr_double :  Total number of neighbors of each atom
        Nbr_triple :  Total number of triple neighbors of each atom

        Parameters
        ----------
        x : torch.Tensor
            input representation/embedding of atomic environments with (B x At x n_in) shape.
        r_double : torch.Tensor
            distances between neighboring atoms with
            (B x At x Nbr_double) shape.
        f_double : torch.tensor
            filtered distances of double pairs with
            (B x At x Nbr_double x n_gaussian_double) shape.
        r_ij : torch.Tensor
            distance between central atom and neighbor j with
            (B x At x Nbr_triple) shape.
        r_ik : torch.Tensor
            distance between central atom and neighbor k with
            (B x At x Nbr_triple) shape.
        triple_ijk : torch.tensor
            combination of filtered distances and angular filters with
            (B x At x Nbr_triple x n_angular) shape.
        neighbors : torch.Tensor
            indices of neighboring atoms with (B x At x Nbr_double) shape.
        neighbor_mask : torch.Tensor
            mask to filter out non-existing neighbors introduced via padding.
            (B x At x Nbr_double) of shape.
        neighbors_j : torch.Tensor
            indices of atom j in tirples with (B x At x Nbr_triple) shape.
        neighbors_k : torch.Tensor
            indices of atom k in tirples with (B x At x Nbr_triple) shape.
        triple_mask : torch.Tensor
            mask to filter out non-existing neighbors introduced via padding.
            (B x At x Nbr_triple) of shape.

        Returns
        -------
        y : torch.Tensor
            block output with (B x At x n_out) shape.
        """
        # pass expanded interactomic distances through filter block (double)
        W_double = self.filter_network_double(f_double)
        # apply cutoff
        if self.cutoff_network is not None:
            C_double = self.cutoff_network(r_double)
            W_double = W_double * C_double.unsqueeze(-1)

        # pass triple distribution through filter block (triple)
        W_triple = self.filter_network_triple(triple_ijk)
        # apply cutoff
        if self.cutoff_network is not None:
            C_ij = self.cutoff_network(r_ij)
            C_ik = self.cutoff_network(r_ik)
            W_triple = W_triple * C_ij.unsqueeze(-1) * C_ik.unsqueeze(-1)

        # pass initial embeddings through Dense layer
        y = self.in2f(x)

        # reshape y for element-wise multiplication by W
        B, At, Nbr_double = neighbors.size()
        nbh = neighbors.reshape(-1, At * Nbr_double, 1)
        nbh = nbh.expand(-1, -1, y.size(2))
        # get atom embedding of neighbors of i.
        y_double = torch.gather(y, 1, nbh)
        y_double = y_double.view(B, At, Nbr_double, -1)

        # element-wise multiplication, aggregating and Dense layer
        y_double = y_double * W_double
        y_double = self.agg(y_double, neighbor_mask)

        # reshape y for element-wise multiplication by W
        _, _, Nbr_tirple = neighbors_j.size()
        nbh_j = neighbors_j.reshape(-1, At * Nbr_tirple, 1)
        nbh_j = nbh_j.expand(-1, -1, y.size(2))
        # r_ij = r_ij.view(-1, At * Nbr_tirple, 1)

        nbh_k = neighbors_k.reshape(-1, At * Nbr_tirple, 1)
        nbh_k = nbh_k.expand(-1, -1, y.size(2))
        # r_ik = r_ik.view(-1, At * Nbr_tirple, 1)
        # get j and k neighbors of i. Add these atomic embeddings.
        y_triple_j = torch.gather(y, 1, nbh_j)
        y_triple_k = torch.gather(y, 1, nbh_k)
        y_triple = y_triple_j + y_triple_k
        y_triple = y_triple.view(B, At, Nbr_tirple, -1)

        # element-wise multiplication, aggregating and Dense layer
        y_triple = y_triple * W_triple
        y_triple = self.agg(y_triple, triple_mask)

        # concatinate double and triple embeddings
        y_total = torch.cat((y_double, y_triple), dim=2)
        # output embbedings through Dense layer
        y_total = self.f2out(y_total)

        return y
