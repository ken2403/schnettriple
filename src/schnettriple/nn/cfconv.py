import torch
from torch import nn
from schnetpack.nn.base import Aggregate

from schnettriple.nn.base import Dense


__all__ = ["CFConvDouble", "CFConvTriple"]


class CFConvDouble(nn.Module):
    """
    Continuous-filter convolution block used in SchNetInteractionDouble module.

    Attributes
    ----------
    n_in : int
        number of input (i.e. atomic embedding) dimensions.
    n_filters : int
        number of filter dimensions.
    n_out : int
        number of output dimensions.
    filternet_double : nn.Module
        filter block for double propperties.
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
        filternet_double,
        activation=None,
        cutoffnet=None,
        normalize_filter=False,
    ):
        super(CFConvDouble, self).__init__()
        self.in2f = Dense(n_in, n_filters, bias=False, activation=None)
        self.f2out = Dense(n_filters, n_out, bias=True, activation=activation)
        self.filternet_double = filternet_double
        self.cutoffnet_double = cutoffnet
        self.agg = Aggregate(axis=2, mean=normalize_filter)

    def forward(
        self,
        x,
        r_double,
        f_double,
        neighbors,
        neighbor_mask,
    ):
        """
        Compute convolution block.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr_double :  Total number of neighbors of each atom

        Parameters
        ----------
        x : torch.Tensor
            input representation/embedding of atomic environments with (B x At x n_in) shape.
        f_double : torch.tensor
            filtered distances of double pairs with
            (B x At x Nbr_double x n_gaussian_double) shape.
        neighbors : torch.Tensor
            indices of neighboring atoms with (B x At x Nbr_double) shape.
        neighbor_mask : torch.Tensor
            mask to filter out non-existing neighbors introduced via padding.
            (B x At x Nbr_double) of shape.

        Returns
        -------
        y : torch.Tensor
            block output with (B x At x n_out) shape.
        """
        # pass triple distribution through filter block (triple)
        W_double = self.filternet_double(f_double)
        if self.cutoffnet_double is not None:
            C_double = self.cutoffnet_double(r_double)
            W_double = W_double * C_double.unsqueeze(-1)

        # pass initial embeddings through Dense layer
        y = self.in2f(x)

        # reshape y for element-wise multiplication by W
        B, At, Nbr_double = neighbors.size()
        nbh = neighbors.reshape(-1, At * Nbr_double, 1)
        nbh = nbh.expand(-1, -1, y.size(2))
        # get j neighbors of centered atom i.
        y_double = torch.gather(y, 1, nbh)
        y_double = y_double.view(B, At, Nbr_double, -1)

        # element-wise multiplication, aggregating and Dense layer
        y_double = y_double * W_double
        y_double = self.agg(y_double, neighbor_mask)
        # # residual net
        # y_double = y_double + y

        # output embbedings through Dense layer
        y_double = self.f2out(y_double)

        return y_double


class CFConvTriple(nn.Module):
    """
    Continuous-filter convolution block used in SchNetInteractionTriple module.

    Attributes
    ----------
    n_in : int
        number of input (i.e. atomic embedding) dimensions.
    n_filters : int
        number of filter dimensions.
    n_out : int
        number of output dimensions.
    filternet_triple : nn.Module
        filter block for triple properties.
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
        filternet_triple,
        activation=None,
        cutoffnet=None,
        normalize_filter=False,
    ):
        super(CFConvTriple, self).__init__()
        self.in2f = Dense(n_in, n_filters, bias=False, activation=None)
        self.f2out = Dense(n_filters, n_out, bias=True, activation=activation)
        self.filternet_triple = filternet_triple
        self.cutoffnet_triple = cutoffnet
        self.agg = Aggregate(axis=2, mean=normalize_filter)

    def forward(
        self,
        x,
        r_ij,
        r_ik,
        triple_ijk,
        neighbors_j,
        neighbors_k,
        triple_mask,
    ):
        """
        Compute convolution block.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr_triple :  Total number of triple neighbors of each atom

        Parameters
        ----------
        x : torch.Tensor
            input representation/embedding of atomic environments with (B x At x n_in) shape.
        triple_ijk : torch.tensor
            combination of filtered distances and angular filters with
            (B x At x Nbr_triple x n_angular) shape.
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
        # pass triple distribution through filter block (triple)
        W_triple = self.filternet_triple(triple_ijk)
        if self.cutoffnet_triple is not None:
            C_ij = self.cutoffnet_triple(r_ij)
            C_ik = self.cutoffnet_triple(r_ik)
            W_triple = W_triple * C_ij.unsqueeze(-1) * C_ik.unsqueeze(-1)

        # pass initial embeddings through Dense layer
        y = self.in2f(x)

        # reshape y for element-wise multiplication by W
        B, At, Nbr_tirple = neighbors_j.size()
        nbh_j = neighbors_j.reshape(-1, At * Nbr_tirple, 1)
        nbh_j = nbh_j.expand(-1, -1, y.size(2))

        nbh_k = neighbors_k.reshape(-1, At * Nbr_tirple, 1)
        nbh_k = nbh_k.expand(-1, -1, y.size(2))
        # get j and k neighbors of centered atom i. Add these atomic embeddings.
        y_triple = torch.gather(y, 1, nbh_j) + torch.gather(y, 1, nbh_k)
        y_triple = y_triple.view(B, At, Nbr_tirple, -1)

        # element-wise multiplication, aggregating and Dense layer
        y_triple = y_triple * W_triple
        y_triple = self.agg(y_triple, triple_mask)
        # # residual net
        # y_triple = y_triple + y

        # output embbedings through Dense layer
        y_triple = self.f2out(y_triple)

        return y_triple
