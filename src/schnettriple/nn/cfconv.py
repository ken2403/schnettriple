import torch
from torch import nn
from schnetpack.nn import Dense
from schnetpack.nn.base import Aggregate


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
        cutoff_network : nn.Module, optional, default=None
            if None, no cut off function is used.
        activation : callable, optional, default=None
            if None, no activation function is used.
        normalize_filter : bool, optional, default=False
            If True, normalize filter to the number of
            neighbors when aggregating.
        crossterm : bool
            if True,
        axis : int, optional, default=2
            axis over which convolution should be applied.

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
        crossterm=False,
        axis=2,
    ):
        super(CFConvTriple, self).__init__()
        self.in2f = Dense(n_in, n_filters, bias=False, activation=None)
        self.f2out = Dense(n_filters, n_out, bias=True, activation=activation)
        self.filter_network_double = filter_network_double
        self.filter_network_triple = filter_network_triple
        self.cutoff_network = cutoff_network
        self.crossterm = crossterm
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def forward(
        self,
        x,
        r_double,
        r_ij,
        r_ik,
        r_jk,
        neighbors,
        neighbor_mask,
        neighbors_j,
        neighbors_k,
        triple_masks,
        d_ijk,
        f_double=None,
    ):
        """
        Compute convolution block.

        Parameters
        ----------
            x : torch.Tensor
                input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_double :

            r_ij :

            r_ik :

            r_jk :

            neighbors :

            neighbor_mask :

            neighbors_j : torch.Tensor
                of (N_b, N_a, N_nbh) shape.
            neighbors_k : torch.Tensor
                of (N_b, N_a, N_nbh) shape.
            triple_masks : torch.Tensor
                mask to filter out non-existing neighbors
                introduced via padding.
            d_ijk : torch.tensor

            f_double : torch.tensor


        Returns
        -------
            torch.Tensor
                block output with (N_batch, N_atoms, N_out) shape.

        """
        if f_double is None:
            f_double = r_double.unsqueeze(-1)

        # pass expanded interactomic distances through filter block (double)
        W_double = self.filter_network_double(f_double)
        # apply cutoff
        if self.cutoff_network is not None:
            C_double = self.cutoff_network(r_double)
            W_double = W_double * C_double.unsqueeze(-1)

        # pass triple distribution through filter block (triple)
        W_triple = self.filter_network_triple(d_ijk)

        # apply cutoff
        if self.cutoff_network is not None:
            C_ij = self.cutoff_network(r_ij)
            C_ik = self.cutoff_network(r_ik)
            W_triple = W_triple * C_ij.unsqueeze(-1) * C_ik.unsqueeze(-1)
            if self.crossterm:
                C_jk = self.cutoff_network(r_jk)
                W_triple = W_triple * C_jk.unsqueeze(-1)

        # pass initial embeddings through Dense layer
        y = self.in2f(x)

        # # reshape y for element-wise multiplication by W
        # nbh_size = neighbors.size()
        # nbh = neighbors.reshape(-1, nbh_size[1] * nbh_size[2], 1)
        # nbh = nbh.expand(-1, -1, y.size(2))
        # y = torch.gather(y, 1, nbh)
        # y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        # # element-wise multiplication, aggregating and Dense layer
        # y = y * W_double
        # y = self.agg(y, neighbor_mask)

        # reshape y for element-wise multiplication by W
        nbh_j_size = neighbors_j.size()
        nbh_j = neighbors_j.reshape(-1, nbh_j_size[1] * nbh_j_size[2], 1)
        nbh_j = nbh_j.expand(-1, -1, y.size(2))
        r_ij_nbh = r_ij.reshape(-1, nbh_j_size[1] * nbh_j_size[2], 1)
        r_ij_nbh = r_ij_nbh.expand(-1, -1, y.size(2))
        nbh_k_size = neighbors_k.size()
        nbh_k = neighbors_k.reshape(-1, nbh_k_size[1] * nbh_k_size[2], 1)
        nbh_k = nbh_k.expand(-1, -1, y.size(2))
        r_ik_nbh = r_ik.reshape(-1, nbh_j_size[1] * nbh_j_size[2], 1)
        r_ik_nbh = r_ik_nbh.expand(-1, -1, y.size(2))
        y = (
            r_ij_nbh * torch.gather(y, 1, nbh_j) + r_ik_nbh * torch.gather(y, 1, nbh_k)
        ) / (r_ij_nbh + r_ik_nbh)
        y = y.view(nbh_j_size[0], nbh_j_size[1], nbh_j_size[2], -1)

        # element-wise multiplication, aggregating and Dense layer
        y = y * W_triple
        y = self.agg(y, triple_masks)

        # output embbedings through Dense layer
        y = self.f2out(y)

        return y
