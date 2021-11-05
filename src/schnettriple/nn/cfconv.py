import torch
from torch import nn
from schnetpack.nn import Dense
from schnetpack.nn.base import Aggregate

from schnettriple.nn.mapping import TripleMapping


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
        max_zeta : int

        n_zeta : int

        filter_network_double : nn.Module
            filter block for double propperties.
        filter_network_triple : nn.Module
            filter block for triple properties.
        cutoff_network : nn.Module, optional, default=None
            if None, no cut off function is used.
        activation : callable, optional, default=None
            if None, no activation function is used.
        crossterm : bool, default=False
            if True,
        normalize_filter : bool, optional, default=False
            If True, normalize filter to the number of
            neighbors when aggregating.
        axis : int, optional, default=2
            axis over which convolution should be applied.

    """

    def __init__(
        self,
        n_in,
        n_filters,
        n_out,
        max_zeta,
        n_zeta,
        filter_network_double,
        filter_network_triple,
        cutoff_network=None,
        activation=None,
        crossterm=False,
        normalize_filter=False,
        axis=2,
    ):
        super(CFConvTriple, self).__init__()
        self.in2f = Dense(n_in, n_filters * 2, bias=False, activation=None)
        self.f2out = Dense(n_filters * 2, n_out, bias=True, activation=activation)
        self.triple_mapping = TripleMapping(
            max_zeta=max_zeta, n_zeta=n_zeta, crossterm=crossterm
        )
        self.filter_network_double = filter_network_double
        self.filter_network_triple = filter_network_triple
        self.cutoff_network = cutoff_network
        self.crossterm = crossterm
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def forward(
        self,
        x,
        r_ij,
        r_ik,
        r_jk,
        neighbors_j,
        triple_masks,
        f_ij=None,
        f_ik=None,
        f_jk=None,
    ):
        """
        Compute convolution block.

        Parameters
        ----------
            x : torch.Tensor
                input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_ij: torch.Tensor
                interatomic distances from the centered atom i
                to the neighbor atom j of (N_b, N_a, N_nbh) shape.
            r_ik : torch.Tensor
                interatomic distances from the centered atom i
                to the neighbor atom k of (N_b, N_a, N_nbh) shape.
            r_jk: torch.Tensor, optional, default=None
                interatomic distances from the neighbor atom j
                to the neighbor atom k of (N_b, N_a, N_nbh) shape.
            neighbors_j : torch.Tensor
                of (N_b, N_a, N_nbh) shape.
            triple_masks : torch.Tensor
                mask to filter out non-existing neighbors
                introduced via padding.
            f_ij: torch.Tensor, optional, default=None
                expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.
            f_ik: torch.Tensor, optional, default=None
                expanded interatomic distances in a basis.
                If None, r_ik.unsqueeze(-1) is used.
            f_jk: torch.Tensor, optional, default=None
                expanded interatomic distances in a basis.
                If None, r_jk.unsqueeze(-1) is used.

        Returns
        -------
            torch.Tensor
                block output with (N_batch, N_atoms, N_out) shape.

        """
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1)
        if f_ik is None:
            f_ik = r_ik.unsqueeze(-1)
        if self.crossterm:
            if f_jk is None:
                f_jk = r_jk.unsqueeze(-1)

        # pass expanded interactomic distances through filter block (double)
        W_ij_double = self.filter_network_double(f_ij)
        # apply cutoff
        if self.cutoff_network is not None:
            C_ij = self.cutoff_network(r_ij)
            W_ij_double = W_ij_double * C_ij.unsqueeze(-1)

        # calculate triple mapping
        m_ijk = self.triple_mapping(r_ij, r_ik, r_jk, f_ij, f_ik, f_jk, triple_masks)
        # pass triple mapping through filter block (triple)
        W_ijk = self.filter_network_triple(m_ijk)
        # apply cutoff
        if self.cutoff_network is not None:
            C_ik = self.cutoff_network(r_ik)
            W_ijk = W_ijk * C_ij.unsqueeze(-1) * C_ik.unsqueeze(-1)
            if self.crossterm:
                C_jk = self.cutoff_network(r_jk)
                W_ijk *= C_jk.unsqueeze(-1)
        # concatinate double and triple filters
        W_total = torch.cat((W_ij_double, W_ijk), -1)

        # pass initial embeddings through Dense layer
        y = self.in2f(x)

        # reshape y for element-wise multiplication by W
        nbh_size = neighbors_j.size()
        nbh = neighbors_j.reshape(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))
        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        # element-wise multiplication, aggregating and Dense layer
        y = y * W_total
        y = self.agg(y, triple_masks)
        y = self.f2out(y)

        return y
