import torch
from torch import nn
from schnetpack.nn import Dense
from schnetpack.nn.base import Aggregate


__all__ = ['CFConvTriple']


class CFConvTriple(nn.Module):
    """
    Continuous-filter convolution block used in SchNet module.

    Attributes
    ----------
        n_in : int
            number of input (i.e. atomic embedding) dimensions.
        n_filters : int
            number of filter dimensions.
        n_out : int
            number of output dimensions.
        filter_network : nn.Module
            filter block.
        cutoff_network : nn.Module, optional, default=None
            if None, no cut off function is used.
        activation : callable, optional, default=None
            if None, no activation function is used.
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
        filter_network,
        cutoff_network=None,
        activation=None,
        normalize_filter=False,
        axis=2,
    ):
        super(CFConvTriple, self).__init__()
        self.in2f = Dense(n_in, n_filters, bias=False, activation=None)
        self.f2out = Dense(n_filters, n_out, bias=True, activation=activation)
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def forward(
        self, x, r_ij, r_ik, r_jk, neighbors_j, neighbors_k
        triple_mask, f_ij=None, f_ik=None, f_jk=None):
        """
        Compute convolution block.

        Parameters
        ----------
            x : torch.Tensor
                input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_ij : torch.Tensor
                interatomic distances from the centered atom i
                to the neighbor atom j of (N_b, N_a, N_nbh) shape.
            r_ik : torch.Tensor
                interatomic distances from the centered atom i
                to the neighbor atom k of (N_b, N_a, N_nbh) shape.
            r_jk : torch.Tensor
                interatomic distances from the neighbor atom j
                to  the neighbor atom k of (N_b, N_a, N_nbh) shape.
            neighbors_j : torch.Tensor
                of (N_b, N_a, N_nbh) shape.
            neighbors_k : torch.Tensor

            triple_mask : torch.Tensor
                mask to filter out non-existing neighbors
                introduced via padding.
            f_ij : torch.Tensor, optional, default=None
                expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.
            f_ik : torch.Tensor, optional, default=None
                expanded interatomic distances in a basis.
                If None, r_ik.unsqueeze(-1) is used.
            f_jk : torch.Tensor, optional, default=None
                expanded interatomic distances in a basis.
                If None, r_jk.unsqueeze(-1) is used.

        Returns
        -------
            torch.Tensor
                block output with (N_b, N_a, n_out) shape.

        """
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1)

        if f_ik is None:
            f_ik = r_ik.unsqueeze(-1)

        if f_jk is None:
            f_jk = r_jk.unsqueeze(-1)

        # pass expanded interactomic distances through filter block
        W_ij = self.filter_network(f_ij)
        # apply cutoff
        if self.cutoff_network is not None:
            C_ij = self.cutoff_network(r_ij)
            W_ij = W_ij * C_ij.unsqueeze(-1)

        # pass expanded interactomic distances through filter block
        W_ik = self.filter_network(f_ik)
        # apply cutoff
        if self.cutoff_network is not None:
            C_ik = self.cutoff_network(r_ik)
            W_ik = W_ik * C_ik.unsqueeze(-1)

        # pass expanded interactomic distances through filter block
        W_jk = self.filter_network(f_jk)
        # apply cutoff
        if self.cutoff_network is not None:
            # Whether cutoff should be applied or not
            C_jk = self.cutoff_network(r_jk)
            W_jk = W_jk * C_jk.unsqueeze(-1)

        # pass initial embeddings through Dense layer
        y = self.in2f(x)

        # reshape y for element-wise multiplication by W
        nbh_size = neighbors_j.size()
        nbh = neighbors_j.reshape(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))
        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        # element-wise multiplication, aggregating and Dense layer
        y = y * W_ij * W_ik * W_jk    # !!!my modify!!!
        y = self.agg(y, triple_mask)
        y = self.f2out(y)

        return y
