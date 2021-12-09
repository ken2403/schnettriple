import numpy as np
import torch
from torch import nn


__all__ = ["ThetaDistribution", "AngularDistribution"]


class ThetaDistribution(nn.Module):
    """
    The block of calculating the theta distribution used in AngularDistribution.

    Attributes
    ----------
    n_theta : int, default=10
        number of theta_s filter.
    zeta : float, default=8.0
        zeta value of angular filter.
    """

    def __init__(self, n_theta=10, zeta=8.0):
        super(ThetaDistribution, self).__init__()
        offset_theta = torch.linspace(0, np.pi, n_theta)
        # self.register_buffer("zeta", torch.FloatTensor([zeta]))
        # self.register_buffer("offset_theta", offset_theta)

        zeta = torch.tensor(zeta)
        self.register_buffer("zetas", zeta)

    def forward(self, cos_theta):
        """
        Compute theta distribution with some shifts of theta_s.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr_double :  Total number of neighbors of each atom
        Nbr_triple :  Total number of triple neighbors of each atom

        Parameters
        ----------
        cos_theta : torch.Tensor
            value of the cosine of the triples with (B x At x Nbr_triple) of shape.

        Returns
        -------
        theta_distribution : torch.Tensor
            theta distribution with (B x At x Nbr_triple x n_theta) of shape.
        """
        # # diff theta
        # diff_theta = (
        #     torch.arccos(cos_theta)[:, :, :, None]
        #     - self.offset_theta[None, None, None, :]
        # )
        # # calculate theta_filters
        # theta_distribution = 2 ** (1.0 - self.zeta) * torch.pow(
        #     1.0 + torch.cos(diff_theta), self.zeta
        # )

        # diff_cos = cos_theta[:, :, :, None] * torch.cos(
        #     self.offset_theta[None, None, None, :]
        # ) + torch.sqrt(1.0 - cos_theta ** 2 + 1.0e-7)[:, :, :, None] * torch.sin(
        #     self.offset_theta[None, None, None, :]
        # )
        # # calculate theta_filters
        # theta_distribution = 2 ** (1.0 - self.zeta) * torch.pow(
        #     1.0 + diff_cos, self.zeta
        # )

        # calculate theta_filters
        theta_pos = [
            2 ** (1 - zeta) * ((1.0 - cos_theta) ** zeta).unsqueeze(-1)
            for zeta in self.zetas
        ]
        theta_neg = [
            2 ** (1 - zeta) * ((1.0 + cos_theta) ** zeta).unsqueeze(-1)
            for zeta in self.zetas
        ]
        theta_distribution = torch.cat(theta_pos + theta_neg, -1)

        return theta_distribution


class AngularDistribution(nn.Module):
    """
    The block of calculating the angular distribution used in SchNetTriple module.

    Attributes
    ----------
    n_theta : int, default=10
        number of angular filter
    zeta : float, default=8.0
        zeta value of angular filter.

    References
    ----------
    .. [1] Justin S. Smith, Olexandr Isayev, Adrian E. Roitberg.
        ANI-1: An extensible neural network potential with DFT accuracy
        at force field computational cost.
        Chemical Science, 3192-3203. 2017.
    """

    def __init__(self, n_theta=10, zeta=8.0):
        super(AngularDistribution, self).__init__()
        self.theta_filter = ThetaDistribution(n_theta=n_theta, zeta=zeta)

    def forward(
        self,
        r_ij,
        r_ik,
        r_jk,
        f_ij,
        f_ik,
        triple_mask=None,
    ):
        """
        Parameters
        ----------
        r_ij : torch.Tensor
            distances between neighboring atoms from atom i to j with
            (B x At x Nbr_triple) shape.
        r_ik : torch.Tensor
            distances between neighboring atoms from atom i to k with
            (B x At x Nbr_triple) shape.
        r_jk : torch.Tensor
            distances between neighboring atoms from atom j to k with
            (B x At x Nbr_triple) shape.
        f_ij : torch.Tensor
            filtered distances of triples from atom i to j.
            (B x At x Nbr_triple x n_gaussian_double) of shape.
        f_ik : torch.Tensor
            filtered distances of triples from atom j to k.
            (B x At x Nbr_triple x n_gaussian_double) of shape.
        triple_mask : torch.Tensor or None, default=None
            mask to filter out non-existing neighbors introduced via padding.
            (B x At x Nbr_triple) of shape.

        Returns
        -------
        angular_distribution : torch.Tensor
            angular distribution with (B x At x Nbr_triple x n_angular_feature) of shape.
        """
        B, At, Nbr_triple = r_ij.size()
        # calculate radial_filter
        radial_filter = f_ij * f_ik

        # calculate theta_filter
        cos_theta = (torch.pow(r_ij, 2) + torch.pow(r_ik, 2) - torch.pow(r_jk, 2)) / (
            2.0 * r_ij * r_ik
        )
        if triple_mask is not None:
            cos_theta[triple_mask == 0] = 0.0
        angular_filter = self.theta_filter(cos_theta)

        if triple_mask is not None:
            radial_filter[triple_mask == 0] = 0.0
            angular_filter[triple_mask == 0] = 0.0

        # combination of angular and radial filter
        angular_distribution = (
            angular_filter[:, :, :, :, None] * radial_filter[:, :, :, None, :]
        )
        # reshape (B x At x Nbr x n_filter_feature)
        angular_distribution = angular_distribution.view(B, At, Nbr_triple, -1)

        return angular_distribution
