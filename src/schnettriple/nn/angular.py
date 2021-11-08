from math import cos
import numpy as np
import torch
from torch import nn


__all__ = ["ThetaDistribution", "TripleDistribution"]


class ThetaDistribution(nn.Module):
    """

    Attributes
    ----------

    """

    def __init__(self, max_zeta=1, n_zeta=1):
        super(ThetaDistribution, self).__init__()
        zetas = torch.logspace(0, end=np.log2(max_zeta), steps=n_zeta, base=2)
        self.register_buffer("zetas", zetas)

    def forward(self, cos_theta):
        """
        Parameters
        ----------

        Returns
        -------

        """
        # calculate theta_filters
        angular_pos = [
            2 ** (1 - zeta) * ((1.0 - cos_theta) ** zeta).unsqueeze(-1)
            for zeta in self.zetas
        ]
        angular_neg = [
            2 ** (1 - zeta) * ((1.0 + cos_theta) ** zeta).unsqueeze(-1)
            for zeta in self.zetas
        ]
        ang_total = torch.cat(angular_pos + angular_neg, -1)

        return ang_total


class TripleDistribution(nn.Module):
    """
    入力：距離(N_b*N_atom*N_nbh)とGaussianSmearing (N_b*N_atom*N_nbh*N_gauss)
    出力：triple filter(カットオフ適用前、doubleのカットオフ適用前と同じところまで)　(N_b*N_atom*N_nbh*N_filter)

    やること：
        距離から角度計算(N_b*N_atom*N_nbh)→(N_b*N_atom*N_nbh)
        zetaの生成(np.logspase)
        zeta分のfilter生成(N_b*N_atom*N_nbh)→(N_b*N_atom*N_nbh＊N_zeta) →AngularMappingで実装

        GaussianSmearingとうまく組み合わせる→(N_b*N_atom*N_nbh＊N_gaussとN_zetaの組み合わせ)
        これをfiler生成NNに流す、Dense2つ→(N_b*N_atom*N_nbh＊N_filter) → CFconvで実装
    Attributes
    ----------

    """

    def __init__(self, max_zeta=1, n_zeta=1, crossterm=False):
        super(TripleDistribution, self).__init__()
        self.theta_filter = ThetaDistribution(max_zeta, n_zeta)
        self.crossterm = crossterm

    def forward(
        self,
        r_ij,
        r_ik,
        r_jk,
        f_ij,
        f_ik,
        f_jk=None,
        triple_masks=None,
    ):
        """
        Parameters
        ----------

        Returns
        -------

        """
        n_batch, n_atoms, n_neighbors = r_ij.size()
        # calculate radial_filter
        radial_filter = f_ij * f_ik
        if self.crossterm:
            if f_jk is None:
                raise TypeError(
                    "TripleDistribution() missing 1 required positional argument: 'f_jk'"
                )
            else:
                radial_filter = radial_filter * f_jk

        radial_filter = radial_filter + r_ij.unsqueeze(-1) + r_ij.unsqueeze(-1)
        if self.crossterm:
            radial_filter = radial_filter + r_jk.unsqueeze(-1)

        # calculate theta_filter
        cos_theta = (torch.pow(r_ij, 2) + torch.pow(r_ik, 2) - torch.pow(r_jk, 2)) / (
            2.0 * r_ij * r_ik
        )
        if triple_masks is not None:
            cos_theta[triple_masks == 0] = 0.0
        angular_filter = self.theta_filter(cos_theta)

        if triple_masks is not None:
            radial_filter[triple_masks == 0] = 0.0
            angular_filter[triple_masks == 0] = 0.0

        # combnation of angular and radial filter
        triple_distribution = (
            angular_filter[:, :, :, :, None] * radial_filter[:, :, :, None, :]
        )
        # reshape (N_batch * N_atom * N_nbh * N_filter_features)
        triple_distribution = triple_distribution.view(
            n_batch, n_atoms, n_neighbors, -1
        )

        return triple_distribution
