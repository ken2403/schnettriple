import numpy as np
import torch
from torch import nn


__all__ = ['AngularMapping', 'TripleMapping']


class AngularMapping(nn.Module):
    """

    Attributes
    ----------

    """
    def __init__(
        self,
        max_zeta=1,
        n_zeta=1,
    ):
        super(AngularMapping, self).__init__()
        self.max_zeta = max_zeta
        self.n_zeta = n_zeta

    def forward(self, r_ij, r_ik, r_jk, triple_masks=None):
        """
        Parameters
        ----------

        Returns
        -------

        """
        # calculate angular_fetures
        cos_theta = (torch.pow(r_ij, 2) + torch.pow(r_ik, 2) - torch.pow(r_jk, 2)) / (
            2.0 * r_ij * r_ik)
        # Required in order to catch NaNs during backprop
        if triple_masks is not None:
            cos_theta[triple_masks == 0] = 0.0

        zetas = np.logspase(0, stop=np.log2(self.max_zeta), num=self.n_zeta, base=2)
        angular_pos = [
            2 ** (1 - zeta) * ((1.0 - cos_theta) ** zeta).unsqueeze(-1)
            for zeta in zetas
        ]
        angular_neg = [
            2 ** (1 - zeta) * ((1.0 + cos_theta) ** zeta).unsqueeze(-1)
            for zeta in zetas
        ]

        return torch.cat(angular_pos + angular_neg, -1)


class TripleMapping(nn.Module):
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
    def __init__(
        self,
        max_zeta=1,
        n_zeta=1,
        crossterm=False
    ):
        super(TripleMapping, self).__init__()
        self.angular_filter = AngularMapping(max_zeta, n_zeta)
        self.crossterm = crossterm

    def forward(
        self, r_ij, r_ik, r_jk, f_ij, f_ik, f_jk=None, triple_masks=None
    ):
        """
        Parameters
        ----------

        Returns
        -------

        """
        n_batch, n_atoms, n_neighbors = r_ij.size()
        # calculate angular_filter
        angular_mapping = self.angular_filter(r_ij, r_ik, r_jk, triple_masks)

        # calculate radial_mapping
        radial_mapping = f_ij * f_ik
        if self.crossterm:
            if f_jk is None:
                raise TypeError("TripleMapping() missing 1 required positional argument: 'f_jk'")
            else:
                radial_mapping *= f_jk

        # combnation of angular and radial filter
        total_mapping = angular_mapping[:, :, :, :, None] * radial_mapping[:, :, :, None, :]
        # reshape (N_batch * N_atom * N_nbh * N_filter_features)
        total_mapping = total_mapping.reshape(n_batch, n_atoms, n_neighbors, -1)

        return total_mapping
