import numpy as np
import torch
from torch import nn


__all__ = ["CosineCutoff", "PolyCutoff"]


class CosineCutoff(nn.Module):
    """
    Class of Behler-Parrinero cosine cutoff.

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}
    
    Attributes
    ----------
    cutoff : float, default=5.0
        cutoff radius.
    
    References
    ----------
    .. [1] Jörg Behler and Michele Parrinello,
        Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces
        M. Phys. Rev. Lett. 2007, 98, 146401.
    """

    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        """
        Compute cutoff.

        Parameters
        ----------
        distances : torch.Tensor
            values of interatomic distances.

        Returns
        -------
        cutoffs : torch.Tensor
            values of cutoff function.
        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


class PolyCutoff(nn.Module):
    """
    Class of Polynomial cutoff function.

    Attributes
    ----------
    cutoff : float, default=5.0
        cutoff radius.

    References
    ----------
    .. [1] Ebert, D. S.; Musgrave, F. K.; Peachey, D.; Perlin, K.; Worley, S.
        Texturing & Modeling: A Procedural Approach;
        Morgan Kaufmann, 2003.
    """

    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        """
        Compute cutoff.

        Parameters
        ----------
        distances : torch.Tensor
            values of interatomic distances.

        Returns
        -------
        cutoffs : torch.Tensor
            values of cutoff function.
        """
        # Compute values of cutoff function
        cutoffs = (
            1.0
            - 6 * torch.pow((distances / self.cutoff), 5)
            + 15 * torch.pow((distances / self.cutoff), 4)
            - 10 * torch.pow((distances / self.cutoff), 3)
        )
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs
