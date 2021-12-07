import torch
from torch import Tensor
from torch import nn


__all__ = ["TriplesDistances", "GaussianFilter"]


def triple_distances(
    positions,
    neighbors_j,
    neighbors_k,
    offset_idx_j=None,
    offset_idx_k=None,
    cell=None,
    cell_offsets=None,
):
    """
    Get all distances between atoms forming a triangle with the central atoms.
    Required e.g. for angular symmetry functions.

    B   :  Batch size
    At  :  Total number of atoms in the batch
    Nbr_double :  Total number of neighbors of each atom
    Nbr_triple :  Total number of triple neighbors of each atom

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions with (B x At x 3) shape.
    neighbors_j : torch.Tensor
        Indices of first neighbor in triangle with (Bx At x Nbr_triple) shape.
    neighbors_k : torch.Tensor
        Indices of second neighbor in triangle with (Bx At x Nbr_triple) shape.
    offset_idx_j : torch.Tensor
        Indices for offets of neighbors j (for PBC) with (Bx At x Nbr_triple x 3) shape.
    offset_idx_k : torch.Tensor
        Indices for offets of neighbors k (for PBC) with (Bx At x Nbr_triple x 3) shape.
    cell : torch.tensor, optional
        periodic cell of (B x 3 x 3) shape.
    cell_offsets : torch.Tensor, optional
        offset of atom in cell coordinates with (Bx At x Nbr_triple x 3) shape.

    Returns
    -------
    r_ij : torch.Tensor
        Distance between central atom and neighbor j
        with (Bx At x Nbr_triple) shape.
    r_ik : torch.Tensor
        Distance between central atom and neighbor k
        with (Bx At x Nbr_triple) shape.
    r_jk : torch.Tensor
        Distance between neighbors j and k
        with (Bx At x Nbr_triple) shape.
    """
    nbatch, _, _ = neighbors_k.size()
    idx_m = torch.arange(nbatch, device=positions.device, dtype=torch.long)[
        :, None, None
    ]

    pos_j = positions[idx_m, neighbors_j[:], :]
    pos_k = positions[idx_m, neighbors_k[:], :]

    if cell is not None:
        # Get the offsets into true cartesian values
        B, A, N, D = cell_offsets.size()

        cell_offsets = cell_offsets.view(B, A * N, D)
        offsets = cell_offsets.bmm(cell)
        offsets = offsets.view(B, A, N, D)

        # Get the offset values for j and k atoms
        B, A, T = offset_idx_j.size()

        # Collapse batch and atoms position for easier indexing
        offset_idx_j = offset_idx_j.view(B * A, T)
        offset_idx_k = offset_idx_k.view(B * A, T)
        offsets = offsets.view(B * A, -1, D)

        # Construct auxiliary aray for advanced indexing
        idx_offset_m = torch.arange(B * A, device=positions.device, dtype=torch.long)[
            :, None
        ]

        # Restore proper dmensions
        offset_j = offsets[idx_offset_m, offset_idx_j[:]].view(B, A, T, D)
        offset_k = offsets[idx_offset_m, offset_idx_k[:]].view(B, A, T, D)

        # Add offsets
        pos_j = pos_j + offset_j
        pos_k = pos_k + offset_k

    # if positions.is_cuda:
    #    idx_m = idx_m.pin_memory().cuda(async=True)

    # Get the real positions of j and k
    R_ij = pos_j - positions[:, :, None, :]
    R_ik = pos_k - positions[:, :, None, :]
    R_jk = pos_j - pos_k

    # + 1e-9 to avoid division by zero
    r_ij = torch.norm(R_ij, 2, 3) + 1e-9
    r_ik = torch.norm(R_ik, 2, 3) + 1e-9
    r_jk = torch.norm(R_jk, 2, 3) + 1e-9

    return r_ij, r_ik, r_jk


class TriplesDistances(nn.Module):
    """
    Layer that gets all distances between atoms forming a triangle with the
    central atoms. Required e.g. for angular symmetry functions.
    """

    def __init__(self):
        super(TriplesDistances, self).__init__()

    def forward(
        self,
        positions,
        neighbors_j,
        neighbors_k,
        offset_idx_j=None,
        offset_idx_k=None,
        cell=None,
        cell_offsets=None,
    ):
        """
        Parameters
        ----------
        positions : torch.Tensor
            Atomic positions with (B x At x 3) shape.
        neighbors_j : torch.Tensor
            Indices of first neighbor in triangle with (Bx At x Nbr_triple) shape.
        neighbors_k : torch.Tensor
            Indices of second neighbor in triangle with (Bx At x Nbr_triple) shape.
        offset_idx_j : torch.Tensor
            Indices for offets of neighbors j (for PBC) with (Bx At x Nbr_triple x 3) shape.
        offset_idx_k : torch.Tensor
            Indices for offets of neighbors k (for PBC) with (Bx At x Nbr_triple x 3) shape.
        cell : torch.tensor, optional
            periodic cell of (B x 3 x 3) shape.
        cell_offsets : torch.Tensor, optional
            offset of atom in cell coordinates with (Bx At x Nbr_triple x 3) shape.

        Returns
        -------
        r_ij : torch.Tensor
            Distance between central atom and neighbor j
            with (Bx At x Nbr_triple) shape.
        r_ik : torch.Tensor
            Distance between central atom and neighbor k
            with (Bx At x Nbr_triple) shape.
        r_jk : torch.Tensor
            Distance between neighbors j and k
            with (Bx At x Nbr_triple) shape.
        """
        return triple_distances(
            positions,
            neighbors_j,
            neighbors_k,
            offset_idx_j,
            offset_idx_k,
            cell,
            cell_offsets,
        )


def gaussian_filter(distances, offsets, widths, centered=False):
    """
    Filtered interatomic distance values using Gaussian functions.

    B   :  Batch size
    At  :  Total number of atoms in the batch
    Nbr :  Total number of neighbors of each atom
    G   :  Filtered features

    Parameters
    ----------
    distances : torch.Tensor
        interatomic distances of (B x At x Nbr) shape.
    offsets : torch.Tensor
        offsets values of Gaussian functions.
    widths : torch.Tensor
        width values of Gaussian functions.
    centered : bool, default=False
        If True, Gaussians are centered at the origin and the offsets are used
        to as their widths.

    Returns
    -------
    filtered_distances : torch.Tensor
        filtered distances of (B x At x Nbr x G) shape.

    References
    ----------
    .. [1] https://github.com/atomistic-machine-learning/schnetpack/blob/67226795af55719a7e4565ed773881841a94d130/src/schnetpack/nn/acsf.py
    """
    if centered:
        # if Gaussian functions are centered, use offsets to compute widths
        eta = 0.5 / torch.pow(offsets, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[:, :, :, None]

    else:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        eta = 0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[:, :, :, None] - offsets[None, None, None, :]

    # compute smear distance values
    filtered_distances = torch.exp(-eta * torch.pow(diff, 2))
    return filtered_distances


class GaussianFilter(nn.Module):
    """
    From inter-atomic distaces, calculates the filtered distances.

    Attributes
    ----------
    start : float, default=0.5
        center of first Gaussian function, :math:`\mu_0`.
    stop : float, default=6.0
        center of last Gaussian function, :math:`\mu_{N_g}`
    n_gaussians : int, default=100
        total number of Gaussian functions, :math:`N_g`.
    centered : bool, default=False
        If False, Gaussian's centered values are varied at the offset values and the width value is constant.
    """

    def __init__(
        self,
        start: float = 0.5,
        stop: float = 6.0,
        n_gaussian: int = 100,
        centered: bool = False,
    ) -> None:
        super().__init__()
        offsets = torch.linspace(start=start, end=stop, steps=n_gaussian)
        widths = torch.FloatTensor((offsets[1] - offsets[0]) * torch.ones_like(offsets))
        self.register_buffer("offset", offsets, persistent=True)
        self.register_buffer("width", widths, persistent=True)
        self.centered = centered

    def forward(self, distances: Tensor) -> Tensor:
        """
        Compute filtered distance values with Gaussian filter.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr :  Total number of neighbors of each atom
        G   :  Filtered features (n_gaussian)

        Parameters
        ----------
        distances : torch.Tensor
            interatomic distance values of (B x At x Nbr) shape.

        Returns
        -------
        filtered_distances : torch.Tensor
            filtered distances of (B x At x Nbr x G) shape.
        """
        return gaussian_filter(
            distances, offsets=self.offset, widths=self.width, centered=self.centered
        )
