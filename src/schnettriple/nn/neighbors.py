import torch
from torch import nn


__all__ = ["TriplesDistances"]


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

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions
    neighbors_j : torch.Tensor
        Indices of first neighbor in triangle
    neighbors_k : torch.Tensor
        Indices of second neighbor in triangle
    offset_idx_j : torch.Tensor
        Indices for offets of neighbors j (for PBC)
    offset_idx_k : torch.Tensor
        Indices for offets of neighbors k (for PBC)
    cell : torch.tensor, optional
        periodic cell of (N_b x 3 x 3) shape.
    cell_offsets : torch.Tensor, optional
        offset of atom in cell coordinates with (N_b x N_at x N_nbh x 3) shape.

    Returns
    -------
    torch.Tensor
        Distance between central atom and neighbor j
    torch.Tensor
        Distance between central atom and neighbor k
    torch.Tensor
            Distance between neighbors

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
            Atomic positions
        neighbors_j : torch.Tensor
            Indices of first neighbor in triangle
        neighbors_k : torch.Tensor
            Indices of second neighbor in triangle
        offset_idx_j : torch.Tensor
            Indices for offets of neighbors j (for PBC)
        offset_idx_k : torch.Tensor
            Indices for offets of neighbors k (for PBC)
        cell : torch.tensor, optional
            periodic cell of (N_b x 3 x 3) shape.
        cell_offsets : torch.Tensor, optional
            offset of atom in cell coordinates with (N_b x N_at x N_nbh x 3) shape.

        Returns
        -------
        torch.Tensor
            Distance between central atom and neighbor j
        torch.Tensor
            Distance between central atom and neighbor k
        torch.Tensor
            Distance between neighbors

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
