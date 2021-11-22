import os
import logging
import torch
from torch import nn
from schnetpack.nn import Dense
from schnetpack import Properties
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.nn.acsf import GaussianSmearing
from schnetpack.nn.activations import shifted_softplus
from schnetpack.nn.neighbors import AtomDistances

from schnettriple.nn.neighbors import TriplesDistances
from schnettriple.nn.cfconv import CFConvTriple
from schnettriple.nn.angular import AngularDistribution


__all__ = ["SchNetInteractionTriple", "SchNetTriple"]


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


class SchNetInteractionTriple(nn.Module):
    """
    SchNet interaction block for modeling interactions of atomistic systems.

    Attributes
    ----------
        n_atom_basis : int
            number of features to describe atomic environments.
        n_spatial_basis : int
            number of input features of filter-generating networks.
        n_zeta : int

        n_filters : int
            number of filters used in continuous-filter convolution.
        cutoff : float
            cutoff radius.
        cutoff_network : nn.Module, default=schnetpack.nn.CosineCutoff
            cutoff layer.
        normalize_filter : bool, default=False
            if True, divide aggregated filter by number
            of neighbors over which convolution is applied.

    """

    def __init__(
        self,
        n_atom_basis,
        n_spatial_basis,
        n_zeta,
        n_filters,
        cutoff,
        cutoff_network=CosineCutoff,
        normalize_filter=False,
    ):
        super(SchNetInteractionTriple, self).__init__()
        # filter block used in interaction block
        self.filter_network_double = nn.Sequential(
            Dense(n_spatial_basis, n_filters, activation=shifted_softplus),
            Dense(n_filters, n_filters, activation=shifted_softplus),
        )
        # fiter block for triple
        self.filter_network_triple = nn.Sequential(
            Dense(n_spatial_basis * n_zeta * 2, n_filters, activation=shifted_softplus),
            Dense(n_filters, n_filters, activation=shifted_softplus),
        )
        # cutoff layer used in interaction block
        self.cutoff_network = cutoff_network(cutoff)
        # interaction block
        self.cfconv = CFConvTriple(
            n_atom_basis,
            n_filters,
            n_atom_basis,
            self.filter_network_double,
            self.filter_network_triple,
            cutoff_network=self.cutoff_network,
            activation=shifted_softplus,
            normalize_filter=normalize_filter,
        )
        # dense layer
        self.dense = Dense(n_atom_basis, n_atom_basis, bias=True, activation=None)

    def forward(
        self,
        x,
        r_double,
        r_ij,
        r_jk,
        neighbors,
        neighbor_mask,
        neighbors_j,
        neighbors_k,
        triple_mask,
        d_ijk,
        f_double=None,
    ):
        """
        Compute interaction output.

        Parameters
        ----------
        x : torch.Tensor
            input representation/embedding of atomic environments
            with (N_b, N_a, n_atom_basis) shape.
        r_double : torch.tensor

        r_ij : torch.tensor

        r_jk : torch.tensor

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

        f_double : torch.Tensor

        Returns
        -------
        torch.Tensor
            block output with (N_b, N_a, n_atom_basis) shape.

        """
        # continuous-filter convolution interaction block followed by Dense
        # layer
        v = self.cfconv(
            x,
            r_double,
            r_ij,
            r_jk,
            neighbors,
            neighbor_mask,
            neighbors_j,
            neighbors_k,
            triple_mask,
            d_ijk,
            f_double,
        )
        v = self.dense(v)

        return v


class SchNetTriple(nn.Module):
    """
    SchNet architecture for learning representations of atomistic systems.

    Attributes
    ----------
    n_atom_basis : int, optional, default=128
        number of features to describe atomic environments.
        This determines the size of each embedding vector;
        i.e. embeddings_dim.
    n_filters : int, optional, default=128
        number of filters used in continuous-filter convolution
    n_interactions : int, optional, default=3
        number of interaction blocks.
    cutoff : float, optional, default=5.0
        cutoff radius.
    n_gaussians : int, optional, default=25
        number of Gaussian functions used to expand atomic distances.
    max_zeta : int, default=1

    n_zeta : int, default=1

    normalize_filter : bool, optional, default=False
        if True, divide aggregated filter by number
        of neighbors over which convolution is applied.
    coupled_interactions : bool, optional, deault=False
        if True, share the weights
        across interaction blocks and filter-generating networks.
    return_intermediate : bool, optional, default=False
        if True, `forward` method also returns intermediate atomic representations
        after each interaction block is applied.
    max_z : int, optional, defalut=100
        maximum nuclear charge allowed in database. This determines
        the size of the dictionary of embedding; i.e. num_embeddings.
    cutoff_network : nn.Module, optional, default=schnetpack.nn.CosineCutoff
        cutoff layer.
    trainable_gaussians : bool, optional, default=False
        If True, widths and offset of Gaussian functions are adjusted
        during training process.
    distance_expansion_double : nn.Module, optional, default=None
        layer for expanding interatomic distances for double in a basis.
    distance_expansion_triple : nn.Module, optional, default=None
        layer for expanding interatomic distances for triple in a basis.
    charged_systems : bool, optional, default=False

    References
    ----------
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis=128,
        n_filters=128,
        n_interactions=3,
        cutoff=5.0,
        n_gaussians=25,
        max_zeta=1,
        n_zeta=1,
        normalize_filter=False,
        coupled_interactions=False,
        return_intermediate=False,
        max_z=100,
        cutoff_network=CosineCutoff,
        trainable_gaussians=False,
        distance_expansion_double=None,
        distance_expansion_triple=None,
        charged_systems=False,
    ):
        super(SchNetTriple, self).__init__()
        self.n_atom_basis = n_atom_basis
        # make a lookup table to store embeddings for each element (up to atomic number max_z)
        # each of which is a vector of size (N_batch * N_atoms * n_atom_basis)
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        # layer for computing interatomic distances
        self.double_ditances = AtomDistances()
        self.triple_distances = TriplesDistances()

        # layer for expanding interatomic distances in a basis
        if distance_expansion_double is None:
            self.distance_expansion_double = GaussianSmearing(
                0.0, cutoff, n_gaussians, centered=False, trainable=trainable_gaussians
            )
        else:
            self.distance_expansion_double = distance_expansion_double

        if distance_expansion_triple is None:
            self.distance_expansion_triple = GaussianSmearing(
                0.0, cutoff, n_gaussians, centered=False, trainable=trainable_gaussians
            )
        else:
            self.distance_expansion_triple = distance_expansion_triple

        # layer for extracting triple features
        self.triple_distribution = AngularDistribution(
            max_zeta=max_zeta,
            n_zeta=n_zeta,
        )
        # block for computing interaction
        if coupled_interactions:
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.ModuleList(
                [
                    SchNetInteractionTriple(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_gaussians,
                        n_zeta=n_zeta,
                        n_filters=n_filters,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                        normalize_filter=normalize_filter,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.ModuleList(
                [
                    SchNetInteractionTriple(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_gaussians,
                        n_zeta=n_zeta,
                        n_filters=n_filters,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                        normalize_filter=normalize_filter,
                    )
                    for _ in range(n_interactions)
                ]
            )

        # set attributes
        self.return_intermediate = return_intermediate
        self.charged_systems = charged_systems
        if charged_systems:
            self.charge = nn.Parameter(torch.Tensor(1, n_atom_basis))
            self.charge.data.normal_(0, 1.0 / n_atom_basis ** 0.5)

    def forward(self, inputs):
        """
        Compute atomic representations/embeddings.

        Parameters
        ----------
        inputs : dict of torch.Tensor
            SchNetPack dictionary of input tensors.

        Returns
        -------
        torch.Tensor
            atom-wise representation.
        list of torch.Tensor
            intermediate atom-wise representations,
            if return_intermediate=True was used.

        """
        # get tensors from input dictionary
        atomic_numbers = inputs[Properties.Z]
        positions = inputs[Properties.R]
        atom_mask = inputs[Properties.atom_mask]
        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]
        # triple property
        neighbors_j = inputs[Properties.neighbor_pairs_j]
        neighbors_k = inputs[Properties.neighbor_pairs_k]
        neighbor_offsets_j = inputs[Properties.neighbor_offsets_j]
        neighbor_offsets_k = inputs[Properties.neighbor_offsets_k]
        triple_mask = inputs[Properties.neighbor_pairs_mask]

        # get atom embeddings for the input atomic numbers
        x = self.embedding(atomic_numbers)

        if self.charged_systems and Properties.charge in inputs.keys():
            n_atoms = torch.sum(atom_mask, dim=1, keepdim=True)
            charge = inputs[Properties.charge] / n_atoms  # B
            charge = charge[:, None] * self.charge  # B x F
            x = x + charge

        # compute double distances of every atom to its neighbors
        r_double = self.double_ditances(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
        # compute tirple distances of every atom to its neighbors
        r_ijk = self.triple_distances(
            positions,
            neighbors_j,
            neighbors_k,
            offset_idx_j=neighbor_offsets_j,
            offset_idx_k=neighbor_offsets_k,
            cell=cell,
            cell_offsets=cell_offset,
            triple_mask=triple_mask,
        )
        # expand interatomic distances (for example, Gaussian smearing)
        f_double = self.distance_expansion_double(r_double)
        f_double = f_double * neighbors.unsqueeze(-1)
        f_ij = self.distance_expansion_triple(r_ijk[0])
        f_ij = f_ij * triple_mask.unsqueeze(-1)
        f_jk = self.distance_expansion_triple(r_ijk[2])
        f_jk = f_jk * triple_mask.unsqueeze(-1)
        # extract angular features
        d_ijk = self.triple_distribution(
            r_ijk[0],
            r_ijk[1],
            r_ijk[2],
            f_ij,
            f_jk,
            triple_mask,
        )
        # store intermediate representations
        if self.return_intermediate:
            xs = [x]
        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            v = interaction(
                x,
                r_double,
                r_ijk[0],
                r_ijk[2],
                neighbors,
                neighbor_mask,
                neighbors_j,
                neighbors_k,
                triple_mask,
                d_ijk=d_ijk,
                f_double=f_double,
            )
            x = x + v
            if self.return_intermediate:
                xs.append(x)

        if self.return_intermediate:
            return x, xs
        return x
