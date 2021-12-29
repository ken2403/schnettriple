import torch
from torch import nn
from schnetpack import Properties
from schnetpack.nn.activations import shifted_softplus
from schnetpack.nn.neighbors import AtomDistances

from schnettriple.nn.angular import AngularDistribution
from schnettriple.nn.base import Dense, FeatureWeighting
from schnettriple.nn.cfconv import CFConvDouble, CFConvTriple
from schnettriple.nn.cutoff import CosineCutoff
from schnettriple.nn.neighbors import TriplesDistances, GaussianFilter


__all__ = ["SchNetInteractionDouble", "SchNetInteractionTriple", "SchNetTriple"]


class SchNetInteractionDouble(nn.Module):
    """
    SchNetTriple interaction block for modeling double interactions of atomistic systems.

    Attributes
    ----------
    n_atom_basis : int
        number of features to describe atomic environments.
    n_gaussian_double : int
        number of gaussian filter of double distances.
    n_filters : int
        number of filters used in continuous-filter convolution.
    normalize_filter : bool, default=False
        if True, divide aggregated filter by number
        of neighbors over which convolution is applied.
    """

    def __init__(
        self,
        n_atom_basis,
        n_gaussian_double,
        n_filters,
        normalize_filter=False,
    ):
        super(SchNetInteractionDouble, self).__init__()
        # filter block used in interaction block
        filternet_double = nn.Sequential(
            Dense(n_gaussian_double, n_filters, activation=shifted_softplus),
            Dense(n_filters, n_filters, activation=None),
        )
        # interaction block
        self.cfconv_double = CFConvDouble(
            n_atom_basis,
            n_filters,
            n_atom_basis,
            filternet_double,
            activation=shifted_softplus,
            normalize_filter=normalize_filter,
        )
        # dense layer
        self.dense = Dense(n_atom_basis, n_atom_basis, bias=True, activation=None)

    def forward(
        self,
        x,
        f_double,
        neighbors,
        neighbor_mask,
    ):
        """
        Compute interaction output.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr_double :  Total number of neighbors of each atom

        Parameters
        ----------
        x : torch.Tensor
            input representation/embedding of atomic environments with
            (B x At x n_atom_basis) shape.
        f_double : torch.Tensor
            filtered distances of double pairs with
            (B x At x Nbr_double x n_gaussian_double) shape.
        neighbors : torch.Tensor
            indices of neighboring atoms with (B x At x Nbr_double) shape.
        neighbor_mask :
            mask to filter out non-existing neighbors introduced via padding.
            (B x At x Nbr_double) of shape.

        Returns
        -------
        v : torch.Tensor
            block output with (B x At x n_atom_basis) shape.
        """
        v = self.cfconv_double(
            x,
            f_double=f_double,
            neighbors=neighbors,
            neighbor_mask=neighbor_mask,
        )
        v = self.dense(v)
        x = x + v

        return x


class SchNetInteractionTriple(nn.Module):
    """
    SchNetTriple interaction block for modeling triple interactions of atomistic systems.

    Attributes
    ----------
    n_atom_basis : int
        number of features to describe atomic environments.
    n_gaussian_triple : int
        number of gaussian filter of triple distances.
    n_theta : int
        number of angular filter.
    n_filters : int
        number of filters used in continuous-filter convolution.
    normalize_filter : bool, default=False
        if True, divide aggregated filter by number
        of neighbors over which convolution is applied.
    """

    def __init__(
        self,
        n_atom_basis,
        n_gaussian_triple,
        n_theta,
        n_filters,
        normalize_filter=False,
    ):
        super(SchNetInteractionTriple, self).__init__()
        # fiter block for triple
        filternet_triple = nn.Sequential(
            Dense(n_gaussian_triple * n_theta, n_filters, activation=shifted_softplus),
            Dense(n_filters, n_filters, activation=None),
        )

        # interaction block
        self.cfconv_triple = CFConvTriple(
            n_atom_basis,
            n_filters,
            n_atom_basis,
            filternet_triple,
            activation=shifted_softplus,
            normalize_filter=normalize_filter,
        )
        # dense layer
        self.dense = Dense(n_atom_basis, n_atom_basis, bias=True, activation=None)

    def forward(
        self,
        x,
        triple_ijk,
        neighbors_j,
        neighbors_k,
        triple_mask,
    ):
        """
        Compute interaction output.

        B   :  Batch size
        At  :  Total number of atoms in the batch
        Nbr_triple :  Total number of triple neighbors of each atom

        Parameters
        ----------
        x : torch.Tensor
            input representation/embedding of atomic environments with
            (B x At x n_atom_basis) shape.
        triple_ijk : torch.tensor
            combination of filtered distances and angular filters with
            (B x At x Nbr_triple x n_angular) shape.
        neighbors_j : torch.Tensor
            indices of atom k in tirples with (B x At x Nbr_triple) shape.
        neighbors_k : torch.Tensor
            indices of atom k in tirples with (B x At x Nbr_triple) shape.
        triple_masks : torch.Tensor
            mask to filter out non-existing neighbors introduced via padding.
            (B x At x Nbr_triple) of shape.

        Returns
        -------
        v : torch.Tensor
            block output with (B x At x n_atom_basis) shape.
        """
        # continuous-filter convolution interaction block followed by Dense layer
        v = self.cfconv_triple(
            x,
            triple_ijk=triple_ijk,
            neighbors_j=neighbors_j,
            neighbors_k=neighbors_k,
            triple_mask=triple_mask,
        )
        v = self.dense(v)
        x = x + v

        return x


class SchNetTriple(nn.Module):
    """
    SchNetTriple architecture for learning representations of atomistic systems.

    Attributes
    ----------
    n_atom_basis : int, default=128
        number of features to describe atomic environments.
        This determines the size of each embedding vector; i.e. embeddings_dim.
    n_filters : int, default=128
        number of filters used in continuous-filter convolution
    n_interactions : int, default=3
        number of interaction blocks.
    cutoff : float, default=6.0
        cutoff radius.
    n_gaussian_double : int, default=25
        number of gaussian filter of double distances.
    n_gaussian_triple : int, default=25
        number of gaussian filter of triple distances.
    n_theta : int, default=10
        number of angular filter.
    zeta : float, default=8.0
        zeta value of angular filter.
    normalize_filter : bool, default=False
        if True, divide aggregated filter by number
        of neighbors over which convolution is applied.
    coupled_interactions : bool, deault=False
        if True, share the weights across interaction blocks
        and filter-generating networks.
    return_intermid : bool, default=False
        if True, `forward` method also returns intermediate atomic representations
        after each interaction block is applied.
    max_z : int, defalut=100
        maximum nuclear charge allowed in database. This determines
        the size of the dictionary of embedding; i.e. num_embeddings.
    cutoff_network : nn.Module, default=schnetpack.nn.CosineCutoff
        cutoff layer.
    charged_systems : bool, default=False
        if True, there is the charge in the system.

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
        cutoff=6.0,
        n_gaussian_double=25,
        n_gaussian_triple=25,
        trainable_gaussian=False,
        n_theta=10,
        zeta=8.0,
        normalize_filter=False,
        coupled_interactions=False,
        return_intermid=False,
        max_z=100,
        cutoff_network=CosineCutoff,
        charged_systems=False,
    ):
        super(SchNetTriple, self).__init__()
        self.n_atom_basis = n_atom_basis
        # make a lookup table to store embeddings for each element (up to atomic number max_z)
        # each of which is a vector of size (B x At x n_atom_basis)
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        # layer for computing interatomic distances
        self.ditances_double = AtomDistances()
        self.distances_triple = TriplesDistances()

        # layer for expanding interatomic distances in a basis
        self.radial_double = GaussianFilter(
            start=0.0,
            stop=cutoff - 0.5,
            n_gaussian=n_gaussian_double,
            centered=False,
            trainable=trainable_gaussian,
        )
        self.radial_triple = GaussianFilter(
            start=0.0,
            stop=cutoff - 0.5,
            n_gaussian=n_gaussian_triple,
            centered=False,
            trainable=trainable_gaussian,
        )
        # cutoff layer
        if cutoff_network is not None:
            self.cutoffnet = cutoff_network(cutoff)
        else:
            self.cutoffnet = None

        # layer for extracting triple features
        self.triple_distribution = AngularDistribution(n_theta=n_theta, zeta=zeta)
        # block for computing interaction
        if coupled_interactions:
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions_double = nn.ModuleList(
                [
                    SchNetInteractionDouble(
                        n_atom_basis=n_atom_basis,
                        n_gaussian_double=n_gaussian_double,
                        n_filters=n_filters,
                        normalize_filter=normalize_filter,
                    )
                ]
                * n_interactions
            )
            self.interactions_triple = nn.ModuleList(
                [
                    SchNetInteractionTriple(
                        n_atom_basis=n_atom_basis,
                        n_gaussian_triple=n_gaussian_triple,
                        n_theta=n_theta,
                        n_filters=n_filters,
                        normalize_filter=normalize_filter,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.interactions_double = nn.ModuleList(
                [
                    SchNetInteractionDouble(
                        n_atom_basis=n_atom_basis,
                        n_gaussian_double=n_gaussian_double,
                        n_filters=n_filters,
                        normalize_filter=normalize_filter,
                    )
                    for _ in range(n_interactions)
                ]
            )

            self.interactions_triple = nn.ModuleList(
                [
                    SchNetInteractionTriple(
                        n_atom_basis=n_atom_basis,
                        n_gaussian_triple=n_gaussian_triple,
                        n_theta=n_theta,
                        n_filters=n_filters,
                        normalize_filter=normalize_filter,
                    )
                    for _ in range(n_interactions)
                ]
            )

        # set attributes
        self.return_intermid = return_intermid
        self.charged_systems = charged_systems
        if charged_systems:
            self.charge = nn.Parameter(torch.Tensor(1, n_atom_basis))
            self.charge.data.normal_(0, 1.0 / n_atom_basis ** 0.5)

    def forward(self, inputs):
        """
        Compute atomic representations/embeddings.

        B   :  Batch size
        At  :  Total number of atoms in the batch

        Parameters
        ----------
        inputs : dict of torch.Tensor
            SchNetPack dictionary of input tensors.

        Returns
        -------
        x : torch.Tensor
            atom-wise representation. (B x At x n_atom_basis) of shape.
        xs : list of torch.Tensor
            intermediate atom-wise representations, if return_intermid=True was used.
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
        r_double = self.ditances_double(
            positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask
        )
        # compute tirple distances of every atom to its neighbors
        r_ijk = self.distances_triple(
            positions,
            neighbors_j,
            neighbors_k,
            offset_idx_j=neighbor_offsets_j,
            offset_idx_k=neighbor_offsets_k,
            cell=cell,
            cell_offsets=cell_offset,
        )

        # expand interatomic distances (for example, GaussianFilter)
        f_double = self.radial_double(r_double)
        f_double = f_double * neighbor_mask.unsqueeze(-1)
        f_ij = self.radial_triple(r_ijk[0])
        f_ik = self.radial_triple(r_ijk[1])
        if self.cutoffnet is not None:
            C_double = self.cutoffnet(r_double).unsqueeze(-1)
            f_double = f_double * C_double
            C_ij = self.cutoffnet(r_ijk[0]).unsqueeze(-1)
            C_ik = self.cutoffnet(r_ijk[1]).unsqueeze(-1)
            f_ij = f_ij * C_ij
            f_ik = f_ik * C_ik

        # extract angular features
        triple_ijk = self.triple_distribution(
            r_ijk[0],
            r_ijk[1],
            r_ijk[2],
            f_ij,
            f_ik,
            triple_mask,
        )

        x_double = x
        x_triple = x
        # store intermediate representations
        if self.return_intermid:
            xs = [(x_double, x_triple)]
        # compute interaction block to update atomic embeddings
        for interaction_double, interaction_triple in zip(
            self.interactions_double, self.interactions_triple
        ):
            x_double = interaction_double(
                x=x_double,
                f_double=f_double,
                neighbors=neighbors,
                neighbor_mask=neighbor_mask,
            )
            x_triple = interaction_triple(
                x=x_triple,
                triple_ijk=triple_ijk,
                neighbors_j=neighbors_j,
                neighbors_k=neighbors_k,
                triple_mask=triple_mask,
            )
            if self.return_intermid:
                xs.append((x_double, x_triple))
        x = torch.cat((x_double, x_triple), dim=-1)
        if self.return_intermid:
            return x, xs
        return x
